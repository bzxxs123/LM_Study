import torch
import argparse
import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from data_loader import save_checkpoint, load_checkpoint, get_batch
from transformer_lm_tool import AdamW,cross_entropy,get_lr_cosine_schedule,gradient_clipping
from transformer_LM import TransformLM

def parse_args():
    parser=argparse.ArgumentParser(description="Transfomer 语言模型训练脚本")

    # 数据集路径配置
    parser.add_argument("--train_data_path", type=str, default="data/train.bin", help="训练集路径")
    parser.add_argument("--val_data_path", type=str, default="data/val.bin", help="验证集路径")
    parser.add_argument("--data_dtype", type=str, default="uint16", choices=["uint16", "int32", "int64"], help="memmap 数据类型")

    # 模型架构超参数
    parser.add_argument("--model_dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"], help="模型计算精度")
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--context_length", type=int, default=256, help="上下文序列长度")
    parser.add_argument("--d_model", type=int, default=256, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer Block 层数")
    parser.add_argument("--num_heads", type=int, default=4, help="注意力头数")
    parser.add_argument("--d_ff", type=int, default=None, help="FFN 隐藏维度 (默认 8/3 * d_model)")
    parser.add_argument("--theta", type=float, default=10000.0, help="RoPE 的基础频率基数")

    # 优化器与训练超参数
    parser.add_argument("--batch_size", type=int, default=64, help="单步训练批次大小")
    parser.add_argument("--max_lr", type=float, default=6e-4, help="最大学习率")
    parser.add_argument("--min_lr", type=float, default=6e-5, help="最小学习率")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="学习率预热步数")
    parser.add_argument("--total_iters", type=int, default=10000,help="退火结束步数")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减率")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="AdamW的beta1, beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值 (0 表示不裁剪)")

    # 日志、评估与 Checkpoint
    parser.add_argument("--eval_interval", type=int, default=1000, help="每隔多少步评估一次验证集")
    parser.add_argument("--eval_iters", type=int, default=20, help="评估时采样的批次数")
    parser.add_argument("--log_interval", type=int, default=10, help="打印日志的步数间隔")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="模型保存目录")
    parser.add_argument("--save_interval", type=int, default=1000, help="保存 checkpoint 的步数间隔")
    parser.add_argument("--resume_from", type=str, default=None, help="恢复训练的 checkpoint 路径")

    # 硬件环境
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="训练设备")

    return parser.parse_args()

def evaluate(model,val_data,args):
    """
    在验证集上采样 args.eval_iters 个 batch,计算平均 loss。
    """

    model.eval()
    losses=[]
    with torch.no_grad():
        for _ in range(args.eval_iters):
            inputs,targets=get_batch(val_data,args.batch_size,context_length=args.context_length,device=args.device)
            logtis=model(inputs)
            loss=cross_entropy(logtis,targets)

            losses.append(loss.item())

    model.train()
    return sum(losses)/len(losses)

def moving_average(values: list[float], window_size: int = 50) -> np.ndarray:
    """计算滑动平均，自动自适应小样本长度"""
    arr = np.array(values, dtype=np.float32)
    if len(arr) < 5:
        return arr
    actual_window = min(window_size, max(3, len(arr) // 4))
    kernel = np.ones(actual_window) / actual_window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # 边缘向前补齐，保证维度与原始序列一致
    pad = np.full(actual_window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])

def plot_loss_metrics(
    train_iters: list[int],
    train_losses: list[float],
    val_iters: list[int] | None = None,
    val_losses: list[float] | None = None,
    lrs: list[float] | None = None,
    save_path: str = "loss_curve.png",
    smooth_window: int = 50,
):
    """绘制带滑动平均与双子图的训练指标曲线

    Args:
        train_iters: 训练 step 列表
        train_losses: 训练 loss 列表
        val_iters: 验证 step 列表 (可选)
        val_losses: 验证 loss 列表 (可选)
        lrs: 学习率列表 (可选)
        save_path: 图片保存路径
        smooth_window: 滑动平均窗口大小
    """
    if len(train_losses) == 0:
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ---------------- 1. 上子图：Train Loss ----------------
    t_iters = np.array(train_iters)
    t_losses = np.array(train_losses)

    # 原始毛刺（透明浅色）
    ax1.plot(t_iters, t_losses, color="#1f77b4", alpha=0.25, label="Train Loss (Raw)")

    # 平滑主线（深色加粗）
    t_smooth = moving_average(train_losses, window_size=smooth_window)
    ax1.plot(
        t_iters,
        t_smooth,
        color="#1f77b4",
        linewidth=2,
        label=f"Train Loss (Smoothed w={smooth_window})",
    )

    ax1.set_ylabel("Training Loss")
    ax1.set_title("Transformer Training Dynamics", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # ---------------- 2. 下子图：Val Loss + LR ----------------
    has_val = val_losses is not None and len(val_losses) > 0
    if has_val:
        ax2.plot(
            val_iters,
            val_losses,
            color="#d62728",
            marker="o",
            markersize=4,
            linewidth=1.8,
            label="Validation Loss",
        )
        ax2.set_ylabel("Validation Loss", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")

    # 右侧副坐标轴展示 LR 调度
    has_lrs = lrs is not None and len(lrs) == len(train_iters)
    if has_lrs:
        ax2_lr = ax2.twinx()
        ax2_lr.plot(
            t_iters,
            lrs,
            color="dimgray",
            linestyle=":",
            alpha=0.7,
            label="Learning Rate",
        )
        ax2_lr.set_ylabel("LR", color="dimgray")
        ax2_lr.tick_params(axis="y", labelcolor="dimgray")

    ax2.set_xlabel("Iteration Steps")
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 合并下子图图例
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    if has_lrs:
        lines_2, labels_2 = ax2_lr.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    elif has_val:
        ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # 1.加载数据(memmap)
    train_data=np.memmap(args.train_data_path,dtype=args.data_dtype,mode="r")
    val_data=np.memmap(args.val_data_path,dtype=args.data_dtype,mode="r") if args.val_data_path else None

    # 2.构建模型（ TransformerLM 类，传入相应超参数）
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    model=TransformLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=args.device,
        dtype=dtype_map[args.model_dtype]
    )
    model=model.to(args.device)

    # 3.构建优化器（ AdamW 类）
    optimizer=AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    # 4.如果指定了resume，恢复状态
    start_iter=0
    if args.resume_from is not None:
        start_iter=load_checkpoint(args.resume_from,model,optimizer)

    start_time=time.time()
    start_time1=time.time()
    train_iters, train_losses = [], []
    val_iters, val_losses = [], []
    # 5.训练主循环
    for iteration in range(start_iter,args.total_iters):
        # 5.1 计算这一步的学习率，写回到optimizer
        lr=get_lr_cosine_schedule(iteration,args.max_lr,args.min_lr,args.warmup_iters,args.total_iters)

        for group in optimizer.param_groups:
            group["lr"]=lr

        # 5.2 采样一个训练batch的数据
        inputs,targets=get_batch(x=train_data,batch_size=args.batch_size,context_length=args.context_length,device=args.device)

        # 5.3 前向计算(开启混合精度)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):

            optimizer.zero_grad() # 梯度清零
            logtis=model(inputs)

            # 5.4 计算loss
            loss=cross_entropy(logtis,targets)

            # 5.5 反向传播
            loss.backward()

        # 5.6 梯度裁剪
        if args.grad_clip>0:
            gradient_clipping(model.parameters(),args.grad_clip)

        # 5.7 更新参数
        optimizer.step()

        
        # 5.8 打印日志
        if iteration %args.log_interval==0:
            elapsed=time.time()-start_time1
            # 计算这 log_interval 步总共处理的 token 数
            tokens_processed = args.batch_size * args.context_length * args.log_interval
            tokens_per_sec = tokens_processed / elapsed

            # 记录训练数据（使用 .item() 避免显存泄露）
            train_iters.append(iteration)
            train_losses.append(loss.item())
            
            print(f"iter {iteration} | train_loss {loss.item():.4f} | lr {lr:.6f} | time {elapsed:.1f}s,吞吐: {tokens_per_sec:,.0f} tok/s")
            start_time1=time.time()

        # 5.9 定期评估验证集
        if val_data is not None and iteration % args.eval_interval==0:
            val_loss=evaluate(model,val_data,args)
            elapsed=time.time()-start_time

            val_iters.append(iteration)
            val_losses.append(val_loss)
            print(f"iter {iteration} | val_loss {val_loss: .4f} | time {elapsed:.1f}")

        # 5.10 定期保存checkpoint
        if iteration % args.save_interval == 0 and iteration>0:
            ckpt_path=f"{args.checkpoint_dir}/ckpt_{iteration}.pt"
            save_checkpoint(model,optimizer,iteration,ckpt_path)

    # 6.训练结束，保存最终checkpoint
    save_checkpoint(model,optimizer,args.total_iters,f"{args.checkpoint_dir}/ckpt_final.pt")

    # 7. 绘制并保存 Loss 曲线
   
  # 训练主循环结束后一键生成：
    plot_loss_metrics(
        train_iters=train_iters,
        train_losses=train_losses,
        val_iters=val_iters,
        val_losses=val_losses,
        # lrs=lrs,
        save_path=f"{args.checkpoint_dir}/loss_curve.png",
        smooth_window=50,
    )

        

if __name__=="__main__":


    args=parse_args()
    if args.d_ff is None:
        args.d_ff=int(round(8/3*args.d_model/64))*64

    print(args.device)
    

    train(args)
