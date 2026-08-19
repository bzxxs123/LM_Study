import torch
import numpy as np
import numpy.typing as npt
import os
from typing import BinaryIO,IO,Union


def get_batch(
    x:npt.NDArray,
    batch_size:int,
    context_length:int,
    device:str
)->tuple[torch.Tensor,torch.Tensor]:
    """
    从 token 序列 x 中随机采样出一个 batch 的输入和目标序列。

    Args:
        x: 形状 (n,) 的 numpy 整数数组，完整的 token id 序列
        batch_size: 一个 batch 里有多少个样本
        context_length: 每个样本的序列长度 m
        device: 目标设备字符串，如 'cpu' 或 'cuda:0'

    Returns:
        (inputs, targets)：两个形状均为 (batch_size, context_length) 的 LongTensor,
        分别放在指定 device 上
    """
    n=len(x)
    # 1.计算合法的起始位置范围，随机采样batch_size个起始索引
    low=0
    high=n-context_length-1
    starts=np.random.randint(low,high,size=batch_size)

    # 2.根据每个起始位置，得到输入片段和目标片段

    inputs_lists=[x[start:start+context_length] for start in starts]
    outputs_lists=[x[start+1:start+context_length+1] for start in starts]
    # 用 np.stack 堆叠为 (batch_size, context_length) 的二维 ndarray
    inputs=np.stack(inputs_lists)
    outputs=np.stack(outputs_lists)

    # 3.转为tensor，要求索引张量的 dtype 是 int64（也就是 long）
    inputs=torch.from_numpy(inputs).long().to(device) 
    outputs=torch.from_numpy(outputs).long().to(device)

    return inputs,outputs

def save_checkpoint(
    model:torch.nn.Module,
    optimizer:torch.optim.Optimizer,
    iteration:int,
    out:Union[str,os.PathLike,BinaryIO,IO[bytes]]
)->None:
    """
    保存训练状态（模型参数、优化器状态、当前迭代步数）到 out。
    """
    #把 model.state_dict()、optimizer.state_dict()、iteration  打包成一个字典，用 torch.save 写入 out
    checkpoint={
        "model_state":model.state_dict(),
        "optimizer_state":optimizer.state_dict(),
        "iteration":iteration
    }
    torch.save(checkpoint,out)

def load_checkpoint(
    src:Union[str,os.PathLike,BinaryIO,IO[bytes]],
    model:torch.nn.Module,
    #optimizer:torch.optim.Optimizer
)->int:
    """
    从 src 加载 checkpoint,恢复 model 和 optimizer 的状态，
    返回保存时的 iteration 数。(iteration是不可变对象,值传递，无法直接修改)
    """
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint["model_state"])
    #optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    return checkpoint["iteration"]