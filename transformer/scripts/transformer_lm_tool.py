import torch
import torch.nn.functional as F
import math
from typing import Iterable

def cross_entropy(logtis:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
    """
    数值稳定的交叉熵损失函数。
    
    参数:
        logits: (..., vocab_size) 预测未归一化得分
        targets: (...) 目标 token 索引，类型为 long
    返回:
        标量 Tensor (标量平均损失)
    """
    # 1.提取logtis的vocab_size维度的最大值
    max_logtis=logtis.max(dim=-1,keepdim=True)[0]  # max返回（value，indice）

    # 2.计算 log(sum(exp(logits - max)))
    logtis_shifted=logtis-max_logtis
    exp_logtis=torch.exp(logtis_shifted)
    sum_exp=exp_logtis.sum(dim=-1,keepdim=True)
    log_sum_exp=max_logtis+torch.log(sum_exp)  # 形状: (..., 1)
    log_sum_exp=log_sum_exp.squeeze(-1)        # 形状: (...)

    # 3. 提取 targets 对应的 logits[targets]
    # 利用 gather 提取指定索引的值,torch.gather 要求 index 张量的维度数（ndim）必须和 input（这里是 logits）完全一致
    '''
    假设当前批次大小为 2,序列长度为 2,词表大小为 3:

    logits 形状：(2, 2, 3)(3 维张量)

    targets 形状：(2, 2)(2 维张量),所以要将targets增加一个维度

    最后删掉这个维度,变成(2 2)
    '''
    targets_logtis=torch.gather(logtis,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)

    # 4.计算每个元素的损失
    loss_per_token=log_sum_exp-targets_logtis

    # 5.求序列的平均损失,返回的必须是一个全局标量（0 维 Tensor，形如 tensor(2.3456)）,所以不用dim=-1,keepdim=True
    loss_mean=loss_per_token.mean() # 对全局求平均

    return loss_mean

class SGD(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3):
        if lr<0:
            raise ValueError(f"Invalid Learning Rate:{lr}")
        defaults={"lr":lr}
        super().__init__(params,defaults)

    def step(self,closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:  # 遍历每个参数组
            lr=group["lr"]               # 拿到这组的学习率
            for p in group["params"]:    # 遍历这组里的每个参数张量
                if p.grad is None:
                    continue
                state=self.state[p]      # 获取该参数张量的状态字典
                t=state.get("t",0)       # 读取当前迭代步数，没有就默认为 0
                grad=p.grad.data         # 获取梯度
                p.data-=lr/math.sqrt(t+1)*grad  # 更新参数(按步数)
                state["t"]=t+1           # 更新步数
        return loss
            

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9,0.999),
        eps=1e-8,
        weight_decay=0.01
    ):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults={"lr":lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params,defaults)

    def step(self,closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            # 获取default字典中的相关参数
            lr=group["lr"]
            beta1,beta2=group["betas"]
            eps=group["eps"]
            weight_decay=group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad=p.grad.data # 获取梯度值
                state=self.state[p] # 获取该参数状态信息
                # 初始化状态
                #  m = 全零张量（形状同 p.data）, v = 全零张量（形状同 p.data）
                if len(state)==0:
                    state["t"]=1
                    state["m"]=torch.zeros_like(p.data)
                    state["v"]=torch.zeros_like(p.data)

                t=state["t"]
                m=state["m"]
                v=state["v"]

                # 更新一阶矩m
                m=beta1*m+(1-beta1)*grad
                # 更新二阶矩v
                v=beta2*v+(1-beta2)*grad**2

                # 计算矫正后的学习率
                lr_t=lr*(math.sqrt(1-beta2**t)/(1-beta1**t))

                # 权重衰减
                p.data-=lr*weight_decay*p.data

                # 进行参数更新
                p.data-=lr_t*m/(torch.sqrt(v)+eps)

                # 更新m v t
                state["m"]=m
                state["v"]=v
                state["t"]=t+1

        return loss


def get_lr_cosine_schedule(
    t:int,
    alpha_max:float,
    alpha_min:float,
    T_w:int,
    T_c:int
)->float:
    """
    计算 cosine annealing with warmup 学习率调度在第 t 步的学习率。

    Args:
        t: 当前迭代步数
        alpha_max: 最大(warmup 结束时的)学习率
        alpha_min: 最小(退火结束后的)学习率
        T_w: warmup 步数
        T_c: 退火结束的步数

    Returns:
        第 t 步应使用的学习率
    """
    # 1.预热
    if t<T_w:
        alpha_t=alpha_max*t/T_w
    # 2.余弦退火
    elif t>T_w and t<T_c:
       alpha_t = alpha_min + 0.5 * (1 + (math.cos((t-T_w)/(T_c-T_w)* math.pi))) * (alpha_max - alpha_min)
    # 3.退火结束
    else:
        alpha_t=alpha_min

    return alpha_t

def gradient_clipping(parameters:Iterable[torch.nn.Parameter],max_L2_norm:float)->None:
    """
    对一组参数的梯度做整体 L2 范数裁剪(原地修改 .grad)。

    Args:
        parameters: 参数的可迭代对象
        max_l2_norm: 允许的最大梯度 L2 范数
    """

    eps=1e-6

    # 1.筛选出有梯度的参数
    grads=[p.grad for p in parameters if p.grad is not None]


    if len(grads)==0:
        return 

    # 2.计算所有梯度合起来的L2范数
    total_norm=torch.sqrt(sum((g**2).sum() for g in grads))

    # 3.判断是否超过阈值
    if total_norm>=max_L2_norm:
       scale=max_L2_norm/(total_norm+eps)
       for g in grads:
           g.mul_(scale) # 原地操作,直接在原有张量的内存上做修改


