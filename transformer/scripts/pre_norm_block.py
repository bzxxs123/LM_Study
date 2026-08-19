import torch
import torch.nn as nn
from transformer.basic_blocks import Linear
from einops import einsum,rearrange
import math

class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()

        # 1.保存eps
        self.eps=eps

        # 2.创建可学习的缩放参数g，shape为（d_model, ),初始化全为1,并注册为parameter
        self.g=nn.Parameter(torch.ones(
            d_model,
            device=device,
            dtype=dtype
        ))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        # 3.记录原本的dtype
        in_dtype=x.dtype

        # 4.将x转为float32
        x=x.to(torch.float32)

        # 5.计算RMS(a),dim=-1:按最后一维度d_model计算均值
        # keepdim=True ,否则维度会被压缩，广播除法会出错
        rms=torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)

        # 6.归一化计算
        result=x/rms*self.g

        return result.to(in_dtype)


"""
d_ff_raw = (8/3) * d_model
d_ff = round(d_ff_raw / 64) * 64   # 四舍五入到最近的64倍数
"""

class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device=None,dtype=None):
        super().__init__()
        '''
        Position-Wise Feed-Forward Network(逐位置前馈网络)
        d_ff : SwiGLU"中间隐藏层"的维度
        '''
        # 1.创建三个Linear层
        # self.w1: d_model -> d_ff (SiLU)
        # self.w2: d_ff -> d_model (投影回d_model)
        # self.w3  d_model -> d_ff(门控另一路)
        self.w1=Linear(in_features=d_model,out_features=d_ff,device=device,dtype=dtype)
        self.w2=Linear(in_features=d_ff,out_features=d_model,device=device,dtype=dtype)
        self.w3=Linear(in_features=d_model,out_features=d_ff,device=device,dtype=dtype)


    def forward(self,x:torch.Tensor)->torch.Tensor:
        # 2.计算SiLU(w1 x),SiLU(z) = z * sigmoid(z)
        z=self.w1(x)
        silu_out=z*torch.sigmoid(z)

        # 3.计算w3 x
        w3_out=self.w3(x)

        # 4.逐元素相乘
        gated=silu_out*w3_out

        # 5.使用W2投影回去
        return self.w2(gated)
    

"""
旋转位置编码
"""
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None,dtype=None):
        super().__init__()
        """
        预计算cos,sin
        """
        self.d_k=d_k
        # 1.计算频率freqs，形状为(d_k//2,)
        k_range = torch.arange(0, d_k, 2, device=device)   # [0, 2, 4, ..., d_k-2]
        freqs = theta**(-k_range.float() / d_k)

        # 2.生成位置序列positions,形状为(max_seq_len,)
        positions=torch.arange(max_seq_len,device=device)

        # 3.外积得到角度矩阵angles,形状(max_seq_len,d_k//2)
        angles=einsum(positions,freqs,"seq,half_d -> seq half_d")

        # 4.计算cos和sin，注册为buffer
        cos_cache=torch.cos(angles)
        sin_cache=torch.sin(angles)
        # persistent 是否保存到 checkpoint
        self.register_buffer("cos_cache",cos_cache,persistent=False)
        self.register_buffer("sin_cache",sin_cache,persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        # 5. 使用token_positons查表，取出对应的cos，sin
        # token_positions 形状 (batch, seq_len)
        cos=self.cos_cache[token_positions]
        sin=self.sin_cache[token_positions]
        # cos sin 形状: (batch, seq_len, d_k//2)
      
        sin=sin.unsqueeze(1)
        cos=cos.unsqueeze(1)
                
        # 6.将x分割为两半
        x1=x[...,:self.d_k//2]
        x2=x[...,self.d_k//2:]
      
     
        # 7.应用旋转公式
       
        x1_rotated=x1*cos-x2*sin
        x2_rotated=x1*sin+x2*cos

        # 8.拼接
        return torch.cat([x1_rotated,x2_rotated],dim=-1)


def softmax(x:torch.Tensor,dim:int)->torch.Tensor:
    # 1.沿着dim这一维度求最大值
    max_value=x.max(dim=dim,keepdim=True).values

    # 2.全都减去最大值
    x_shifted=x-max_value

    # 3.计算exp
    exp_x=torch.exp(x_shifted)

    # 4.沿dim维度求和
    sum_exp=exp_x.sum(dim=dim,keepdim=True)
    return exp_x/sum_exp


def scaled_dot_product_attention(
        Q:torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask:torch.Tensor=None
)->torch.Tensor:
    """
    计算缩放点积注意力。
    
    参数形状:
    q, k: (batch_size, ..., seq_len, d_k)
    v: (batch_size, ..., seq_len, d_v)
    mask: (seq_len, seq_len) Boolean 类型,True 表示可以看,False 表示屏蔽
    """
    # 提取 d_k (特征维度)，即最后一个维度的大小
    d_k=Q.shape[-1]

    # 1.计算 QK^T / sqrt(d_k)
    scores=einsum(Q,K,"... q d_k,... k d_k-> ... q k")/math.sqrt(d_k)

    # 2.如果有mask，将mask==flase的位置填为-inf
    if mask is not None:
        scores=scores.masked_fill(mask==False,value=float('-inf'))

    # 3.对scores的最后一维做softmax，得到注意力权重
    attn_weight=softmax(scores,dim=-1)

    # 4.与V加权求和
    output=einsum(attn_weight,V,"... q k,... k d_v -> ... q d_v")

    return output


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,max_seq_len:int,theta:float,device=None,dtype=None):
        """
        初始化多头注意力机制。
        参数:
            d_model: 输入特征的维度 (例如 512)
            num_heads: 头的数量 (例如 8)
            max_seq_len:最大序列长度
            theta:用于rope
        """
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads # 每个头的维度

        # 1. 创建四个 Linear 层：w_q, w_k, w_v, w_o ,输入输出维度都是 d_model -> d_model
        self.w_q=Linear(d_model,d_model,device,dtype)
        self.w_k=Linear(d_model,d_model,device,dtype)
        self.w_v=Linear(d_model,d_model,device,dtype)
        self.w_o=Linear(d_model,d_model,device,dtype)

        # 2.创建RoPE模块
        self.rope=RotaryPositionalEmbedding(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len,device=device,dtype=dtype)


    def forward(self, x:torch.Tensor,token_positions:torch.Tensor=None)->torch.Tensor:
        """
        x 形状: (batch_size, seq_len, d_model)
        token_positions 形状: (batch_size, seq_len)
        """
        batch_size,seq_len,_=x.shape
        # 3.线性投影得到Q K V （形状: batch, seq, d_model)
        Q=self.w_q(x)
        K=self.w_k(x)
        V=self.w_v(x)

        # 4.拆分多头
        Q=rearrange(Q,"batch seq_len (h d_k) -> batch h seq_len d_k",h=self.num_heads)

        K=rearrange(K,"batch seq_len (h d_k) -> batch h seq_len d_k",h=self.num_heads)

        V=rearrange(V,"batch seq_len (h d_k) -> batch h seq_len d_k",h=self.num_heads)

        # 5.对Q K使用RoPE，嵌入位置信息
        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # 6.构造casual mask 形状：(seq_len,seq_len)
        # 生成下三角矩阵，左下角全是 True，右上角全是 False
        casual_mask=torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=x.device))

        # 7.计算缩放点积注意力
        attn_out=scaled_dot_product_attention(Q,K,V,casual_mask)

        # 8.拼接多头
        # (batch_size, num_heads, seq_len, d_v) -> (batch_size, seq_len, num_heads, d_v)
        attn_out=rearrange(attn_out,"batch h seq_len d_v -> batch seq_len (h d_v)")

        # 9.线性投影
        return self.w_o(attn_out)


class TransformerBlock(nn.Module):
    """
    Transformer Block(把 attention 和 FFN 拼成一个block,带残差连接)
    """
    def __init__(self,d_model:int,num_heads:int,d_ff:int,max_seq_len:int,theta:float,device=None,dtype=None):
        super().__init__()
        # 1.创建两个RMSNorm（一个用于attention前，一个用于FFN前)
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)

        # 2.创建attention子层
        self.attn=CausalMultiHeadSelfAttention(d_model,num_heads,max_seq_len,theta,device=device, dtype=dtype)

        # 3.创建FFN子层(SwiGLU)
        self.ffn=SwiGLU(d_model,d_ff,device=device, dtype=dtype)

    def forward(self,x:torch.Tensor,token_position:torch.Tensor=None)->torch.Tensor:
        # 第一个子层(attention) + 残差
        z = x + self.attn(self.norm1(x),token_position)

        # 第二个子层（FFN） + 残差
        y = z + self.ffn(self.norm2(z))

        return y







        
