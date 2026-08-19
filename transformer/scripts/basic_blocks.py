import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device=None,dtype=None):
        super().__init__()
        # 1.计算初始化所用标准差
        std=math.sqrt(2/(in_features+out_features))
        # 2.创建形状为(out_features,in_features)的空tensor
        empty_weight=torch.empty(
            (out_features,in_features),
            device=device,
            dtype=dtype
        )
        # 3.使用 trunc_normal_ 进行初始化
        # 截断范围是 [-3 * sigma, 3 * sigma]
        nn.init.trunc_normal_(
            empty_weight,
            mean=0.0,
            std=std,
            a=-3.0*std,
            b=3.0*std
        )
        # 4.将权重打包成nn.Parameter，以便 PyTorch 追踪它的梯度
        self.weight=nn.Parameter(empty_weight)


    def forward(self,x:torch.Tensor)->torch.Tensor:
        # 使用 einsum 进行线性变换
        # x 的形状：任意前置维度 + in_features
        # weight 的形状：out_features + in_features
        # 输出的形状：相同的前置维度 + out_features
        return einsum(x,self.weight,"... in_features, out_features in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,device=None,dtype=None):
        super().__init__()
        """
        Embedding 不是矩阵乘法，而是查表。有一张形状为 (vocab_size, embedding_dim) 的表（矩阵），每一行对应一个 token ID 的向量表示。输入是一串整数 ID,输出就是把每个 ID 对应的那一行"取出来"拼在一起
        """

        # 1.创建形状为(num_embeddings,embedding_dim)的嵌入矩阵  （num_embeddings为词表大小）
        empty_weight=torch.empty(
            (num_embeddings,embedding_dim),
            device=device,
            dtype=dtype
        )
        # 2.初始化矩阵
        nn.init.trunc_normal_(
            empty_weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )
        # 3.注册为Parameter
        self.weight=nn.Parameter(empty_weight)


    def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
        # 利用 PyTorch 的高级索引进行查表提取，得到id对应的向量表达
        # 输入 token_ids 形状: (batch_size, sequence_length)
        # 输出形状: (batch_size, sequence_length, embedding_dim)
        return self.weight[token_ids]

