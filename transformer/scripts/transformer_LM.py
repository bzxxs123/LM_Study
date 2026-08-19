import torch
import torch.nn as nn
from transformer.basic_blocks import Linear,Embedding
from transformer.pre_norm_block import TransformerBlock,RMSNorm


class TransformLM(nn.Module):
    """
    完整的transformer网络
    **结构**

    输入 token IDs (batch, seq_len)
        ↓
    Token Embedding
        ↓
    Transformer Block x num_layers  (循环堆叠)
        ↓
    最终的 RMSNorm
        ↓
    Linear (LM head,投影到词表大小vocab_size)
        ↓
    输出 logits (batch, seq_len, vocab_size)
    """
    def __init__(
        self,
        vocab_size:int,
        context_length:int, # max_seq_len
        d_model:int,
        num_layers:int, # 层数
        num_heads:int,
        d_ff:int,
        theta:float = 10000.0,
        device=None,
        dtype=None
    ):
        super().__init__()

        self.context_length=context_length

        # 1.创建token embedding
        self.token_embedding=Embedding(num_embeddings=vocab_size,embedding_dim=d_model,device=device,dtype=dtype)

        # 2.创建num_layers个transform block
        self.blocks=nn.ModuleList([
            TransformerBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,max_seq_len=context_length,theta=theta,device=device,dtype=dtype) for _ in range(num_layers)
        ])

        # 3.创建最后的一个RMSNorm
        self.norm_final=RMSNorm(d_model=d_model,device=device,dtype=dtype)

        # 4.创建线性层 LM_head 将d_model 投影回 vocab_size
        self.lm_head=Linear(in_features=d_model,out_features=vocab_size,device=device,dtype=dtype)

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:

        # 输入为token_ids ，形状 (batch, seq_len)
        batch_size=token_ids.shape[0]
        seq_len=token_ids.shape[-1]

        # 5.生成位置序列
        positions = torch.arange(seq_len, device=token_ids.device)   # (seq_len,)
        token_positions = positions.unsqueeze(0).expand(batch_size, seq_len)   # (batch, seq_len)

        # 6. 对输入进行token embedding
        # 编码之后的形状(batch, seq_len, d_model)
        x=self.token_embedding(token_ids) 

        # 7.输入到transformer blokcs模块
        for block in self.blocks:
            x=block(x,token_positions)

        # 8.RMS归一化
        x=self.norm_final(x)

        # 9.LM_head 投影回vocab
        logits=self.lm_head(x)

        return logits
