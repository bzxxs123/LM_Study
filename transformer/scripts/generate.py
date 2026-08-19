import torch
from transformer.pre_norm_block import softmax

def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    对一个概率分布做 nucleus (top-p) 截断，返回截断+重新归一化后的分布。
    """
    # 1.将logtis按概率从大到小排序，同时记录排序后对应的原始索引
    sorted_probs,sorted_indices=torch.sort(probs,descending=True)

    # 2.计算累加值
    # probs = [0.5, 0.3, 0.15, 0.05]
    # torch.cumsum(probs, dim=-1) 的结果是：
    # cumulative_probs = [0.5, 0.8, 0.95, 1.0]

    cumulative_probs=torch.cumsum(sorted_probs,dim=-1)

    # 3.找到需要截断的位置（>=top-p)，用mask来标记
    # 去掉当前这个 token 之后，前面剩下的累积和是不是已经超过top-p
    mask=cumulative_probs-sorted_probs>top_p

    # 4.将mask标记为True的截掉
    sorted_probs=sorted_probs.masked_fill(mask,0.0)

    # 5. 重新归一化
    sorted_probs = sorted_probs / sorted_probs.sum()

    # 6.还原原始顺序
    # scatter_(dim, index, src) 的语义是"分散赋值"——按照 index 告诉你的位置，把 src 里的值一个个填到目标张量对应的位置上。
    final_probs=torch.zeros_like(probs)
    final_probs.scatter_(0,sorted_indices,sorted_probs)

    return final_probs


def generate(
    model,
    prompt_ids:list[int],
    max_new_tokens:int,
    eos_token_id:int,
    temperature:float=1.0,
    top_p:float=1.0,
    device=None
)->list[int]:
    """
    自回归地从模型采样生成文本。

    Args:
        model: 已训练好的 TransformerLM
        prompt_ids: 提示词的 token id 列表
        max_new_tokens: 最多生成多少个新 token
        eos_token_id: <|endoftext|> 对应的 token id,生成到它就停止
        temperature: 温度缩放系数
        top_p: nucleus 采样的阈值,1.0 表示不做截断
        device: 设备

    Returns:
        完整的 token id 列表（包含 prompt 和新生成的部分）
    """

    model.eval()
    ids=list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1.准备输入张量，形状 (batch, seq_len)
            # 如果超过模型支持的最大长度，只保留最后 context_length 个 token 作为输入

            context_length=model.context_length 
            input_seq=ids[-context_length:] if len(ids)>context_length else ids
            inputs_ids=torch.tensor([input_seq],device=device)

            # 2.前向传播，取最后一个位置的logtis
            # 比如 “我爱” 取“爱”
            logits=model(inputs_ids) # (batch, seq_len, vocab_size)
            last_logtis=logits[0,-1,:] # 形状 (vocab_size,)

            # 3.温度缩放
            scaled_logtis=last_logtis/temperature

            # 4.softmax得到概率分布
            probs=softmax(scaled_logtis,dim=-1)

            # 5.top-p截断
            if top_p<1.0:
                probs=apply_top_p(probs,top_p)

            # 6.从截断后的概率分布采样一个token
            # torch.multinomial(probs, num_samples=1) 可以按给定的离散分布采样
            next_id=torch.multinomial(probs,num_samples=1).item()

            # 7.拼接到序列ids，并检查eos
            ids.append(next_id)
            if next_id==eos_token_id:
                break

    model.train()

    return ids




