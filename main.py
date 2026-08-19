import torch
import json
from tokenization.bpe_tokenizer import Tokenizer
from transformer.data_loader import load_checkpoint
from transformer.generate import generate
from transformer.transformer_LM import TransformLM  # 导入你的模型类定义
from transformer.transformer_lm_tool import AdamW
if __name__ == "__main__":
    
    vocab_path = "tokenization/vocab.json"
    merges_path = "tokenization/merges.txt"
    checkpoint_path = "transformer/checkpoints/ckpt_final.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
    # 3. 初始化 Tokenizer
    tokenizer = Tokenizer.from_files(vocab_path,merges_path, special_tokens=["<|endoftext|>"])
   
    # 获取 eos_token_id
    eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    # 4. 先按训练时的超参数实例化模型架构
    # （参数必须和训练时完全一致）
    model = TransformLM(
        vocab_size=len(tokenizer.vocab),
        context_length=256,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=int(round(8/3*256/64))*64,
    )
    
    # 5. 加载权重并移至 GPU
    iters = load_checkpoint(checkpoint_path, model)
    model.to(device)

    print("=" * 50)
    print("模型加载完成！输入你的 Prompt 开始续写 (输入 q / quit 退出)")
    print("=" * 50)

    while True:
        try:
            # 1.输入提示词
            user_prompt=input("\n请输入提示词:").strip()

            # 2. 退出条件
            if user_prompt.lower() in ["q", "quit", "exit"]:
                print("退出生成。")
                break
            # 3. 过滤空输入
            if not user_prompt:
                continue

            # 4. 编码 Prompt
            input_token_ids = tokenizer.encode(user_prompt)

            # 5. 自回归生成
            out_token_ids = generate(
                model=model,
                prompt_ids=input_token_ids,
                max_new_tokens=200,
                eos_token_id=eos_token_id,
                temperature=0.7,
                top_p=0.9,  
                device=device,
            )

            # 6. 解码并打印输出
            out_texts = tokenizer.decode(out_token_ids)
            print(out_texts)

        except KeyboardInterrupt:
            print("\n退出。")
            break