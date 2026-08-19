import os
import numpy as np
from tokenization.bpe_tokenizer import Tokenizer
from tokenization.bpe_tokenization_train import find_chunk_boundaries
from tqdm import tqdm
import multiprocessing
from functools import partial
import time

def encode_chunk(interval,input_path,tokenizer):
    """
    子进程worker:读取文件的某一段区间，编码成 token id 列表并返回。
    """
    start,end=interval
    with open(input_path,"rb") as f:
        f.seek(start)
        chunk_bytes=f.read(end-start)

    chunk_text=chunk_bytes.decode("utf-8",errors="ignore")
    return tokenizer.encode(chunk_text)

def parallel_tokenize_file_to_bin(input_path:str,output_path:str,tokenizer:Tokenizer,num_processes: int = None,chunk_size:int=1_000_000):
    """
    并行+流式地把文本文件编码成 token id,分块写入 .bin 文件，避免一次性占用过多内存。

    Args:
        input_path: 输入文本文件路径
        output_path: 输出的 .bin 文件路径
        tokenizer:  Tokenizer 对象
        chunk_size: 每积累多少个 token 就写一次磁盘
    """

    if num_processes is None:
        num_processes=max(1,multiprocessing.cpu_count()-1)

    # 1. 复用 find_chunk_boundaries，切出多个并行处理区间
    desired_num_chunks=num_processes*8   # 块数
    with open(input_path,"rb") as f:
        chunk_boundaries=find_chunk_boundaries(f,desired_num_chunks,tokenizer.special_tokens[0].encode("utf-8"))
    intervals = list(zip(chunk_boundaries, chunk_boundaries[1:]))

    # 2. 如果文件已存在，先清空（因为用的是追加模式，防止重复运行时叠加旧数据）
    open(output_path,"wb").close()  

    worker=partial(encode_chunk,input_path=input_path,tokenizer=tokenizer)


    # 3.用 imap（保持顺序），边收集边写入磁盘
    buffer=[]
    total_tokens=0
    start_time = time.time()

    with multiprocessing.Pool(num_processes) as pool:
        pbar = tqdm(total=len(intervals), desc="并行编码", unit="chunk")
        for chunk_token_ids in pool.imap(worker,intervals):
            buffer.extend(chunk_token_ids)
            total_tokens += len(chunk_token_ids)   # 累计已处理（含还没写盘的）token 数
            pbar.update(1)
            elapsed = time.time() - start_time
            speed = total_tokens / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                "tokens": total_tokens,
                "speed": f"{speed:.0f}/s"
            })
            if len(buffer)>=chunk_size:
                # 将这一批写入磁盘
                chunk_array=np.array(buffer,dtype=np.uint16)
                with open(output_path,"ab") as out_f:
                    chunk_array.tofile(out_f)
                total_tokens+=len(buffer)
                buffer=[] # 清空缓冲区
        pbar.close()
    
        # 循环结束，把最后不满一整批的也写入
        if buffer:
            chunk_array=np.array(buffer,dtype=np.uint16)
            with open(output_path,"ab") as out_f:
                chunk_array.tofile(out_f)
            total_tokens+=len(buffer)

    total_elapsed = time.time() - start_time
    print(f"完成：{input_path} -> {output_path}")
    print(f"共写入 {total_tokens} 个 token,耗时 {total_elapsed:.1f}s,平均速度 {total_tokens/total_elapsed:.0f} tokens/s")
    return total_tokens
    


def prepare_dataset(
        input_train_path:str,
        input_val_path:str,
        vocab_file_path:str,
        merges_file_path:str,
        output_train_path:str ="./data/train.bin",
        output_val_path:str ="./data/val.bin",
        dtype:np.dtype=np.uint16
      
):
    os.makedirs(os.path.dirname(output_train_path),exist_ok=True)
    os.makedirs(os.path.dirname(output_val_path),exist_ok=True)
    
    # 1.创建分词器
    tokenizer=Tokenizer.from_files(vocab_file_path,merges_file_path,special_tokens=["<|endoftext|>"])

    # 2.读取文本并分词
    print(f"正在读取并编码文本: {input_train_path} ...")
   
    parallel_tokenize_file_to_bin(input_train_path,output_train_path,tokenizer,chunk_size=1_000_000)

    print(f"正在读取并编码文本: {input_val_path} ...")

    parallel_tokenize_file_to_bin(input_val_path,output_val_path,tokenizer,chunk_size=1_000_000)

  
    print(f"训练集已保存至: {output_train_path}")
    print(f"验证集已保存至: {output_val_path}")

if __name__=="__main__":
    input_train_path="../tokenization/data/TinyStories-train.txt"
    input_val_path="../tokenization/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_file_path="../tokenization/vocab.json"
    merges_file_path="../tokenization/merges.txt"

    prepare_dataset(input_train_path,input_val_path,vocab_file_path,merges_file_path)

