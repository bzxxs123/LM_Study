import regex as re
from collections import defaultdict
import multiprocessing
from functools import partial
import time
import json
from typing import BinaryIO
import os
import io
"""
word_freqs:每个词的频率,由预分词得到
数据结构:dict[tuple[bytes, ...], int]
比如
(b'l', b'o', b'w'): 5
pair_counts:字节对以及频率,对文本逐个统计得到
数据结构:dict[tuple[bytes, bytes], int]
比如
(b'l', b'o'): 7,
(b'o', b'w'): 7
pair_to_words:倒排索引账本。记录每个字节对分别出现在哪些单词(word key)里
数据结构:dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
比如
(b'l', b'o'): {(b'l', b'o', b'w'), (b'l', b'o', b'w', b'e', b'r')},
"""

# GPT-2正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def init_vocab(special_tokens:list[str]):
    """
    初始化词表,dict类型
    key是整数(0-255) value是bytes
    """
    vocab={}

    # 初始化基础的 256 个字节 (0 - 255)
    for i in range(256):
        vocab[i]=bytes([i])

    # 加入特殊token
    if special_tokens:
        for tok in special_tokens:
            new_id=len(vocab)
            vocab[new_id]=tok.encode("utf-8")

    return vocab
#special_tokens = ["<|endoftext|>"]
#print(init_vocab(special_tokens))

def find_chunk_boundaries(
        file: BinaryIO,  # 二进制模式
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    将大文件分块块
    """
    file.seek(0,os.SEEK_END) # 将文件指针移到末尾
    file_size=file.tell()    # 获取文件大小
    file.seek(0)             # 文件指针回到开头

    # 平均分每一块
    chunk_size=file_size//desired_num_chunks

    # 理想切割点
    chunk_boundaries=[i*chunk_size for i in range(desired_num_chunks+1)]
    chunk_boundaries[-1]=file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 把文件指针跳到预测的位置

        while True:
            mini_chunk = file.read(mini_chunk_size)  # 每次往后读 4KB (4096字节) 看看

            if mini_chunk == b"": # 如果读到文件末尾了还没找到
                chunk_boundaries[bi] = file_size # 边界设在文件末尾
                break

            # 在这 4KB 的内容里，寻找特殊 token
            found_at = mini_chunk.find(split_special_token)
            
            if found_at != -1: # 找到了！
                # 分割位置 = 当前搜寻的起始点 + 偏移量
                chunk_boundaries[bi] = initial_position + found_at
                break
                
            # 如果这 4KB 里没有，那就把搜寻起点往后挪 4KB，继续下一轮 while 循环
            initial_position += mini_chunk_size

    # 返回切割位置
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(text:str,special_tokens:list[str])->dict:
    """
    单块文本的预分词,去除特殊token和统计词频

    1.把原本是一整块的超大文本，顺着特殊 Token 的位置切成一段段的“小块”，并且把特殊 Token 本身扔掉
    """
    
    if special_tokens:
        """
        给特殊Token加上反斜杠,让它们变成普通文本符号
        ["<|endoftext|>", "[UNK]"]变成
        ["<\\|endoftext\\|>", "\\[UNK\\]"]
        """
        
        escaped_tokens=[re.escape(tok) for tok in special_tokens]

        """
        把刚刚转义好的 Token 用 | 拼接起来
        <\\|endoftext\\|>|\\[UNK\\]
        (?: ... )：非捕获组
        """
        spilt_pattern="(?:" + "|".join(escaped_tokens) + ")"

        """
        split:按正则表达式切分字符串
        非捕获组 (?:)，切分出来的列表里就会直接丢掉分隔符
        """
        chunks=re.split(spilt_pattern,text)
    else:
        chunks=[text]

    # 词频统计
    # defaultdict 的好处是，碰到新 key 时它会自动初始化为 0
    word_counts=defaultdict(int)

    """
    2.遍历每一个切分好的文本块,进行预分词和统计
    """

    for chunk in chunks:
        if not chunk: # 跳过空字符串
            continue
        # 使用正则表达式找出该chunk中的预分词（pre-token),比如'low'
        for match in re.finditer(PAT,chunk):
            """
            re.finditer:
            在字符串中查找所有匹配正则的部分
            返回一个迭代器(iterator),比如
            <re.Match object; span=(6, 9), match='123'>
            <re.Match object; span=(16, 19), match='456'>

            match.group():用来取出实际匹配到的字符串
            """
            word_str=match.group()

            # 将预分词编码为UTF-8字节序列
            word_bytes=word_str.encode("UTF-8")

           # 把字节序列拆解成单个字节组成的tuple
            word_tuple= tuple(bytes([b]) for b in word_bytes)

            # 统计频率
            # 比如{ (b'l', b'o', b'w'): 1 }
            word_counts[word_tuple]+=1

    return dict(word_counts)


def pre_tokenize_from_file(start_end:tuple[int,int],input_path:str,special_tokens:list[str]):

    """
    1.读取文件的start - end
    2.处理特殊token,并按特殊token分割成块
    3.对每一块进行预分词
    4.统计词频
    """
    start,end=start_end
    with open(input_path,"rb") as f:
        f.seek(start)
        text=f.read(end-start) #此时是2进行类型数据
        '''
        io.BytesIO(text) 把已经读到的字节数据包装成一个"内存里的文件对象"，然后 io.TextIOWrapper 像包装真实文件一样包装它,默认参数下就会自动做UTF-8解码 + 通用换行符转换（\r\n/\r → \n)
        '''
        text = io.TextIOWrapper(io.BytesIO(text), encoding="utf-8").read()

    word_counts=pre_tokenize_chunk(text,special_tokens)

    return word_counts
        
        


# 测试代码
#test_text = "hello world <|endoftext|> hello"
#print(pre_tokenize_chunk(test_text, ["<|endoftext|>"]))



def build_initial_indexes(word_freqs:dict[tuple[bytes,...],int])->tuple[dict,dict]:
    '''
    维护一个全局的pair_counts,只需要初始化时统计一次，合并之后更新它
    并维护一个pair_to_words:倒排索引(哪些word包含这个pair,比如l,o包含在low和lower中)
    '''
    pair_counts=defaultdict(int)
    pair_to_words=defaultdict(set)

    for word,freq in word_freqs.items():
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            pair_counts[pair]+=freq
            pair_to_words[pair].add(word)

    return pair_counts,pair_to_words

def apply_merge_optimized(word_freqs:dict[tuple[bytes,...],int],pair_counts:dict,pair_to_words:dict,best_pair:tuple):
    '''
    1.找到受影响的 word
    2.删除旧 pair 统计
    3.替换 token
    4.添加新 pair 统计
    '''
    # 将set转化为list，防范边迭代边修改导致的报错
    # 比如找到low和lower(l,o)
    words_to_process=list(pair_to_words.get(best_pair,[]))

   
    for old_word in words_to_process:
        freq=word_freqs[old_word]
         # 去除旧词
        for i in range(len(old_word)-1):
            p=(old_word[i],old_word[i+1])
            pair_counts[p]-=freq
            if old_word in pair_to_words[p]:
                pair_to_words[p].remove(old_word)

        del word_freqs[old_word]

        # 生成新词
        new_word_list=[]
        i=0
        while i<len(old_word):
            if i<len(old_word)-1 and (old_word[i],old_word[i+1])==best_pair:
                new_word_list.append(old_word[i]+old_word[i+1])
                i+=2
            else:
                new_word_list.append(old_word[i])
                i+=1
        new_word=tuple(new_word_list)

        # 新词加入(更新)
        word_freqs[new_word]=freq
        for i in range(len(new_word)-1):
            p=(new_word[i],new_word[i+1])
            pair_counts[p]+=freq
            pair_to_words[p].add(new_word)




    



def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]):
    """
    训练bpe分词器主函数
    返回:
        vocab:dict[int,bytes]
        merges:list[tuple[bytes,bytes]]
    """

    # 初始化词表
    vocab=init_vocab(special_tokens)

    # 初始化merges列表,记录合并
    merges=[]

    '''
    获取cpu核心数量
    留一个核心给系统
    主进程还需要调度任务
    '''
    num_processes=max(1,multiprocessing.cpu_count()-1)

    # 1.读取文件
    desired_num_chunk=num_processes*8
    
    # 用二进制读取（优化）
    with open(input_path,"rb") as f:
        chunk_boundaries=find_chunk_boundaries(f,desired_num_chunk,special_tokens[0].encode("utf-8"))

    intervals = list(zip(chunk_boundaries, chunk_boundaries[1:]))
   

    # 3. 多进程并行处理
    worker=partial(pre_tokenize_from_file,input_path=input_path,special_tokens=special_tokens)
   
   

    # 汇总所有进程的结果
    global_word_counts=defaultdict(int)

    print(f"启动多进程预分词，使用 {num_processes} 个核心，处理 {len(intervals)} 个文本块...")

    with multiprocessing.Pool(num_processes) as pool:
        # pool.map 会把 chunks 列表里的元素一个个给 worker 函数，并收集返回的字典列表
        chunk_dicts=pool.map(worker,intervals)

        # 4. 汇总结果
        for chunk_dict in chunk_dicts:
            for word_tuple,count in chunk_dict.items():
                global_word_counts[word_tuple]+=count

    print(f"预分词完成！共提取出 {len(global_word_counts)} 种不同的 pre-token。")

    word_freqs=dict(global_word_counts)

    # 4.合并


    # 优化合并方法
    start_time=time.time()
    pair_counts,pair_to_words=build_initial_indexes(word_freqs)

    while len(vocab)<vocab_size:
        # 去掉频次<=0的数据
        # pair_counts={p:c for p,c in pair_counts.items() if c>0}
        if not pair_counts:
            break

        # 寻找最高频次
        best_pair=max(pair_counts,key=lambda p:(pair_counts[p],p))

      

        # 记录合并
        merges.append(best_pair)

        # 新token加入词表
        new_id=len(vocab)
        vocab[new_id]=best_pair[0]+best_pair[1]

        # 更新
        apply_merge_optimized(word_freqs,pair_counts,pair_to_words,best_pair)

        if len(vocab) % 500 == 0:
            print(f"当前词表大小: {len(vocab)}/{vocab_size}")

    end_time=time.time()
    print(f"优化后的 BPE 训练完成！耗时: {end_time - start_time:.2f} 秒")
        

    return vocab, merges



# 测试
if __name__== "__main__":
    vocab,merges=train_bpe("data/TinyStories-train.txt", 10000, ["<|endoftext|>"])
    print(merges[:10])
    # 保存vocab为json格式数据
    # 把 key 转成字符串，把 bytes 解码成能够保存的字符串（比如用 latin-1 无损解码）
    vocab_for_json={str(k):v.decode("latin-1") for k,v in vocab.items()}
    with open("vocab.json","w",encoding="utf-8") as f:
        json.dump(vocab_for_json,f,indent=2,ensure_ascii=False)

    # 保存 Merges 为 TXT
    with open("merges.txt","w",encoding="utf-8") as f:
        for token1,token2 in merges:
            line=f"{token1.decode("latin-1")} {token2.decode("latin-1")}\n"
            f.write(line)

    
