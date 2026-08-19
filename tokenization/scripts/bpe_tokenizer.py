import os
import regex as re
from typing import Iterable,Iterator
import json
import time
import cProfile

class Tokenizer:
    def __init__(
        self,
        vocab:dict[int,bytes],
        merges:list[tuple[bytes,bytes]],
        special_tokens:list[str]|None=None  # 用户的special_token
    ):
        """
        初始化Tokenizer
        保存传入的vocab,merges,special_tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.compiled_pat = re.compile(self.PAT)

        # 反向查表字典：key为bytes，value为id（vocab是key为id，value为bytes）
        self.bytes_to_id={v:k for k,v in self.vocab.items()}
        # 记录 Merge 优先级
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        # 将用户的special_token加入到vocab
        for token in special_tokens:
            token_bytes=token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                new_id=len(vocab)
                self.vocab[new_id]=token_bytes
                self.bytes_to_id[token_bytes]=new_id

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    @classmethod
    def from_files(
        cls,
        vocab_filepath:str|os.PathLike,
        merges_filepath:str|os.PathLike,
        special_tokens:list[str]|None=None
    ):
        """
        类方法：从磁盘文件加载 vocab 和 merges,然后返回一个 Tokenizer 实例。
        """
        # 1.读取vocab,并还原成bytes
        with open(vocab_filepath,"r",encoding="utf-8") as f:
            vocab_json=json.load(f)
        # 把字符串 key 转回 int，把字符串 value 重新编码为 bytes
        vocab={int(k):v.encode("latin-1") for k,v in vocab_json.items()}

        # 2.读取merges
        merges=[]
        with open(merges_filepath,"r",encoding="utf-8") as f:
            for line in f:
                # 去除尾行换行符
                line=line.strip('\n')
                if not line:
                    continue

                # 按空格切分成为两个token，转换为bytes
                parts=line.split(" ")
                if len(parts)==2:
                    merges.append((parts[0].encode("latin-1"),parts[1].encode("latin-1")))

        return cls(vocab=vocab,merges=merges,special_tokens=special_tokens)




    def encode(self,text:str)->list[int]:
        """
        字符串 -> 整数 ID 列表。
        1.隔离特殊token
        2.预分词
        3.转换为字节,根据merges合并,然后查表得到ID
        """
        final_ids=[]

        # 1.切分special_tokens
        if self.special_tokens:
            escaped=[re.escape(tok) for tok in self.special_tokens]
            pattern="("+ "|".join(escaped) + ")"

            chunks=re.split(pattern,text)
        else:
            chunks=[text]

        # 遍历切分的每一块
        for chunk in chunks:
            if not chunk:
                continue

            if chunk in self.special_tokens:
                # 遍历到特殊token
                special_token_id=self.bytes_to_id[chunk.encode("utf-8")]
                final_ids.append(special_token_id)

            else:
                # 普通文本
                # 2.预分词
                for match in self.compiled_pat.finditer(chunk):
                    # 提取字符
                    word_str=match.group()
                    # 编码
                    word_bytes=word_str.encode("utf-8")
                    # 3.合并（按照merges）
                    # 3.1 先把word_bytes分开，比如把 b'low' 打散成 (b'l', b'o', b'w')
                    word=tuple(bytes([b]) for b in word_bytes)
                    while True:

                        best_pair=None
                        best_rank=float('inf')
                        for i in range(len(word)-1):
                            pair=(word[i],word[i+1])
                            rank=self.merge_ranks.get(pair,float('inf'))
                            if rank<best_rank:
                                best_pair=pair
                                best_rank=rank

                        # 如果该best_pair不在merges里（无法合并），其他的也无法合并，直接跳出
                        if best_rank == float('inf'):
                            break

                        # 可以合并
                        else:
                            new_word=[]
                            i=0
                            while i < len(word):
                                if i<len(word)-1 and (word[i],word[i+1])==best_pair:
                                   new_word.append(best_pair[0]+best_pair[1])
                                   i+=2
                                else:
                                    new_word.append(word[i])
                                    i+=1

                        # word 转化为元组 (tuple)，比如：(b'lo', b'w')
                        word=tuple(new_word)

                    # 4.得到了合并之后的word，查找对应的id，然后加入final_ids
                    for word_token in word:
                        word_id=self.bytes_to_id[word_token]
                        final_ids.append(word_id)

        return final_ids

                            
                                   


    def encode_iterable(self,iterable:Iterator[str])->Iterator[int]:
        """
        流式编码：用于处理连内存都塞不下的大文件。
        每次从 iterable 拿一段文本进行 encode,并使用 yield 懒加载返回 ID。
        """

        # 1.每次从迭代器里区一块文本
        for chunk_text in iterable:

            # 2.调用encode
            chunk_ids=self.encode(chunk_text)

            # 3.把这串id 逐个返回
            yield from chunk_ids


    def decode(self,ids:list[int])->str:
        """
        整数 ID 列表 -> 字符串
        """

        # 接收字节
        text_bytes=b""

        # 1.查表，根据ids查找对应的bytes
        for token_id in ids:
            token_bytes=self.vocab[token_id]
            text_bytes+=token_bytes

        # 2.将文本字节转化为字符

        text=text_bytes.decode("utf-8",errors="replace")

        return text

if __name__=="__main__":
    tokenizer = Tokenizer.from_files("vocab.json", "merges.txt", special_tokens=["<|endoftext|>"])

    text = "Hello world! <|endoftext|> Testing."*1000
    start = time.time()
    ids = tokenizer.encode(text)
    print(f"耗时: {time.time()-start:.3f}s, 得到 {len(ids)} 个 token")

    # 用 profiler 找到具体哪一行最耗时
    cProfile.run("tokenizer.encode(text)", sort="cumulative")

    
