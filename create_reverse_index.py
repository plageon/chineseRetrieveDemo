import json
import os
from collections import defaultdict

import pandas as pd
import spacy
from tqdm import tqdm

# 加载中文模型
nlp = spacy.load('zh_core_web_trf', enable=['tokenizer'])

# vocab 存储路径
vocab_path = "miracl-zh-corpus-22-12/miracle-zh-vocab.json"
vocab = set()
# reverse index 存储路径
reverse_index_path = "miracl-zh-corpus-22-12/miracle-zh-reverse-index.json"

parquet_data_path = "miracl-zh-corpus-22-12/data/"
parquet_file_list = os.listdir(parquet_data_path)[:4]


def collect_vocab():
    print("--------------------Collecting vocab--------------------")
    reverse_index = {}
    # 依次读取文件 miracl-zh-corpus-22-12/train-00000-of-00034-334bb8d9c6b3d57e.parquet
    for parquet_file in parquet_file_list:
        file_name = os.path.join(parquet_data_path, parquet_file)
        # 读取文件
        print(f"Reading {file_name}")
        df = pd.read_parquet(file_name, engine='fastparquet')
        # 读取中文文本
        # 字段包含 docid, title, text, emb
        texts = df['text'].tolist()
        print(f"Processing {len(texts)} texts")
        # 用pipeline处理文本
        # 依次处理每个文本
        for idx, doc in tqdm(enumerate(nlp.pipe(texts, batch_size=512, n_process=8)), total=len(texts)):
            # 依次处理每个token
            for token in doc:
                # 如果token不是标点符号或者停用词
                if not token.is_punct and not token.is_stop:
                    # 将token加入vocab
                    vocab.add(token.text)
                    # 将token的docid加入reverse index
                    if token.text in reverse_index:
                        reverse_index[token.text].append(df['docid'][idx])
                    else:
                        reverse_index[token.text] = [df['docid'][idx]]

    # 将vocab写入文件
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(list(vocab), ensure_ascii=False))
    # 将reverse index写入文件
    with open(reverse_index_path, 'w') as f:
        f.write(json.dumps(reverse_index, ensure_ascii=False))


# 在词表中加入title
def collect_title():
    print("--------------------Collecting title--------------------")
    reverse_index = json.load(open(reverse_index_path, 'r'))

    # 依次读取文件 miracl-zh-corpus-22-12/train-00000-of-00034-334bb8d9c6b3d57e.parquet
    for parquet_file in parquet_file_list:
        file_name = os.path.join(parquet_data_path, parquet_file)
        # 读取文件
        print(f"Reading {file_name}")
        df = pd.read_parquet(file_name, engine='fastparquet')
        # 在词表中直接添加title
        last_title = None
        for idx, title in enumerate(df['title']):
            if title != last_title:
                vocab.add(title)
                last_title = title
                # 将title的docid加入reverse index
                if title in reverse_index:
                    reverse_index[title].append(df['docid'][idx])
                else:
                    reverse_index[title] = [df['docid'][idx]]

    # 将vocab写入文件
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(list(vocab), ensure_ascii=False))
    # 将reverse index写入文件
    with open(reverse_index_path, 'w') as f:
        f.write(json.dumps(reverse_index, ensure_ascii=False))


def make_doc_id2file_index():
    print("--------------------Making doc_id2file_index--------------------")
    file2doc_id = defaultdict(list)
    for parquet_file in parquet_file_list:
        file_name = os.path.join(parquet_data_path, parquet_file)
        # 读取文件
        print(f"Reading {file_name}")
        df = pd.read_parquet(file_name, engine='fastparquet')
        last_doc_id = None
        for full_doc_id in df['docid']:
            doc_id = full_doc_id.split('#')[0]
            if doc_id != last_doc_id:
                file2doc_id[parquet_file].append(doc_id)
                last_doc_id = doc_id
    doc_id2file_index = {}
    for file_name, doc_ids in file2doc_id.items():
        for doc_id in doc_ids:
            doc_id2file_index[doc_id] = file_name

    with open("miracl-zh-corpus-22-12/doc_id2file_index.json", 'w') as f:
        f.write(json.dumps(doc_id2file_index, ensure_ascii=False))


if __name__ == "__main__":
    collect_vocab()
    collect_title()
    make_doc_id2file_index()
