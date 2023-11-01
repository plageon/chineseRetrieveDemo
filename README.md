## 1. 项目介绍
做一个中文检索的项目，主要涉及到的技术有：   
1. **收集文档**：首先，我们需要收集所有要索引的文档。这些文档可以来自各种来源，例如网站、数据库、文件等。
2. **分词**：接下来，我们需要对文档进行分词。分词是将文本分成单独的单词或短语的过程。这可以通过使用现有的分词器库来完成。
3. **建立倒排索引**：现在，我们可以开始建立倒排索引了。对于每个单词，我们需要找到包含该单词的所有文档，并将其添加到该单词的文档列表中。
4. **重排序**：检索重排序技术是一种用于优化搜索引擎结果的技术。它的目的是通过重新排序搜索结果，以便最相关的结果排在前面，从而提高搜索结果的质量。
## 2. 实验环境
我使用了miracl-zh-corpus-22-12数据集作为实验数据集[miracl/miracl-corpus](https://huggingface.co/datasets/Cohere/miracl-zh-corpus-22-12)
MIRACL是通过从维基百科转储中提取纯文本并丢弃图像、表格等非文本元素等数据清洗过程收集的。然后，使用 WikiExtractor 将每篇文章分成多个段落，这些段落基于自然语言单位（例如段落）。每个段落都包含一个“文档”或检索单元。同时，还保留了每个段落的维基百科文章标题。
## 3. 项目结构
```
├── README.md
├── miracl-zh-corpus-22-12
│   ├── data
│   │   ├── README.md
│   │   ├── train-00000-of-00034.parquet
│   │   ├── train-00001-of-00034.parquet
│   │   ├── ...
│   │   ├── train-00033-of-00034.parquet
│   ├── doc_id2file_index.json
│   ├── miracle-zh-reverse-index.json
│   ├── miracle-zh-vocab.json
├──bool_retrieve_rerank.py
├──create_reverse_index.py
├──rerank_model.py
```
## 4. 项目运行
### 4.0 数据集下载
```shell
# 安装git lfs
sudo apt-get install git-lfs
# 下载数据集
git lfs clone https://huggingface.co/datasets/Cohere/miracl-zh-corpus-22-12
```
### 4.1 创建倒排索引
当我们需要快速查找文档中的特定单词时，倒排索引是一种非常有用的数据结构。倒排索引是一个映射，将每个单词映射到包含该单词的文档列表。
我们使用了scacy的中文分词库，具体版本是[zh-core-web-sm-3.7.0](https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.7.0/zh_core_web_sm-3.7.0-py3-none-any.whl)，具体的实现可以参考[Available trained pipelines for Chinese](https://spacy.io/models/zh)
```shell
python -m spacy download zh_core_web_sm
```
```shell
python create_reverse_index.py
```
### 4.2 重排序模型
使用[nboost/pt-bert-base-uncased-msmarco](https://huggingface.co/nboost/pt-bert-base-uncased-msmarco)，具体的实现可以参考[rerank_model.py](rerank_model.py)
```shell
git lfs clone https://huggingface.co/nboost/pt-bert-base-uncased-msmarco
```

### 4.3 布尔检索和重排序实现
使用qt5做一个简单的界面,具体的实现可以参考[bool_retrieve_rerank.py](bool_retrieve_rerank.py)
```shell
python bool_retrieve_rerank.py
```
## 5. 项目效果
![image](search_ui.png)

