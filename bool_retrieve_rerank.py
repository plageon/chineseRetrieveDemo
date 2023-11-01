import os
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
# 使用pyqt5的做一个简单的检索界面
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QTextEdit, QGridLayout, QLabel, QComboBox
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextCursor

from rerank_model import RerankModel, RerankConfig


class SearchWindow(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_ui()
        rerank_config = RerankConfig()
        self.rerank_model = RerankModel(rerank_config, *args, **kwargs)
        self.nlp = spacy.load('zh_core_web_sm', enable=['tokenizer'])
        self.reverse_index = self.init_reverse_index()

    def init_reverse_index(self):
        # 读取倒排索引
        docs = []
        # 打开reverse_index文件
        try:
            with open('miracl-zh-corpus-22-12/miracle-zh-reverse-index.json', 'r') as f:
                reverse_index = json.load(f)
            return reverse_index
        except FileNotFoundError:
            print('倒排索引文件不存在')
            return

    def init_ui(self):
        # 设置窗口大小
        self.resize(800, 600)
        # 设置窗口标题
        self.setWindowTitle('检索界面')
        # 设置窗口图标
        self.setWindowIcon(QIcon('icon.png'))

        # 设置字体
        font = QFont()
        font.setFamily('微软雅黑')
        font.setPixelSize(20)

        # 设置控件
        # 检索框
        self.searchEdit = QLineEdit()
        self.searchEdit.setFont(font)
        self.searchEdit.setPlaceholderText('请输入检索内容')
        # 检索按钮
        self.searchButton = QPushButton('检索')
        self.searchButton.setFont(font)
        self.searchButton.clicked.connect(self.search)
        # 检索结果
        self.searchResult = QTextEdit()
        self.searchResult.setFont(font)
        self.searchResult.setReadOnly(True)
        # 检索结果数量
        self.searchResultNum = QLabel()
        self.searchResultNum.setFont(font)
        # 检索结果排序方式
        self.sortMethod = QComboBox()
        self.sortMethod.setFont(font)
        self.sortMethod.addItems(['相关度', '时间', '热度'])
        # 检索结果排序方式标签
        self.sortMethodLabel = QLabel()
        self.sortMethodLabel.setFont(font)
        self.sortMethodLabel.setText('排序方式：')
        # 检索结果数量标签
        self.searchResultNumLabel = QLabel()
        self.searchResultNumLabel.setFont(font)
        self.searchResultNumLabel.setText('检索结果数量：')

        # 设置布局
        layout = QGridLayout()
        layout.addWidget(self.searchEdit)
        layout.addWidget(self.searchButton)
        layout.addWidget(self.searchResult)
        layout.addWidget(self.sortMethodLabel)
        layout.addWidget(self.sortMethod)
        layout.addWidget(self.searchResultNumLabel)
        layout.addWidget(self.searchResultNum)
        self.setLayout(layout)

        # 显示窗口
        self.show()

    def search(self):
        # 获取检索内容
        search_content = self.searchEdit.text()
        # 获取排序方式
        sort_method = self.sortMethod.currentText()
        # 清空检索结果
        self.searchResult.clear()
        # 检索
        search_result = self.bool_retrieve(search_content)
        # 显示检索结果
        self.searchResult.setText(search_result[0])
        # 显示检索结果数量
        self.searchResultNum.setText(str(search_result[1]))
        # 滚动到顶部
        self.searchResult.moveCursor(QTextCursor.Start)

    def bool_retrieve(self, query):
        # 给query分词
        query_tokens = [token.text for token in self.nlp(query)]

        # 读取倒排索引，按照query中的词在倒排索引中查找，并按照列表长度从小到大排序
        token_doc_lists = []
        for token in query_tokens:
            if token in self.reverse_index:
                token_doc_lists.append(set(self.reverse_index[token]))
        token_doc_lists.sort(key=lambda x: len(x))
        display_doc_list = token_doc_lists[0]
        token_num = len(token_doc_lists)
        cnt = 0
        # 从短的列表开始，对所有文档列表进行交集操作，直到文档数目小于100
        while len(display_doc_list) > 100 and cnt < token_num - 1:
            display_doc_list = set.intersection(display_doc_list, token_doc_lists[cnt + 1])
            cnt += 1

        # 读取所有文档
        docs = []
        with open('miracl-zh-corpus-22-12/doc_id2file_index.json', 'r') as f:
            doc_id2file_index = json.load(f)
        # 读取文档
        file2docid = defaultdict(list)
        for docid in display_doc_list:
            prefix = docid.split('#')[0]
            file2docid[doc_id2file_index[prefix]].append(docid)

        for file in file2docid:
            file_path = os.path.join('miracl-zh-corpus-22-12/data', file)
            df = pd.read_parquet(file_path, engine='fastparquet')
            for docid in file2docid[file]:
                docs.append(df[df['docid'] == docid]['text'].iloc[0])

        result_num = len(docs)
        if result_num == 0:
            return '没有检索到相关文档', 0
        # 只显示前64个文档
        if result_num > 64:
            docs = docs[:64]

        # 对所有文档进行rerank
        rerank_result = self.rerank(query, docs)
        return rerank_result, result_num

    def rerank(self, query, docs):
        rerank_scores = self.rerank_model.rerank_batch([query], [docs])[0]
        #  按照相关度从大到小排序
        rerank_scores = rerank_scores.detach().cpu().numpy()
        rankings = np.argsort(-rerank_scores)
        rerank_result = ''
        for rank, doc_idx in enumerate(rankings):
            rerank_result += f'{rank + 1}. {docs[doc_idx]}\n'

        return rerank_result


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SearchWindow()
    sys.exit(app.exec_())
