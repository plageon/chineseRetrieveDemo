import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
#  添加环境变量 TRANSFORMERS_CACHE=/data00/jiejuntan/huggingface/
os.environ['TRANSFORMERS_CACHE'] = 'D:\huggingface/'


@dataclass
class RerankConfig:
    rerank_model_name_or_path: str = 'nboost/pt-bert-base-uncased-msmarco'
    projection: bool = True
    indexing_dimension: int = 512
    embed_batch_size: int = 8
    query_max_length: int = 128
    doc_max_length: int = 512


class RerankModel:
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.rerank_model_name_or_path)
        self.rerank_model = AutoModel.from_pretrained(config.rerank_model_name_or_path)

        self.max_length_count = 0

        self.proj = nn.Linear(
            self.rerank_model.config.hidden_size,
            self.config.indexing_dimension
        )
        self.norm = nn.LayerNorm(self.config.indexing_dimension)

    @torch.no_grad()
    def rerank_batch(self, query_batch, docs_batch):
        query_inputs = self.tokenizer(
            query_batch,
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.config.query_max_length,
            padding='longest',
            truncation=True)
        query_output = self.embed_text(
            text_ids=query_inputs.input_ids,
            text_mask=query_inputs.attention_mask,
            extract_cls=False,
        )
        bsz = len(query_batch)
        n_docs = len(docs_batch[0])
        doc_inputs = self.tokenizer(
            [doc for docs in docs_batch for doc in docs],
            add_special_tokens=True,
            return_tensors='pt',
            max_length=self.config.doc_max_length,
            padding='longest',
            truncation=True)
        doc_output = self.embed_text(
            text_ids=doc_inputs.input_ids,
            text_mask=doc_inputs.attention_mask,
            extract_cls=False,
        )
        # track how often we truncate to max_seq_length
        # if doc_inputs['input_ids'].shape[1] == self.args.max_seq_length:
        #     self.max_length_count += 1
        # use log_softmax to get log probabilities
        score = torch.einsum(
            'bd,bid->bi',
            query_output,
            doc_output.view(bsz, n_docs, -1)
        )
        score = score / np.sqrt(query_output.size(-1))

        return score

    def embed_text(self, text_ids, text_mask, extract_cls=False):
        text_ids = text_ids.to(self.rerank_model.device)
        text_mask = text_mask.to(self.rerank_model.device)
        if text_ids.size(0) > self.config.embed_batch_size:
            #  use batch embedding
            text_output = []
            for batch_idx in range(0, len(text_ids), self.config.embed_batch_size):
                batch_text_output = self.rerank_model(
                    input_ids=text_ids[batch_idx: batch_idx + self.config.embed_batch_size],
                    attention_mask=text_mask[batch_idx: batch_idx + self.config.embed_batch_size]
                )
                text_output.append(batch_text_output.last_hidden_state)
            text_output = torch.cat(text_output, dim=0)
        else:
            text_output = self.rerank_model(
                input_ids=text_ids,
                attention_mask=text_mask
            ).last_hidden_state
        # if type(text_output) is not tuple:
        #     text_output.to_tuple()
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            text_output = torch.mean(text_output, dim=1)
        return text_output

