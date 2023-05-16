import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from ..base_model import base_model

class embedding_layer(base_model):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim = 50, pos_embedding_dim = None, requires_grad = True):
        super(embedding_layer, self).__init__()
        self.max_length = max_length
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        
        # Word embedding  词嵌入
        # unk = torch.randn(1, word_embedding_dim) / math.sqrt(word_embedding_dim)
        # blk = torch.zeros(1, word_embedding_dim)
        word_vec_mat = torch.from_numpy(word_vec_mat)

        #词嵌入函数 torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None,max_norm=None,  norm_type=2.0,   scale_grad_by_freq=False, sparse=False,  _weight=None)
        # 单词的总数目，词嵌入的维度   m个单词  每个单词嵌入向量维度为n
        # 其为一个简单的存储固定大小的词典的嵌入向量的查找表，意思就是说，给一个编号，嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系
        # https://www.jianshu.com/p/63e7acc5e890

        self.word_embedding = nn.Embedding(word_vec_mat.shape[0], self.word_embedding_dim, padding_idx = word_vec_mat.shape[0] - 1)
        self.word_embedding.weight.data.copy_(word_vec_mat)
        self.word_embedding.weight.requires_grad = requires_grad

        # Position Embedding   位置嵌入
        if self.pos_embedding_dim != None:
            self.pos1_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx = 0)
            self.pos2_embedding = nn.Embedding(2 * max_length, pos_embedding_dim, padding_idx = 0)
    
    # 输入为一个编号列表，输出为对应的符号嵌入向量列表
    def forward(self, word, pos1 = None, pos2 = None):
        if pos1 != None and pos2 != None and self.pos_embedding_dim != None:
            #torch.cat()  将多个tensor进行拼接
            x = torch.cat([self.word_embedding(word), 
                            self.pos1_embedding(pos1), 
                            self.pos2_embedding(pos2)], 2)
        else:
            x = self.word_embedding(word)
        return x


