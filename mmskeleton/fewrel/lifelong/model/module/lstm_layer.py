import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from ..base_model import base_model

class lstm_layer(base_model):

    def __init__(self, max_length = 128, input_size = 50, hidden_size = 256, dropout = 0, bidirectional = True, num_layers = 1, config = None):
        """
        Args:
            input_size: dimention of input embedding
            hidden_size: hidden size
            dropout: dropout layer on the outputs of each RNN layer except the last layer
            bidirectional: if it is a bidirectional RNN
            num_layers: number of recurrent layers
            activation_function: the activation function of RNN, tanh/relu
        """
        super(lstm_layer, self).__init__()
        self.device = config['device']
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.input_size = input_size
        if bidirectional:
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = bidirectional, num_layers = num_layers, dropout = dropout)

    #初始化隐藏层
    def init_hidden(self, batch_size = 1, device='cpu'):
        self.hidden = (torch.zeros(2, batch_size, self.hidden_size).to(device),
                torch.zeros(2, batch_size, self.hidden_size).to(device))
    
    #定义前向传播
    #LSTM单元接受的输入都必须是3维的张量(Tensors).每一维代表的意思不能弄错。
    #第一维体现的是序列（sequence）结构,也就是序列的个数
    #第二维度体现的是batch_size，也就是一次性喂给网络多少条句子
    #第三位体现的是输入的元素（elements of input），也就是，每个具体的单词用多少维向量来表示
    
    def forward(self, inputs, lengths, inputs_indexs):
        #统一序列长度
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths)
        lstm_out, hidden = self.lstm(packed_embeds, self.hidden)
        permuted_hidden = hidden[0].permute([1,0,2]).contiguous()
        permuted_hidden = permuted_hidden.view(-1, self.hidden_size * 2)
        output_embedding = permuted_hidden[inputs_indexs]
        return output_embedding

    #对sequence进行排序
    def ranking_sequence(self, sequence):
        word_lengths = torch.tensor([len(sentence) for sentence in sequence])  # 词长度 = sentence列表中每一个句子的长度
        rankedi_word, indexs = word_lengths.sort(descending = True)   #按找句子长度进行排序
        ranked_indexs, inverse_indexs = indexs.sort()   #反向排序
        sequence = [sequence[i] for i in indexs]
        return sequence, inverse_indexs       #返回排序后的序列，及对应序号
    
    #序列预处理：在RNN中处理变长序列，需要保证每个样本具有相同的序列长度

    #使用pack自动裁剪Tensor，并将其堆叠
    #对于没有使用0填充的不等长tensor，直接使用torch.nn.utils.rnn.pack_sequence即可
    #如果是已经使用0填充过的tensor，使用torch.nn.utils.rnn.pack_padded_sequence即可
    
    #使用pad操作可以进行扩充，使每一个序列tensor的shape保持一致
    #对于一般的样本，直接使用torch.nn.utils.rnn.pad_sequence
    #如果目标对象是sequence对象，则使用torch.nn.utils.rnn.pad__packed__sequence即可
    def pad_sequence(self, inputs, padding_value = 0):
        self.init_hidden(len(inputs), self.device)
        inputs, inputs_indexs = self.ranking_sequence(inputs)
        lengths = [len(data) for data in inputs]    #获取每条数据的长度
        pad_inputs = torch.nn.utils.rnn.pad_sequence(inputs, padding_value = padding_value)    #对不等长的数据进行0填充处理
        return pad_inputs, lengths, inputs_indexs
