import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import types    #包含python中各种数据类型
import numpy as np
from ..base_model import base_model
#from ...utils.tokenization import WordTokenizer

class base_encoder(base_model):

    def __init__(self, 
                 word_size = 18,
                 max_length = 300,
                 blank_padding = True):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence, used for postion embedding
            hidden_size: hidden size
            word_size: size of word embedding
            position_size: size of position embedding
            blank_padding: padding for CNN
            word2vec: pretrained word2vec numpy
        """
        # hyperparameters
        super(base_encoder, self).__init__()

        self.max_length = max_length
        self.word_size = word_size
        self.blank_padding = blank_padding

    def set_encoder_layer(self, encoder_layer):
        self.encoder_layer = encoder_layer 

    def forward(self):
        pass
    
    
