import torch
from torch import nn, optim
import math
from ..base_model import base_model

class proto_softmax_layer(base_model):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __distance__(self, rep, rel):
        rep_ = rep.view(rep.shape[0], 1, rep.shape[-1])
        rel_ = rel.view(1, -1, rel.shape[-1])
        dis = (rep_ * rel_).sum(-1)
        return dis

    def __init__(self, sentence_encoder, num_class, id2rel, drop = 0, config = None, rate = 1.0):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super(proto_softmax_layer, self).__init__()

        self.config = config
        self.sentence_encoder = sentence_encoder     #lstm_encoder((encoder_layer): lstm_layer((lstm): LSTM(18, 200, bidirectional=True)))
        self.num_class = num_class    #400
        self.hidden_size = self.sentence_encoder.output_size       #hidden_size * 2 = 200 * 2 
        self.fc = nn.Linear(self.hidden_size, self.num_class, bias = False)
        self.drop = nn.Dropout(drop)
        self.id2rel = id2rel      #[0,400)
        self.rel2id = {}
        for i in id2rel:
            self.rel2id[i] = i

    #设定记忆原型
    def set_memorized_prototypes(self, protos):
        self.prototypes = protos.detach().to(self.config['device'])
   
    #获取编码器的返回值，并返回numpy格式
    def get_feature(self, sentences, length = None):
        rep = self.sentence_encoder(sentences, length)
        return rep.cpu().data.numpy()

    def get_mem_feature(self, rep):
        dis = self.mem_forward(rep)
        return dis.cpu().data.numpy()
    
    def forward(self, sentences, length = None):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        rep = self.sentence_encoder(sentences, length) # (B, H)
        rep = self.drop(rep)
        logits = self.fc(rep)
        return logits, rep

    def mem_forward(self, rep):
        """
        Args:
            args: depends on the encoder
        Return:
            logits, (B, N)
        """
        dis_mem = self.__distance__(rep, self.prototypes)
        return dis_mem
