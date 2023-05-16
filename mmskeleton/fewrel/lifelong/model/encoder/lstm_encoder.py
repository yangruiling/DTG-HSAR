from mmskeleton.fewrel.lifelong.model.encoder.base_encoder import base_encoder
from mmskeleton.fewrel.lifelong.model.module import embedding_layer, lstm_layer
# from lifelong.model.encoder.base_encoder import base_encoder
# from lifelong.model.module import embedding_layer, lstm_layer
import json
import torch
import numpy as np

#使用LSTM记忆网络编码，继承base_encoder
class lstm_encoder(base_encoder):

    def __init__(self, word_size = 18, max_length = 300, 
            pos_size = None, hidden_size = 230, dropout = 0, bidirectional = True, num_layers = 1, config = None):
        super(lstm_encoder, self).__init__(word_size, max_length, blank_padding = False)
        self.config = config
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.word_size = word_size
        self.pos_size = pos_size
        self.input_size = word_size
        if bidirectional:
            self.output_size = hidden_size * 2
        else:
            self.output_size = hidden_size
        if pos_size != None:
            self.input_size += 2 * pos_size
        self.encoder_layer = lstm_layer(max_length, self.input_size, hidden_size, dropout, bidirectional, num_layers, config)

    def forward(self, inputs, lengths = None):
        #对输入sequence进行预处理：填充为相同长度
        inputs, lengths, inputs_indexs = self.encoder_layer.pad_sequence(inputs)
        #将处理后的 sentence 序列送入GPU
        inputs = inputs.to(self.config['device'])
        #调用输入sentence序列进行编码处理
        x = self.encoder_layer(inputs, lengths, inputs_indexs)
        #print("encoder_layer",x)
        return x



if __name__ == '__main__':
    f = open("/home/yangruiling/mmskeleton/mmskeleton/fewrel/config/config_fewrel.json", "r")
    config = json.loads(f.read())
    f.close()
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])
    
    #设置根目录
    root_path = '.'
    encoder = lstm_encoder(      #初始化编码
        word_size = 18,    #每个单词对应的的嵌入向量的长度
        max_length = 300, 
        pos_size = None, 
        hidden_size = config['hidden_size'], 
        dropout = 0, 
        bidirectional = True, 
        num_layers = 1, 
        config = config)
    
    print(np.array(encoder))