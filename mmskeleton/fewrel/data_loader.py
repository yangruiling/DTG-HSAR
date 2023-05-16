import torch
import torch.nn as nn
from mmskeleton.deprecated.datasets.utils import skeleton
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import numpy as np
import os

class data_set(Dataset):   #定义数据集  Torch包 Dataset

    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples

        N: batch size ,json文件的个数
        C: 3 （x,y,score）
        T: 300  帧数
        V: 18  关节点个数
        M: 2  2个人
    """
    def __init__(self,
                 data_path,
                 label_path,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        f = open(self.label_path, 'rb')
        self.sample_name, self.label = pickle.load(f)

        # load data
        self.data = np.load(self.data_path,allow_pickle=True)
        self.data = torch.tensor(self.data)

        #debug只使用前100个json文件
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

        #对数据进行序列化处理
        #原先tensor中的数据按照行优先的顺序排成一个一维的数据,然后按照参数组合成其他维度的tensor
        self.seqdata = self.data.permute(0, 4, 2, 3, 1)
        self.seqdata = torch.reshape(self.seqdata, (self.N * self.M, self.T , self.V * self.C))
        self.label = self.label + self.label

        #生成全标签列表[0 - 399],共400个标签
        #self.all_labels = [i for i in range(400)]
        #self.neg_labels = self.all_labels.remove(self.label)

    #返回数据集长度 
    def __len__(self):
        return len(self.label)

    #获取对应编号位置的数据
    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])    #读取index对应json文件中的数据，存入numpy数组
        label = self.label[index]    #读取index对应的label

        # processing
        if self.random_choose:
            data_numpy = skeleton.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = skeleton.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = skeleton.random_move(data_numpy)

        return data_numpy, label


    def collate_fn(self, data):          #读取数据集中的数据
        labels = torch.tensor(self.label)        #正确分类标签集合
        #neg_labels = [torch.tensor(self.neg_labels)]      #错误的分类标签
        sentences = [torch.tensor(self.seqdata)]       #句子集合
        lenghts = [torch.tensor(self.T)]         #句子长度
        return ( labels, sentences, lenghts)

#数据载入函数
#shuffle 设置为True在每个时期重新随机播放数据
#drop_last  不能整除时，是否删除最后一个不完整的批次
#batch_size  单批次数据集大小
def get_data_loader(config, data_path,label_path,shuffle = True, drop_last = False, batch_size = None):        
    dataset = data_set(data_path,label_path)
    if batch_size == None:
        batch_size = min(config['batch_size'], dataset.N * dataset.V)      #取配置项和数据长度中的最小值
    else:
        batch_size = min(batch_size, dataset.N * dataset.V)       #取batch size和数据长度中的最小值
    data_loader = DataLoader(         #Torch DataLoader函数
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        pin_memory = True,
        num_workers = config['num_workers'],     #用于数据加载的子进程数
        collate_fn = dataset.collate_fn,        #合并样本列表以形成张量，批量载入数据时使用
        drop_last = drop_last)
    return data_loader

if __name__ == '__main__':
    #载入配置文件
    f = open("config/config_fewrel.json", "r")
    config = json.loads(f.read())
    f.close()
    data_path = '/home/yangruiling/mmskeleton/data/Kinetics/kinetics-skeleton/val_data.npy'
    label_path = '/home/yangruiling/mmskeleton/data/Kinetics/kinetics-skeleton/val_label.pkl'
    M = get_data_loader(config, data_path, label_path)
    label = np.array(M.dataset.label)  #(39592,)
    data = np.array(M.dataset.data)   #(19796, 3, 300, 18, 2)
    seqdata = np.array(M.dataset.seqdata)   #(39592, 300, 54)
    print(label.shape,data.shape,seqdata.shape)