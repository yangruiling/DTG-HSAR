import torch
import torch.nn as nn
from lifelong.data.sampler import data_sampler
import json
from torch.utils.data import Dataset, DataLoader

class simple_data_set(Dataset):   #定义数据集  Torch包 Dataset

    def __init__(self, data):     #初始化数据集
        self.data = data

    def __len__(self):       #返回数据集长度
        return len(self.data)

    def __getitem__(self, idx):      #返回数据集编号
        return self.data[idx]

    def collate_fn(self, data):          #定义将一个多个样本拼接成一个batch的方式
        #print(data.__len__())    #50
        #print(data[0][0],data[0][1])     #112 tensor([[],[],[]])
        
        if data.__len__() == 1:
            labels = torch.tensor(data[0])
            sentences = [torcher.tensor(data[1])]
            lenghts = [300]
        else:
            lengths_list = []
            for i in range(data.__len__()):
                lengths_list.append(300)
            labels = torch.tensor([item[0] for item in data])        #id   
            sentences = [item[1].clone().detach() for item in data]    #句子
            #sentences = [torch.tensor(item[1]) for item in data]       #句子
            lenghts = [torch.tensor(item) for item in lengths_list]

        return (
            labels,
            sentences,
            lenghts
        )

#数据载入函数
#shuffle 设置为True在每个时期重新随机播放数据
#drop_last  不能整除时，是否删除最后一个不完整的批次
#batch_size  单批次数据集大小
def get_simple_data_loader(config, data, shuffle = True, drop_last = False, batch_size = None):        
    dataset = simple_data_set(data)
    if batch_size == None:
        batch_size = min(config['batch_size'], len(data))      #取配置项和数据长度中的最小值
    else:
        batch_size = min(batch_size, len(data))       #取batch size和数据长度中的最小值
    data_loader = DataLoader(         #Torch DataLoader函数
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        pin_memory = True,
        num_workers = config['num_workers'],     #用于数据加载的子进程数
        collate_fn = dataset.collate_fn,        #合并样本列表以形成张量，批量载入数据时使用
        drop_last = drop_last)
    return data_loader

'''
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, 
num_workers=0, collate_fn=default_collate, pin_memory=False, 
drop_last=False)

dataset：加载的数据集(Dataset对象)
batch_size：batch size   =   400
shuffle:：是否将数据打乱
sampler： 样本抽样，后续会详细介绍
num_workers：使用多进程加载的进程数，0代表不使用多进程
collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃

'''


if __name__ == '__main__':
    #载入配置文件
    f = open("config/config_fewrel.json", "r")
    config = json.loads(f.read())
    f.close()
    data_path = '/home/yangruiling/mmskeleton/data/Kinetics/kinetics-skeleton/val_data.npy'
    label_path = '/home/yangruiling/mmskeleton/data/Kinetics/kinetics-skeleton/val_label.pkl'
    sampler = data_sampler(config,None)
    # for steps, (training_data, valid_data, test_data, test_all_data, seen_relations, current_relations) in enumerate(sampler):
    #     data_loader = get_simple_data_loader(config, training_data, False, False)
    #     for step, (labels, sentences , lenghts) in enumerate(data_loader):
    #         print(len(labels),len(sentences))
    data_loader = get_standard_data_loader(config,sampler.id2rel_pattern,False,False)
    for step, (labels, sentences , lenghts) in enumerate(data_loader):
        print(len(labels),len(sentences))