import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import random
import time
import json
import os
from tqdm import tqdm, trange
from sklearn.cluster import KMeans

import lifelong 
from lifelong.model.encoder import lstm_encoder
from lifelong.model.module import proto_softmax_layer, simple_lstm_layer
from lifelong.data.sampler import data_sampler
from lifelong.utils import set_seed
from lifelong.utils import outputer
from data_loader_twice import get_simple_data_loader

def evaluate_model(config, model, test_set, num_class):     #估计模型
    model.eval()      
    data_loader = get_simple_data_loader(config, test_set, False, False)     #数据载入
    num_correct = 0
    total = 0.0
    for step, (labels, sentences, lengths) in enumerate(data_loader):
        logits, rep = model(sentences, lengths)
        distances = model.get_mem_feature(rep)
        logits = logits
        short_logits = distances
        for index, logit in enumerate(logits):
            score = short_logits[index]#logits[index] + short_logits[index] + long_logits[index]
            total += 1.0
            golden_score = score[labels[index]]
            #print('golden_score:',golden_score)
            max_neg_score = -2147483647.0
            
            #构建错误标签序列
            # neg_labels = []
            # all_labels = list(range(60))
            # for i in range(len(labels)):
            #     # print(test_label)
            #     # print(labels[i])
            #     all_labels.remove(labels[i])
            #     neg_labels.append(all_labels)
            #     all_labels = list(range(60))
            
            # for i in neg_labels[index]: #range(num_class): 
            #     if (i != labels[index]) and (score[i] > max_neg_score):
            #         max_neg_score = score[i]
            score.sort()
            neg_label_score = []
            if len(score) > 30:
                for i in range(30):
                    neg_label_score.append(score[i])
            else:
                neg_label_score = score
            max_neg_score = np.mean(neg_label_score)
            # max_neg_score = np.median(neg_label_score)        
            #print('max_neg_score:',max_neg_score)
            if golden_score > max_neg_score:
                num_correct += 1
    #print('numcorrect:',num_correct)
    #print('total:',total)
    return num_correct / total       #在每一个分类测试数据集上的准确率

if __name__ == '__main__':
    #载入配置文件
    # f = open("/home/yangruiling/mmskeleton/mmskeleton/fewrel/config/config_kinetic.json", "r")
    f = open("/home/yangruiling/mmskeleton/mmskeleton/fewrel/config/config_NTU-RGB-D-xview.json", "r")
    #f = open("/home/yangruiling/mmskeleton/mmskeleton/fewrel/config/config_NTU-RGB-D-xsub.json", "r")
    config = json.loads(f.read())
    f.close()
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])

    #设置根目录
    root_path = '.'
    sampler = data_sampler(config,None)    #载入数据集
    #model = torch.load('/home/yangruiling/.cache/torch/checkpoints/model.pth')  #初始化新的模型对象
    model = torch.load('/home/yangruiling/.cache/torch/checkpoints/model_xview.pth')  #初始化新的模型对象
    model = model.to(config["device"])     #将模型送入GPU
    print(model.state_dict())   #查看模型参数

    printer = outputer()
        
    set_seed(config, config['random_seed'] + 100 )
    sampler.set_seed(config['random_seed'] + 100 )

    results = []
    num_class = 400
    i = 1
    #从sample对象中读取数据集,每读取一个数据集训练后都会对分类准确率进行一次评估，直到60个分类的数据集全部送入网络后，每个类别分类的准确率。最终的准确率是每个类别准确率的平均值
    for steps, (training_data, valid_data, test_data, test_all_data, seen_relations, current_relations) in enumerate(sampler):
        result = evaluate_model(config, model, test_all_data, num_class)
        results.append(round(result,4))    #单batch testdata评估模型   #使用 test_data 基于 neg_label评估模型
        print ("class"+str(i)+"_Top1: ",round(result,4))     #结果列表的平均值
        i = i+1
    mean = round(np.array(results).mean(),4)
    print("avreage Top1: ",mean)  
    doc = open('/home/yangruiling/mmskeleton/static/draw_data/longlife—kinetic.txt', 'w')
    doc.write(str([results,[mean]]))
    doc.close()
    # initialize the models 
    model = model.to('cpu')
    del model

