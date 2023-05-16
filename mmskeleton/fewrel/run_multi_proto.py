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

def get_memory(config, model, proto_set):      #读取已存储的关系
    memset = []
    resset = []
    rangeset= [0]
    for i in proto_set:
        memset += i
        rangeset.append(rangeset[-1] + len(i))
    data_loader = get_simple_data_loader(config, memset, False, False)
    features = []
    for step, (labels, sentences, lengths) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
    features = np.concatenate(features)
    protos = []
    print ("proto_instaces:%d"%len(features))
    for i in range(len(proto_set)):
        protos.append(torch.tensor(features[rangeset[i]:rangeset[i+1],:].mean(0, keepdims = True)))
    protos = torch.cat(protos, 0)
    return protos

# Use K-Means to select what samples to save, similar to at_least = 0      使用K-Means聚类来选择哪些关系需要更新
#mem_data：一个空数据集合
#proto_memory 标准数据集
#config 配置文件
#model train_simple_model --> proto_softmax_layer.get_feature()  --> rep = sentence_encoder(sentences, length)
#training_data 训练集数据
#num_sel_data = 10
def select_data(mem_set, proto_set, config, model, sample_set, num_sel_data):
    data_loader = get_simple_data_loader(config, sample_set, False, False)
    features = []
    labels = []
    for step, (labels, sentences, lengths) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)   #获取sentence编码后的特征向量 返回 rep numpy数组   lstm
        features.append(feature)              #将特征向量存入数组
    features = np.concatenate(features)        #特征数组拼接
    num_clusters = min(num_sel_data, len(sample_set))      #聚类中心个数 = （num_sel_data）10和训练集长度的最小值
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)    #使用KMean计算距离
    for i in range(num_clusters):
        #argmax(a, axis=None, out=None)   返回的是沿轴axis最大值的索引值（将向量平铺，输出最大值对应的索引）
        # a 表示array
        # axis 表示指定的轴，默认是None，表示把array平铺，
        # out 默认为None，如果指定，那么返回的结果会插入其中
        sel_index = np.argmin(distances[:,i])
        instance = sample_set[sel_index]     #在现有的训练集列表中找出该索引对应的关系
        mem_set.append(instance)    #将该关系存储到mem_set集合中
        proto_set[instance[0]].append(instance)     #并将该关系转存为一个新的元组
    return mem_set

# Use K-Means to select what samples to save
def select_data_twice(mem_set, proto_set, config, model, sample_set, num_sel_data, at_least = 3):
    data_loader = get_simple_data_loader(config, sample_set, False, False)
    features = []
    for step, (_, sentences, lengths) in enumerate(data_loader):
        feature = model.get_feature(sentences, lengths)
        features.append(feature)
    features = np.concatenate(features)
    num_clusters = min(num_sel_data, len(sample_set))
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)
    rel_info = {}
    rel_alloc = {}
    for index, instance in emumerate(sample_set):
        if not instance[0] in rel_info:
            rel_info[instance[0]] = []
            rel_alloc[instance[0]] = 0
        rel_info[instance[0]].append(index)
    for i in range(num_clusters):
        sel_index = np.argmin(distances[:,i])
        instance = sample_set[sel_index]
        rel_alloc[instance[0]] += 1
    rel_alloc = [(i, rel_alloc[i]) for i in rel_alloc]
    at_least = min(at_least, num_sel_data // len(rel_alloc))
    while True:
        rel_alloc = sorted(rel_alloc, key=lambda num : num[1], reverse = True)
        if rel_alloc[-1][1] >= at_least:
            break
        index = 0
        while rel_alloc[-1][1] < at_least:
            if rel_alloc[index][1] <= at_least:
                index = 0
            rel_alloc[-1][1] += 1
            rel_alloc[index][1] -= 1
            index+=1
    print (rel_alloc)
    for i in rel_alloc:
        label = i[0]
        num = i[1]
        tmp_feature = features[rel_info[label]]
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(tmp_feature)

    mem_set.append(instance)
    proto_set[instance[0]].append(instance)
    return mem_set

def train_simple_model(config, model, train_set, epochs):       #训练样例模型  model = proto_softmax_layer()对象
    data_loader = get_simple_data_loader(config, train_set)

    #在使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval；
    #model.eval()，pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值，在模型测试阶段使用
    #model.train() 让model变成训练模式，此时 dropout和batch normalization的操作在训练q起到防止网络过拟合的问题，在模型训练阶段使用
    model.train()
    criterion = nn.CrossEntropyLoss()     #交叉商损失函数
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])    #使用Adam优化器进行梯度下降
    for epoch_i in range(epochs):
        losses = []
        for step, (labels , sentences, lengths) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            logits, _ = model(sentences, lengths)
            labels = labels.to(config['device'])
            loss = criterion(logits, labels)
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])   #梯度裁剪
            optimizer.step()
        print ('loss_mean:',np.array(losses).mean())    #输出损失的平均值  #numpy.mean()  返回数组元素的平均值   
    return model
#torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2) 
#进行梯度裁剪，返回单个向量
#parameters – 一个基于变量的迭代器，会进行归一化
#max_norm (float or int) – 梯度的最大范数     #max_grad_norm = 1
#norm_type(float or int) – 规定范数的类型，默认为L2


#model = train_simple_model() 输出的结果
#单独使用新的关系对模型进行训练
def train_model(config, model, mem_set, epochs, current_proto):        #训练模型
    data_loader = get_simple_data_loader(config, mem_set, batch_size = 5)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    for epoch_i in range(epochs):
        # current_proto = get_memory(config, model, proto_memory)
        model.set_memorized_prototypes(current_proto)
        losses = []
        for step, (labels, sentences, lengths) in enumerate(tqdm(data_loader)):
            model.zero_grad()
            logits, rep = model(sentences, lengths)
            logits_proto = model.mem_forward(rep)
            labels = labels.to(config['device'])
            loss = (criterion(logits_proto, labels))
            loss.backward()
            losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
    return model

if __name__ == '__main__':
    #载入配置文件
    f = open("config/config_kinetic.json", "r")
    #f = open("config/config_NTU-RGB-D-xview.json", "r")
    #f = open("config/config_NTU-RGB-D-xsub.json", "r")
    config = json.loads(f.read())
    f.close()
    config['device'] = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
    config['n_gpu'] = torch.cuda.device_count()
    config['batch_size_per_step'] = int(config['batch_size'] / config["gradient_accumulation_steps"])
    
    #设置根目录
    root_path = '.'
    encoder = lstm_encoder(      #初始化编码
        word_size = config['joint_sizes']*3,    #每个单词对应的的嵌入向量的长度   （#25个关节点*3）
        max_length = 300, 
        pos_size = None, 
        hidden_size = config['hidden_size'], 
        dropout = 0, 
        bidirectional = True, 
        num_layers = 1, 
        config = config)
    sampler = data_sampler(config,None)    #载入数据集
    model = proto_softmax_layer(
        encoder, 
        num_class = config['num_clusters'],      #动作类别数
        id2rel = sampler.id2rel, 
        drop = 0, 
        config = config)
    model = model.to(config["device"])     #将模型送入GPU

    # if torch.cuda.is_available() and config['n_gpu'] > 1:
    #     model = nn.DataParallel(model)
    
    printer = outputer()
    for i in range(1):       #循环迭代5次
        
        set_seed(config, config['random_seed'] + 100 * i)
        sampler.set_seed(config['random_seed'] + 100 * i)

        sequence_results = []
        result_whole_test = []
        mem_data = []       #存储需要修改的/更新的旧关系   需要加入的新关系三元组
        proto_memory = []
        num_class = config['num_clusters']
        #sampler.id2rel 标准关系名称集合
        #sampler.id2rel_pattern 转存为列表形式数据集（序号i，[i]，tokens，length）
        #for循环：打乱关系
        for i in range(num_class):
            proto_memory.append([sampler.id2rel_pattern[i]])
        
        #从sample对象中读取数据集,每读取一个数据集训练后都会对分类准确率进行一次评估，直到60个分类的数据集全部送入网络后，每个类别分类的准确率。最终的准确率是每个类别准确率的平均值
        for steps, (training_data, valid_data, test_data, test_all_data, seen_relations, current_relations) in enumerate(sampler):
            mem_data_back = mem_data.copy()
            proto_memory_back = []
            #再次打乱关系
            for i in range(num_class):
                proto_memory_back.append((proto_memory[i]).copy())

            model = train_simple_model(config, model, mem_data + training_data, 1)
            #使用K-Means聚类来选择哪些关系需要更新
            #mem_data：一个空数据集合
            #proto_memory 转存为元组形式的关系名称和对应id（存储标准数据集）
            #config 配置文件
            #model train_simple_model
            #training_data 训练集数据   长度为60
            select_data(mem_data, proto_memory, config, model, training_data, config['task_memory_size'])

            for i in range(2):
                #读取一存储的关系
                current_proto = get_memory(config, model, proto_memory)
                # model.set_memorized_prototypes(current_proto)
                #将新的关系加入数据集重新训练train_simple_model
                model = train_simple_model(config, model, mem_data + training_data, 1)
                # 单独使用新的关系数据对模型进行训练
                model = train_model(config, model, mem_data, 1, current_proto)
            
            # mem_data = mem_data_back
            # proto_memory = proto_memory_back
            # select_data(mem_data, proto_memory, config, model, training_data, config['task_memory_size'])

            current_proto = get_memory(config, model, proto_memory)    #读取已存储的关系，与新关系进行比对
            model.set_memorized_prototypes(current_proto)
            results = [evaluate_model(config, model, item, num_class) for item in test_data]    #单batch testdata评估模型   #使用 test_data 基于 neg_label评估模型
            #print ((np.array(results)).mean())     #结果列表的平均值
            #printer.print_list(results)      #输出得分列表，保留小数点后三位
            sequence_results.append(np.array(results))
            result_whole_test.append(evaluate_model(config, model, test_all_data, num_class))   #整个testdata评估模型
        # store the result
        printer.append(sequence_results, result_whole_test)     
        # initialize the models 
        torch.save(model,config['model_save_path'])    #保存完整模型结构
        model = model.to('cpu')
        del model
        torch.cuda.empty_cache()   #清除缓存
        # encoder = lstm_encoder( 
        #     word_size = 75, 
        #     max_length = 300, 
        #     pos_size = None, 
        #     hidden_size = config['hidden_size'], 
        #     dropout = 0, 
        #     bidirectional = True, 
        #     num_layers = 1, 
        #     config = config)
        # model = proto_softmax_layer(
        #     sentence_encoder = encoder, 
        #     num_class = 60, 
        #     id2rel = sampler.id2rel, 
        #     drop = 0, 
        #     config = config)
        # model.to(config["device"])
    # output the final avg result
    printer.output()

