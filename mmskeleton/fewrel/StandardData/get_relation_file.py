import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt

def get_standard_data_xsub():
    '''文件读取'''
    f = open('/home/yangruiling/mmskeleton/data/NTU-RGB-D/xsub/val_label.pkl', 'rb')
    sample_name,label = pickle.load(f)
    data = np.load('/home/yangruiling/mmskeleton/data/NTU-RGB-D/xsub/val_data.npy',mmap_mode='r')
    standard_name = []
    standard_label = []
    standard_data = []

    for index in range(65):       #类别数  400或60
        for i in range(len(label)):
            if label[i] == index:
                standard_name.append(sample_name[i])
                standard_label.append(label[i])
                standard_data.append(data[i])
                break 
    
    print(standard_data[0])
    print(standard_label)
    print(len(standard_data) , len(standard_label))

    '''文件存储'''

    #将 label 存为 pkl文件
    f = open('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/NTU-RGB-D/xsub/standard_label.pkl','wb')
    standard_label = pickle.dump([standard_name,standard_label],f)
    f.close()
    # standard_data = np.asarray(standard_data)
    #将 data 存为 npy文件
    np.save('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/NTU-RGB-D/xsub/standard_data', standard_data) 

    #查看存储的标准数据
    f = open('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/NTU-RGB-D/xsub/standard_label.pkl','rb')
    sample_name ,label = pickle.load(f)
    f.close()
    print(sample_name[1])
    print(label[1])


def get_standard_data_xview():
    '''文件读取'''
    f = open('/home/yangruiling/mmskeleton/data/NTU-RGB-D/xview/val_label.pkl', 'rb')
    sample_name,label = pickle.load(f)
    data = np.load('/home/yangruiling/mmskeleton/data/NTU-RGB-D/xview/val_data.npy',mmap_mode='r')
    standard_name = []
    standard_label = []
    standard_data = []

    for index in range(65):       #类别数  400或60
        for i in range(len(label)):
            if label[i] == index:
                standard_name.append(sample_name[i])
                standard_label.append(label[i])
                standard_data.append(data[i])
                break

    print(standard_data[0])
    print(standard_label)
    print(len(standard_data) , len(standard_label))
    '''文件存储'''

    #将 label 存为 pkl文件
    f = open('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/NTU-RGB-D/xviwe/standard_label.pkl','wb')
    standard_label = pickle.dump([standard_name,standard_label],f)
    f.close()
    # standard_data = np.asarray(standard_data)
    #将 data 存为 npy文件
    np.save('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/NTU-RGB-D/xviwe/standard_data', standard_data) 
    
    #查看存储的标准数据
    f = open('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/NTU-RGB-D/xviwe/standard_label.pkl','rb')
    sample_name ,label = pickle.load(f)
    f.close()
    print(sample_name[1])
    print(label[1])

def get_standard_data_Kinetic():
    '''文件读取'''
    f = open('/home/yangruiling/mmskeleton/data/Kinetics/kinetics-skeleton/val_label.pkl', 'rb')
    sample_name,label = pickle.load(f)
    data = np.load('/home/yangruiling/mmskeleton/data/Kinetics/kinetics-skeleton/val_data.npy',mmap_mode='r')
    standard_name = []
    standard_label = []
    standard_data = []

    for index in range(400):       #类别数  400或60
        for i in range(len(label)):
            if label[i] == index:
                standard_name.append(sample_name[i])
                standard_label.append(label[i])
                standard_data.append(data[i])
                break
    
    print(standard_data[0])
    print(standard_label)
    print(len(standard_data) , len(standard_label))
    '''文件存储'''

    #将 label 存为 pkl文件
    f = open('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/standard_label.pkl','wb')
    standard_label = pickle.dump([standard_name,standard_label],f)
    f.close()
    # standard_data = np.asarray(standard_data)
    #将 data 存为 npy文件
    np.save('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/standard_data', standard_data) 

    #查看存储的标准数据
    f = open('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/standard_label.pkl','rb')
    sample_name ,label = pickle.load(f)
    f.close()
    print(sample_name[1])
    print(label[1])

'''
    #划分val数据集为3:1，充当训练集和验证集
    train_label = label[0:15000]
    train_data = data[0:15000]
    train_sample_name = sample_name[0:15000]

    val_label = label[15001:19795]
    val_data = data[15001:19795]
    val_sample_name = sample_name[15001:19795]

    #将 label 存为 pkl文件
    f = open('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/train_label.pkl','wb')
    standard_label = pickle.dump([train_sample_name,train_label],f)
    f.close()
    #将 data 存为 npy文件
    np.save('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/train_data', train_data) 
    #将 label 存为 pkl文件
    f = open('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/val_label.pkl','wb')
    standard_label = pickle.dump([val_sample_name,val_label],f)
    f.close()
    #将 data 存为 npy文件
    np.save('/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/val_data', val_data) 
'''


if __name__ == '__main__':
    get_standard_data_xsub()
    get_standard_data_xview()
    get_standard_data_Kinetic()





