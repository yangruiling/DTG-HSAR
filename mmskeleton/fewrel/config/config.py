import torch
import json

CONFIG_xsub= {
    'learning_rate': 0.001,      #学习率
    'embedding_dim': 300,        #嵌入维度
    'hidden_size': 200,          #隐藏层大小
    'batch_size': 50,            #批次大小
    'gradient_accumulation_steps':1,       #梯度累计
    'num_clusters': 60,      #Kmeans 聚类算法的质心数量
    'epoch': 2,        #迭代次数
    'random_seed': 100,
    'task_memory_size': 10,
    'loss_margin': 0.5,
    'sequence_times': 5,
    'lambda': 100,
    'num_cands': 10,
    'num_steps': 1,
    'num_constrain': 10,        
    'data_per_constrain': 5,
    'lr_alignment_model': 0.0001,
    'epoch_alignment_model': 20,
    'checkpoint_path': 'checkpoint',
    'use_gpu': True,          #是否使用GPU
    'num_workers':4,          #用于数据加载的子进程数
    'max_grad_norm':1,
    'joint_sizes':25,
    'task_name':'NTU-RGB-D-xsub',
    'model_save_path':'/home/yangruiling/.cache/torch/checkpoints/model.pth',
    'val_data_path':'/home/yangruiling/mmskeleton/data/NTU-RGB-D/xsub/val_data.npy',
	'val_label_path':'/home/yangruiling/mmskeleton/data/NTU-RGB-D/xsub/val_label.pkl',
	'train_data_path':'/home/yangruiling/mmskeleton/data/NTU-RGB-D/xsub/train_data.npy',
	'train_label_path':'/home/yangruiling/mmskeleton/data/NTU-RGB-D/xsub/train_label.pkl',
	'standard_data_path':'/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/NTU-RGB-D/xsub/standard_data.npy',
	'standard_laben_path':'/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/NTU-RGB-D/xsub/standard_label.pkl',
}

CONFIG1_xview= {
    'learning_rate': 0.001,      #学习率
    'embedding_dim': 300,        #嵌入维度
    'hidden_size': 200,          #隐藏层大小
    'batch_size': 50,            #批次大小
    'gradient_accumulation_steps':1,       #梯度累计
    'num_clusters': 60,      #Kmeans 聚类算法的质心数量
    'epoch': 2,        #迭代次数
    'random_seed': 100,
    'task_memory_size': 10,
    'loss_margin': 0.5,
    'sequence_times': 5,
    'lambda': 100,
    'num_cands': 10,
    'num_steps': 1,
    'num_constrain': 10,        
    'data_per_constrain': 5,
    'lr_alignment_model': 0.0001,
    'epoch_alignment_model': 20,
    'checkpoint_path': 'checkpoint',
    'use_gpu': True,          #是否使用GPU
    'num_workers':4,          #用于数据加载的子进程数
    'max_grad_norm':1,
    'joint_sizes':25,
    'task_name':'NTU-RGB-D-xview',
    'model_save_path':'/home/yangruiling/.cache/torch/checkpoints/model_xview.pth',
    'val_data_path':'/home/yangruiling/mmskeleton/data/NTU-RGB-D/xview/val_data.npy',
	'val_label_path':'/home/yangruiling/mmskeleton/data/NTU-RGB-D/xview/val_label.pkl',
	'train_data_path':'/home/yangruiling/mmskeleton/data/NTU-RGB-D/xview/train_data.npy',
	'train_label_path':'/home/yangruiling/mmskeleton/data/NTU-RGB-D/xview/train_label.pkl',
	'standard_data_path':'/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/NTU-RGB-D/xviwe/standard_data.npy',
	'standard_laben_path':'/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/NTU-RGB-D/xviwe/standard_label.pkl',
}

CONFIG1_Kinetic= {
    'learning_rate': 0.001,      #学习率
    'embedding_dim': 300,        #嵌入维度
    'hidden_size': 200,          #隐藏层大小
    'batch_size': 50,            #批次大小
    'gradient_accumulation_steps':1,       #梯度累计
    'num_clusters': 400,      #Kmeans 聚类算法的质心数量
    'epoch': 2,        #迭代次数
    'random_seed': 100,
    'task_memory_size': 10,
    'loss_margin': 0.5,
    'sequence_times': 5,
    'lambda': 100,
    'num_cands': 10,
    'num_steps': 1,
    'num_constrain': 10,        
    'data_per_constrain': 5,
    'lr_alignment_model': 0.0001,
    'epoch_alignment_model': 20,
    'checkpoint_path': 'checkpoint',
    'use_gpu': True,          #是否使用GPU
    'num_workers':4,          #用于数据加载的子进程数
    'max_grad_norm':1,
    'joint_sizes':18,   #关节点数目,
    'task_name':'Kinetic',
    'model_save_path':'/home/yangruiling/.cache/torch/checkpoints/model_kinetic.pth',
    'val_data_path':'/home/yangruiling/mmskeleton/data/Kinetics/kinetics-skeleton/val_data.npy',
	'val_label_path':'/home/yangruiling/mmskeleton/data/Kinetics/kinetics-skeleton/val_label.pkl',
	'train_data_path':'/home/yangruiling/mmskeleton/data/Kinetics/kinetics-skeleton/train_data.npy',
	'train_label_path':'/home/yangruiling/mmskeleton/data/Kinetics/kinetics-skeleton/train_label.pkl',
    # 'val_data_path':'/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/val_data.npy',
	# 'val_label_path':'/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/val_label.pkl',
	# 'train_data_path':'/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/train_data.npy',
	# 'train_label_path':'/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/train_label.pkl',
	'standard_data_path':'/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/standard_data.npy',
	'standard_laben_path':'/home/yangruiling/mmskeleton/mmskeleton/fewrel/StandardData/Kinetic/standard_label.pkl',
}

# 将配置写入json文件
# f = open("config_NTU-RGB-D-xsub.json", "w")
# f.write(json.dumps(CONFIG_xsub))
# f.close()


# f = open("config_NTU-RGB-D-xview.json", "w")
# f.write(json.dumps(CONFIG1_xview))
# f.close()

#将配置写入json文件
f = open("config_kinetic.json", "w")
f.write(json.dumps(CONFIG1_Kinetic))
f.close()