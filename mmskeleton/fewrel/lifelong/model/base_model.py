import torch
import torch.nn as nn
import os
import json
import numpy as np

class base_model(nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        #参数Parameter
        #torch.Tensor([0]) 从原始数据直接生成张量 tensor([0.])
        #nn.Parameter(data,require_grad） 从tensor构建参数列表 require_grad默认为True

        #定义参数 常量0，常量pi
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    #载入检查点预训练文件
    #Module.load_state_dict(state_dict): 用来加载模型参数 字典格式
    #Module.eval() 将模型设置成evaluation(评估)模式，只影响dropout和batchNorm
    #python os.join模块，用于获取文件属性    os.path.join()把目录和文件名合成一个路径
    #torch.load()  加载.pt 和.pth格式模型  参考： https://blog.csdn.net/weixin_40522801/article/details/106563354
    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    #保存预训练网络（模型）为.pt 和.pth文件
    #Module.state_dict()返回一个包含模型状态信息的字典。包含参数（weighs and biases）和持续的缓冲值（如：观测值的平均值）。只有具有可更新参数的层才会被保存在模型的 state_dict 数据结构中
    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    #从文件中载入参数
    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        #遍历将参数转化为Tensor
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        #将参数保存到模型权重中去
        self.load_state_dict(parameters, strict = False)
        self.eval()

    #保存参数
    # json.dumps	将 Python 对象编码成 JSON 字符串
    # json.loads	将已编码的 JSON 字符串解码为 Python 对象
    def save_parameters(self, path):
        f = open(path, "w")
        #读取Tensor格式的参数，并转化为Python列表
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    #获取模型参数,返回参数字典
    def get_parameters(self, mode = "numpy", param_dict = None):
        #未指定参数字典时使用默认Module 参数字典
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":    #Tensor -->  numpy
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":      #Tensor -->  list
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:     #Tensor
                res[param] = all_param_dict[param]
        return res

    #设置（自定义）参数
    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()