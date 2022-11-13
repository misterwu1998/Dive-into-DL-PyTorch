import torch
from torch import nn
import numpy as np
import d2lzh_pytorch as d2l
from torchvision import transforms

def getOptimizer(net,lr,opt=torch.optim.SGD):
    '''
    net：nn.ModuleList或nn.Sequential对象
    lr：按顺序指定各子层的学习率的tuple；如果仅含1个元素，就统一各子层的学习率。
    opt：优化算法函数对象的构造器
    返回：针对不同子层有不同学习率的优化器
    '''
    if len(lr)==1:
        return opt(params=net.parameters(),lr=lr[0])
    kw=[]
    for i in range(len(net)):
        kw.append({'params':net[i].parameters(),'lr':lr[i]})
    return opt(kw)

def load_data_fashion_mnist(batch_size, resize=None, root='./Datasets/FashionMNIST'):
    '''
    加载FASHION_MNIST数据集。
    resize：期望把图片重整为什么形状；tuple，(高度,宽度)
    返回：train_iter, test_iter。用法：“for 批次,标签 in 迭代器:”。
        其中，批次是(batch_size,1（通道数目）,高度,宽度)的Tensor，
        标签是(batch_size,1)的Tensor。
    '''
    return d2l.load_data_fashion_mnist(batch_size,resize,root)

def saveModel(model : nn.Module, name : str) -> str:
    '''
    把模型的参数保存为当前目录下的models目录下指定名称（免后缀）的文件。
    model：nn.Module对象。
    name：模型名称。
    返回：文件的相对路径。
    '''
    path='./models/'+name+'.pt'
    torch.save(model.state_dict(),path)
    return path

def loadModel(model:nn.Module,name:str)->nn.Module:
    '''
    加载当前目录下的models目录下指定名称（免后缀）的文件到模型model中。
    '''
    model.load_state_dict(torch.load('./models/'+name+'.pt'))

def get_data_ch7(path):  
    '''
    path:本书GitHub项目中的'data/airfoil_self_noise.dat'文件下载路径。
    '''
    # '../../data'就是本书GitHub项目中的data目录，由于项目未下载，改为手动指定文件路径。
    # data = np.genfromtxt('../../data/airfoil_self_noise.dat', delimiter='\t')
    data=np.genfromtxt(fname=path,delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return (torch.tensor(data[:1500, :-1], 
        dtype=torch.float32), 
        torch.tensor(data[:1500, -1], 
        dtype=torch.float32) # 前1500个样本(每个样本5个特征)
    )

def preprocession_pretrainedModel():
    '''
    按照 torchvision.models包 的要求预处理输入数据。
    返回相应的Tranform对象（用法：函数对象），
        也可可将其添加到Transform对象的list中交给 transforms.Compose()，
        以实现多种预处理的组合。
    '''
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
