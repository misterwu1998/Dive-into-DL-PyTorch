import torch
from torch import nn
from torch.nn import init
import numpy as np
import d2lzh_pytorch as d2l
import myUtil

def softmaxNet(num_inputs,num_outputs):
    '''
    初始化并返回一个softmax回归模型。
    模型.forward()索求(样本数目,……（若干个维度）)的Tensor对象，
    返回(样本数目,num_outputs)的Tensor，表示某样本（某行）被模型判定为某一类（某列）的概率。
    flatten层将第0维之外的其余维度坍缩为一个维度，即使用行向量表示一个样本。
    num_inputs：输入的特征数目（一个样本中元素的总和），
        比如手写数字图片MNIST的像素个数（二维上分布的像素摊平为一条向量）。
    num_outputs：输出的类别数目。
    该模型名义上是softmax回归，但是PyTorch将softmax运算与交叉熵损失函数合并了，
    所以实际上并没有包含softmax运算。
    '''
    net=nn.Sequential()
    net.add_module('flatten',d2l.FlattenLayer())
    net.add_module('linear',nn.Linear(num_inputs,num_outputs))
    init.normal_(net.linear.weight,mean=0,std=0.01)
    init.zeros_(net.linear.bias)
    return net

def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
            params=None, lr=None, optimizer=None):
    '''
    在CPU上训练
    net：softmaxNet。
    loss：损失函数对象。
    params：待学习的参数Tensor对象的list。
    optimizer：优化算法对象。
    '''
    d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr,optimizer)

if __name__ == "__main__":
    
    # 数据集
    batch_size = 256
    train_iter, test_iter = myUtil.load_data_fashion_mnist(batch_size)
    # 定义模型、初始化参数
    num_inputs = 784 #输入的特征数目，即像素个数（平摊到一条向量上）
    num_outputs = 10 #输出的类别数目
    net=softmaxNet(num_inputs,num_outputs)
    # softmax运算+交叉熵损失函数
    loss = nn.CrossEntropyLoss()
    # 优化算法
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    # 训练
    num_epochs = 5
    train(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
