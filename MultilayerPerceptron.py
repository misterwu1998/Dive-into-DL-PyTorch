import torch
from torch import nn
from torch.nn import init
import numpy as np
import d2lzh_pytorch as d2l
import myUtil

def mlp(num_inputs, num_outputs, num_hiddens):
    '''
    num_inputs：输入的特征数目，比如手写数字图片MNIST的像素个数（二维上分布的像素摊平为一条向量）。
    num_outputs：输出的类别数目。
    num_hiddens：隐藏层单元数目。
    返回：多层感知机。softmax运算被合并到交叉熵损失函数中。
        子层：flatten、hidden、activate、linear。
        多层感知机.forward()索求(样本数目,……（若干个维度）)的Tensor，
        返回(样本数目,num_outputs)的Tensor，表示某样本（某行）被模型判定为某一类（某列）的概率。
    '''
    net=nn.Sequential()
    net.add_module('flatten',d2l.FlattenLayer())
    net.add_module('hidden',nn.Linear(num_inputs,num_hiddens))
    net.add_module('activate',nn.ReLU())
    net.add_module('linear',nn.Linear(num_hiddens,num_outputs))
    for p in net.parameters():
        init.normal_(p,mean=0,std=0.01)
    return net

def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
            params=None, lr=None, optimizer=None):
    '''
    CPU上训练。
    net：多层感知机
    loss：损失函数对象。
    params：待学习的参数Tensor对象的list。
    optimizer：优化算法对象。
    '''
    d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr,optimizer)

if __name__ == "__main__":
    num_inputs, num_outputs, num_hiddens = 784, 10, 256
    net=mlp(num_inputs,num_outputs,num_hiddens)
    batch_size = 256
    train_iter, test_iter = myUtil.load_data_fashion_mnist(batch_size)
    loss = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

    num_epochs = 5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
