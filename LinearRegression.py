import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import random
import d2lzh_pytorch as d2l
import myUtil

class MyLinearRegression:
    def __init__(self,num_inputs):
        '''
        num_inputs：输入的特征数目（对应(样本数目,特征数目)的输入训练数据、(样本数目,1)的标签）
        '''
        super().__init__()
        # 初始化参数
        self.w=torch.tensor(np.random.normal(0, 0.01, (num_inputs,1)), dtype=torch.float32)
            #(num_inputs,1)的FloatTensor
        self.b = torch.zeros(1, dtype=torch.float32)
            #偏移量，标量
        self.w.requires_grad_()
        self.b.requires_grad_()
    def forward(self,X):
        '''
        前向计算。
        X：(样本数目,特征数目)的Tensor
        不关心是否记录梯度，因此测试时应当在“with torch.no_grad():”下调用。
        '''
        return d2l.linreg(X,self.w,self.b)
    def train(self,
              batch_size,
              learning_rate,
              num_epochs,
              features,
              labels,
              loss=d2l.squared_loss,
              opt=d2l.sgd):
        '''
        features：(样本数目,特征数目)的Tensor。
        labels：(特征个数,1)的“正确答案”，也接受仅1维的向量(特征个数,)。
        loss：损失函数。
        opt：优化算法函数。
        '''
        for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
            # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
            # 和y分别是小批量样本的特征和标签
            for X, y in d2l.data_iter(batch_size, features, labels):
                l = loss(self.forward(X), y).sum()  # l是有关小批量X和y的损失
                l.backward()  # 小批量的损失对模型参数求梯度
                opt([self.w, self.b], learning_rate, batch_size)  # 使用小批量随机梯度下降迭代模型参数
                # 不要忘了梯度清零
                self.w.grad.data.zero_()
                self.b.grad.data.zero_()
            train_l = loss(self.forward(features), labels)
            print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

# class LinearNet(nn.Module):
#     def __init__(self, n_feature):
#         '''
#         定义线性回归模型并初始化参数。
#         n_feature：输入的特征数目（对应(样本数目,特征数目)的输入训练数据、(样本数目,1)的标签）。
#         '''
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(in_features=n_feature,
#                                 out_features=1,
#                                 bias=True)
#         # 初始化模型参数
#         nn.init.normal_(self.linear.weight,mean=0,std=0.01)
#         nn.init.zeros_(self.linear.bias)
#     def forward(self, x):
#         '''
#         x：(样本数目,特征数目)的Tensor
#         '''
#         y = self.linear(x)
#         return y

def linearNet(n_feature):
    '''
    返回一个参数已初始化的线性回归模型，其中Linear层的名字是'linear'
    n_feature：输入的特征数目（对应(样本数目,特征数目)的输入训练数据、(样本数目,1)的标签）。
    '''
    net=nn.Sequential()
    net.add_module('linear', nn.Linear(n_feature, 1))
    nn.init.normal_(net.linear.weight,mean=0,std=0.01)
    nn.init.zeros_(net.linear.bias)
    return net

def getDataIter(batch_size,features,labels):
    '''
    features：(样本数目,特征数目)的Tensor。
    labels：(特征个数,1)的“正确答案”，也接受仅1维的向量(特征个数,)。
    返回：打乱的批量数据迭代器。
    '''
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features,labels),
        batch_size=batch_size,
        shuffle=True
    )

def train(net,batch_size,num_epochs,features,labels,loss,optimizer):
    '''
    CPU上训练。
    features：(样本数目,特征数目)的Tensor。
    labels：(特征个数,1)的“正确答案”，也接受仅1维的向量(特征个数,)。
    loss：损失函数对象。
    optimizer：优化器。
    '''
    data_iter=getDataIter(batch_size,features,labels)
    for epoch in range(1, num_epochs + 1):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))

if __name__=='__main__':

    # 数据集
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
    # 定义模型、初始化参数
    net=linearNet(2)
    # 损失函数
    loss=nn.MSELoss()
    # 优化算法
    optimizer=myUtil.getOptimizer(net,(0.03,),opt=torch.optim.SGD)
    print(optimizer)
    # 训练
    train(net,10,5,features,labels,loss,optimizer)

    # # 生成数据集
    # num_inputs = 2 #特征数目
    # num_examples = 1000 #样本数目
    # true_w = [2, -3.4] #权重向量的“正确答案”，即真实值
    # true_b = 4.2 #偏移量的“正确答案”，即真实值
    # features = torch.randn(num_examples, num_inputs,
    #                     dtype=torch.float32) #形状为(num_examples,num_inputs)
    # labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    #     #无噪声情况下的标签；标量
    # labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
    #                     dtype=torch.float32) #掺入噪声
    # # 定义模型、初始化参数
    # net=MyLinearRegression(num_inputs)
    # # 未训练时测试模型
    # with torch.no_grad():
    #     print('original loss:',
    #           torch.sum(d2l.squared_loss(net.forward(features),labels)).item())
    # # 训练模型
    # net.train(batch_size=10,
    #           learning_rate=0.03,
    #           num_epochs=5,
    #           features=features,
    #           labels=labels,
    #           loss=d2l.squared_loss,
    #           opt=d2l.sgd)
    # # 训练后测试模型
    # with torch.no_grad():
    #     print('final loss:',
    #           torch.sum(d2l.squared_loss(net.forward(features),labels)).item())

    # 从零开始实现
    # # 生成数据集
    # num_inputs = 2 #特征数目
    # num_examples = 1000 #样本数目
    # true_w = [2, -3.4] #权重向量的“正确答案”，即真实值
    # true_b = 4.2 #偏移量的“正确答案”，即真实值
    # features = torch.randn(num_examples, num_inputs,
    #                     dtype=torch.float32) #形状为(num_examples,num_inputs)
    # labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    #     #无噪声情况下的标签；标量
    # labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
    #                     dtype=torch.float32) #掺入噪声
    # # 读取数据利用d2l.data_iter()
    # # 初始化模型参数
    # w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
    #     #(num_inputs,1)的FloatTensor
    # b = torch.zeros(1, dtype=torch.float32)
    #     #标量
    # w.requires_grad_(requires_grad=True)
    # b.requires_grad_(requires_grad=True) 
    # # 定义模型为d2l.linreg()
    # # 定义损失函数为d2l.squared_loss()
    # # 定义优化算法为d2l.sgd()
    # # 训练模型
    # batch_size=10
    # lr = 0.03
    # num_epochs = 3
    # net = d2l.linreg
    # loss = d2l.squared_loss
    # for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    #     # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    #     # 和y分别是小批量样本的特征和标签
    #     for X, y in d2l.data_iter(batch_size, features, labels):
    #         l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
    #         l.backward()  # 小批量的损失对模型参数求梯度
    #         d2l.sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

    #         # 不要忘了梯度清零
    #         w.grad.data.zero_()
    #         b.grad.data.zero_()
    #     train_l = loss(net(features, w, b), labels)
    #     print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
    # print('--end--')
    # 从零开始实现
