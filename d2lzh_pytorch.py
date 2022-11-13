import torch
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.utils.data as Data
# import torch.nn as nn
from torch import nn
from torch.nn import init
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
import zipfile
import math

def use_svg_display():
    # # 用矢量图显示
    # display.set_matplotlib_formats('svg')
    '''无法使用'''
    pass

def set_figsize(figsize=(3.5, 2.5)):
    # use_svg_display()
    # # 设置图的尺寸
    # plt.rcParams['figure.figsize'] = figsize
    '''无法使用'''
    pass

def data_iter(batch_size, features, labels):
    '''
    generator，每次返回batch_size（批量大小）个随机样本的特征和标签。
    features：形状为(样本数目,……（若干个维度）)；
        对于线性回归，形状为(样本数目,特征个数)。
    labels：形状为(样本数目,……（若干个维度）)；
        对于线性回归，形状为(样本数目,1)。
    返回：形状为(batch_size（最后一批可能没这么多）,……（若干个维度）)的features、labels。
    '''
    num_examples = len(features)
    indices = list(range(num_examples)) #0,1,...,num_examples-1的list
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
            #（剩余元素足够的话）j是含batch_size个元素的LongTensor，元素用于“点兵”
        yield  features.index_select(0, j), labels.index_select(0, j) 

def linreg(X, w, b):  # 计算线性回归
    '''
    X：(样本数目,特征数目)
    w：(特征数目,1)
    b：(样本数目,1)；也接收标量（利用广播机制）
    返回：(样本数目,1)的Tensor
    '''
    return torch.mm(X, w) + b
    
def squared_loss(y_hat, y):  #损失函数
    '''
    y_hat：(样本数目,1)的预测结果。
    y：(样本数目,1)的“正确答案”，也接收仅1维的向量(样本数目,)。
    返回：与y_hat同形状的向量；另外, pytorch里的MSELoss并没有除以 2
    '''
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用	
    '''
    优化算法；迭代模型参数
    params：要‘学习’的参数的list，如[w向量,b]
    lr：学习率超参数；该函数把lr“平摊”到batch的各样本上去
    '''
    for param in params:
        param.data -= lr * param.grad / batch_size # 注意这里更改param时用的param.data

def get_fashion_mnist_labels(labels):
    '''
    传入各样本的label数字，
    返回fashion_mnist数据集的各label文本的list
    '''
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
    
def show_fashion_mnist(images, labels):
    # '''
    # 传入各图片Tensor,各label文本，
    # 画出多张图像和对应标签
    # '''
    # d2l.use_svg_display()
    # # 这里的_表示我们忽略（不使用）的变量
    # _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    # for f, img, lbl in zip(figs, images, labels):
    #     f.imshow(img.view((28, 28)).numpy())
    #     f.set_title(lbl)
    #     f.axes.get_xaxis().set_visible(False)
    #     f.axes.get_yaxis().set_visible(False)
    # plt.show()
    '''无法使用'''
    pass
    
#def load_data_fashion_mnist():
def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    '''
    完整实现。
    Download the fashion mnist dataset and then load into memory.
    '''
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_iter, test_iter

#def evaluate_accuracy(data_iter, net):
def evaluate_accuracy(data_iter, net, device=None):
    '''
    计算准确度。
    用指定的device来加速计算，如果没指定device就使用net的device；
    完事后data_iter中的数据留在device，计算结果在CPU。
    data_iter：torch.utils.data.DataLoader对象。用法：“for X, y in data_iter:”。
        X是样本，要求形状适用于net()，y是样本的标签，要求与net(X)同形状。
    该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述。
    '''

    # 改用GPU来加速计算，完事后data_iter中的数据留在GPU，计算结果在CPU
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

    #acc_sum, n = 0.0, 0
    #for X, y in data_iter:
    #	if isinstance(net, torch.nn.Module):
    #		net.eval() # 评估模式, 这会关闭dropout
    #		acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
    #		net.train() # 改回训练模式
    #	else: # 自定义的模型
    #		if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
    #			# 将is_training设置成False
    #			acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
    #		else:
    #			acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
    #	n += y.shape[0]
    #return acc_sum / n

    #acc_sum, n = 0.0, 0
    #for X, y in data_iter:
    #	acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
    #	n += y.shape[0]
    #return acc_sum / n

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
            params=None, lr=None, optimizer=None):
    '''
    loss：损失函数对象。
    params：待学习的参数Tensor对象的list。
    optimizer：优化算法对象。
    '''
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                # d2l.sgd(params, lr, batch_size)
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
            % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

class FlattenLayer(nn.Module):
    '''
    摊平一个样本的其余维度，用一个行向量表示一个样本。
    用法：函数对象、nn层
    '''
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
            legend=None, figsize=(3.5, 2.5)):
    '''
    作图，y轴使用对数尺度
    '''
    # d2l.set_figsize(figsize)
    # d2l.plt.xlabel(x_label)
    # d2l.plt.ylabel(y_label)
    # d2l.plt.semilogy(x_vals, y_vals)
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        # d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        # d2l.plt.legend(legend)
        plt.semilogy(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)

def corr2d(X, K):  
    '''
    二维互相关运算，用于替代矩阵卷积。
    卷积本来要求将核矩阵上下翻转、左右翻转然后才进行互相关运算，
    但深度学习中卷积核都是“学”出来的，“学”出来的卷积核又得翻转后再互相关运算，
    那不如直接“学”互相关运算所用的核矩阵。
    因此本书直接用互相关运算所用的核矩阵替代原本定义的卷积核。
    '''
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    '''
    使用指定的device训练，完事后net、train_iter、test_iter留在device上。
    损失函数使用nn.CrossEntropyLoss()，即softmax运算+交叉熵损失函数。
    train_iter,test_iter:用法：“for X,y in ~”；
        X形状(样本数目,……（若干个维度）)，y形状(样本数目,1)。
    device：torch.device对象。想使用GPU加速，可指定为“
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ”
    optimizer：装载了net中所有参数的优化器；无需关心device。
    '''
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)

            # print('X: ',X.size())
            # print('y_hat: ',y_hat.size())

            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

class Residual(nn.Module): 
    '''ResNet'''

    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

class GlobalAvgPool2d(nn.Module): 
    '''
    全局平均池化层。
    该层的输入要求形状为(样本数目,通道数目,height,width)的Tensor，
    输出是形状为(样本数目,通道数目,1,1)的Tensor。
    '''
    # 通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        # nn.AvgPool2d层的输出的高度或宽度
        #     =下取整(
        #         (输入的高度或宽度+2*padding[·]-kernel_size[·])/stride[·] + 1
        #     )。
        # x.size()是(样本数目,通道数目,height,width)，
        # x.size()[2:]是(height,width)。
        return F.avg_pool2d(x, kernel_size=x.size()[2:])

def load_data_jay_lyrics():
    '''
    加载数据集：周杰伦的歌词“./Datasets/jaychou_lyrics/jaychou_lyrics.txt.zip”。
    返回tuple:(
        按顺序将各字符（包括空格）替换为对应的索引号后的list ,
        键为字符、值为索引号的dict ,
        按索引号顺序排列的各字符的list ,
        实际用到的字符数，即dict的键的个数
    )
    '''
    # with zipfile.ZipFile('../../data/jaychou_lyrics.txt.zip') as zin:
    with zipfile.ZipFile('./Datasets/jaychou_lyrics/jaychou_lyrics.txt.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    # corpus_chars[:40] #显示前40个字符
    # 这个数据集有6万多个字符。为了打印方便，我们把换行符替换成空格，
    # 然后仅使用前1万个字符来训练模型。
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    corpus_chars = corpus_chars[0:10000]
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    vocab_size # 1027
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    # 查看前20个字符的索引
    # sample = corpus_indices[:20]
    # print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
    # print('indices:', sample)
    return corpus_indices,char_to_idx,idx_to_char,vocab_size

def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    '''
    每个样本是原始序列上任意截取的一段序列。
    相邻的两个随机小批量在原始序列上的位置不一定相毗邻。
    无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。
    在训练模型时，每次随机采样前都需要重新初始化隐藏状态。
    返回tuple:(
        形状为(batch_size,……（若干个维度）)的输入张量,
        形状为(batch_size,1)的标签向量
    )。
    对于数据集“周杰伦歌词”，“输入张量”的形状是(batch_size,num_steps)的Tensor，
    即每个样本是一条长为num_steps的向量。
    '''

    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)

def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    '''
    相邻的两个随机小批量在原始序列上的位置相毗邻。
    只需在每一个迭代周期开始时初始化隐藏状态。
    同一迭代周期中，随着迭代次数的增加，梯度的计算开销会越来越大。
    为了使模型参数的梯度计算只依赖一次迭代读取的小批量序列，
    需在每次读取小批量前将隐藏状态从计算图中分离出来。
    返回tuple:(
        形状为(batch_size,……（若干个维度）)的输入张量,
        形状为(batch_size,1)的标签向量
    )。
    对于数据集“周杰伦歌词”，“输入张量”的形状是(batch_size,num_steps)的Tensor，
    即每个样本是一条长为num_steps的向量。
    '''

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

def one_hot(x, n_class, dtype=torch.float32): 
    '''
    X shape: (batch) .
    output:(batch, n_class)的Tensor；每一行中，只有x中对应那行的数所指的那一列为1。
    '''
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(dim=1,index=x.view(-1, 1),src=1)#在第1维（列的维度）上“听从”index，
        #把res中被index指出的元素（即某一列）替换为src。
    return res

def to_onehot(X, n_class):  
    '''
    X shape: (batch, seq_len).
    output: A list, containing seq_len elements of (batch, n_class).
        list中i号元素即X（整个批量）的序列的i号的one_hot向量（们）。
        整个list让torch.stack()处理一下恰好可以作为nn.RNN层的输入之一input。
    '''
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]

def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, device, idx_to_char, char_to_idx):
    '''
    基于前缀prefix（含有数个字符的字符串）来预测接下来的num_chars个字符。
    rnn(inputs, state, params) 循环神经网络，函数对象，
        inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    params 隐藏层、输出层等层的全部参数
    init_rnn_state(batch_size, num_hiddens, device)
        返回一个tuple，元素是形状为(批量大小, 隐藏单元个数)、值为0的Tensor。
        使用元组是为了更便于处理隐藏状态含有多个Tensor的情况。
    '''
    state = init_rnn_state(1, num_hiddens, device)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # 将上一时间步的输出作为当前时间步的输入
        X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
        # 计算输出和更新隐藏状态
        (Y, state) = rnn(X, state, params)
        # 下一个时间步的输入是prefix里的字符或者当前的最佳预测字符
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

def grad_clipping(params, theta, device): 
    '''
    裁剪梯度。把所有参数梯度的元素拼接成一个向量g，
    裁剪后的梯度=min(θ/g的L2范数,1)*g，裁剪后的梯度的L2范数小于等于θ。
    θ:裁剪阈值。
    '''
    norm = torch.tensor([0.0], device=device)
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)

def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                        vocab_size, device, corpus_indices, idx_to_char,
                        char_to_idx, is_random_iter, num_epochs, num_steps,
                        lr, clipping_theta, batch_size, pred_period,
                        pred_len, prefixes):
    '''
    这里的模型训练函数有以下特点：
        使用困惑度评价模型。
        在迭代模型参数前裁剪梯度。
        对时序数据采用不同采样方法将导致隐藏状态初始化的不同。相关讨论可参考6.3节（语言模型数据集（周杰伦专辑歌词））。
    rnn(inputs, state, params) 循环神经网络，函数对象，
        inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
    get_params() 返回隐藏层、输出层等层的全部参数
    init_rnn_state(batch_size, num_hiddens, device)
        返回一个tuple，元素是形状为(批量大小, 隐藏单元个数)、值为0的Tensor。
        使用元组是为了更便于处理隐藏状态含有多个Tensor的情况。
    corpus_indices 按顺序将各字符（包括空格）替换为对应的索引号后的list
    is_random_iter 要不要随机截取序列
    pred_period 每逢多少个epoch才用当前的模型预测一次
    pred_len 一次预测多少个字符
    prefixes list，预测用的前缀们
    '''
    if is_random_iter:
        # data_iter_fn = d2l.data_iter_random
        data_iter_fn = data_iter_random
    else:
        # data_iter_fn = d2l.data_iter_consecutive
        data_iter_fn = data_iter_consecutive
    params = get_params()
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # 如使用相邻采样，在epoch开始时初始化隐藏状态
            state = init_rnn_state(batch_size, num_hiddens, device)
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
        for X, Y in data_iter:
            if is_random_iter:  # 如使用随机采样，在每个小批量更新前初始化隐藏状态
                state = init_rnn_state(batch_size, num_hiddens, device)
            else:  
            # 否则需要使用detach函数从计算图分离隐藏状态, 这是为了
            # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                for s in state:
                    s.detach_()

            inputs = to_onehot(X, vocab_size)
            # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
            (outputs, state) = rnn(inputs, state, params)
            # 拼接之后形状为(num_steps * batch_size, vocab_size)
            outputs = torch.cat(outputs, dim=0)
            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            # 使用交叉熵损失计算平均分类误差
            l = loss(outputs, y.long())

            # 梯度清0
            if params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            grad_clipping(params, clipping_theta, device)  # 裁剪梯度
            # d2l.sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            sgd(params, lr, 1)  # 因为误差已经取过均值，梯度不用再做平均
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, device, idx_to_char, char_to_idx))

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) 
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = to_onehot(inputs, self.vocab_size) # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char,
                      char_to_idx):
    '''
    基于前缀prefix（含有数个字符的字符串）预测接下来的num_chars个字符。
    model:循环神经网络。具体的输入输出参考 RNN.RNNModel 。
    vocab_size, idx_to_char, char_to_idx:参见load_data_jay_lyrics()。
    返回长为 num_chars+len(prefix) 的字符串。
    '''
    state = None
    output = [char_to_idx[prefix[0]]] # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple): # LSTM, state:(h, c)  
                state = (state[0].to(device), state[1].to(device))
            else:   
                state = state.to(device)

        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])

def train_and_predict_rnn_pytorch(model, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    '''
    model:循环神经网络。具体的输入输出参考 RNN.RNNModel 。
    num_hiddens:在函数体内没有使用该参数，因此删去。
    vocab_size,corpus_indices,idx_to_char,char_to_idx:参见load_data_jay_lyrics()。
    num_steps:取数据时的步长，序列长度。
    clipping_theta:梯度裁剪的阈值。
    pred_period:每逢多少个epoch才用当前的模型预测一次。
    pred_len:预测多少个字符。
    prefixes:list，预测用的前缀们。
    '''
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance (state, tuple): # LSTM, state:(h, c)  
                    state = (state[0].detach(), state[1].detach())
                else:   
                    state = state.detach()

            (output, state) = model(X, state) # output: 形状为(num_steps * batch_size, vocab_size)

            # Y的形状是(batch_size, num_steps)，转置后再变成长度为
            # batch * num_steps 的向量，这样跟输出的行一一对应
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())

            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            grad_clipping(model.parameters(), clipping_theta, device)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))

def train_2d(trainer):  
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1和s2是自变量状态，第7章后续几节会使用
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results

def show_trace_2d(f, results): 
    # d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    # d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    # d2l.plt.xlabel('x1')
    # d2l.plt.ylabel('x2')
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')

def get_data_ch7():  
    '''
    未下载整个GitHub项目，该函数不可用。
    '''
    data = np.genfromtxt('../../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
    torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

def train_pytorch_ch7(optimizer_fn, optimizer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    '''
    本函数与原书不同的是这里第一个参数优化器函数而不是优化器的名字。
    例如: optimizer_fn=torch.optim.SGD, optimizer_hyperparams={"lr": 0.05}
    '''
    # 初始化模型
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1)
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparams)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]
    data_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels), batch_size, shuffle=True)

    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            # 除以2是为了和train_ch7保持一致, 因为squared_loss中除了2
            l = loss(net(X).view(-1), y) / 2 

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # 打印结果和作图
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    # d2l.set_figsize()
    # d2l.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    # d2l.plt.xlabel('epoch')
    # d2l.plt.ylabel('loss')
    # set_figsize() #无法使用
    plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    plt.xlabel('epoch')
    plt.ylabel('loss')

class Benchmark():  
    '''
    简单的计时器。
    使用方式：“
        with Benchmark('略略略'):
            …… #需要计时的事情
        ”
    '''
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        #与此对象相关的运行时上下文，
        #个人理解：程序计数器进入该对象代码段时，“插队”率先执行的代码，
        #with 语句将会绑定这个方法的返回值到 as 子句中指定的目标，如果有的话。
        self.start = time.time()

    def __exit__(self, *args):
        #退出关联到此对象的运行时上下文。对应 __enter__().
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))

def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    '''
    使用指定的device训练，完事后net、train_iter、test_iter留在device上。
    train_iter,test_iter:用法：“for X,y in ~”；
        X形状(样本数目,……（若干个维度）)，y形状(样本数目,1)。
    loss:损失函数对象。
    optimizer:优化器。
    num_epochs:训练的周期数。
    '''
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

def bbox_to_rect(bbox, color):  
    '''
    将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式:((左上x, 左上y), 宽, 高)。
    color:边框颜色。
    '''
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
    按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores. 
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores. 
        为了降低复杂度, sizes, ratios 中的元素并没有两两组合，见笔记。
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1.
        num_anchors = W*H * (n+m-1), n和m见笔记。
    """
    pairs = [] # pair of (size, sqrt(ration))
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])

    pairs = np.array(pairs)

    ss1 = pairs[:, 0] * pairs[:, 1] # size * sqrt(ration)
    ss2 = pairs[:, 0] / pairs[:, 1] # size / sqrt(ration)

    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2

    h, w = feature_map.shape[-2:]
    shifts_x = np.arange(0, w) / w
    shifts_y = np.arange(0, h) / h
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)

    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))

    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)
