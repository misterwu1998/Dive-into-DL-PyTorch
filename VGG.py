import time
import torch
from torch import nn, optim
import d2lzh_pytorch as d2l
import myUtil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs, in_channels, out_channels):
    '''
    构造一个VGG块。
    VGG块的输入要求形状为(样本数目,in_channels,高度,宽度)的Tensor，其中高度、宽度≥2。
    输出是形状为(样本数目,out_channels,高度>>1,宽度>>1)的Tensor。
    '''
    blk = []
    for i in range(num_convs):
        # 输出的高度或宽度
        #     =下取整(
        #     1+(输入的高度或宽度-1+2*padding[·]-dilation[·]*(kernel_size[·]-1))/stride[·]
        #     )。
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                # →(样本数目,out_channels,高度,宽度)
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
                # →(样本数目,out_channels,高度,宽度)
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        # →(样本数目,out_channels,高度>>1,宽度>>1)
    return nn.Sequential(*blk)

def vgg11(conv_arch, fc_features, fc_hidden_units=4096, num_out=10):
    '''
    构造一个VGG-11模型。
    模型的输入要求形状为(样本数目,首个in_channels（即conv_arch[0][1]）,输入的高度,输入的宽度)，
    其中输入的高度、输入的宽度≥32；输出的形状为(样本数目,num_out)
    conv_arch：一个list，含5个list对应5个VGG块，每个list含3元素，对应vgg_block()的3个参数。
    fc_features：最后一个out_channels（即conv_arch[4][2]）*(输入的高度>>5)*(输入的宽度>>5)。
    '''
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        # 每经过一个vgg_block都会使宽高减半
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    # 经过5个VGG块，形状→(样本数目,最后一个out_channels,输入的高度>>5,输入的宽度>>5)，H和W≥32
    # 全连接层部分
    net.add_module(
        "fc", 
        nn.Sequential(
            d2l.FlattenLayer(),
                # →(样本数目,最后一个out_channels*(输入的高度>>5)*(输入的宽度>>5))
            nn.Linear(fc_features, fc_hidden_units),
                # →(样本数目,fc_hidden_units)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, fc_hidden_units),
                # →(样本数目,fc_hidden_units)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_units, num_out)
                # →(样本数目,num_out)
                                ))
    return net

def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    '''
    使用指定的device训练，完事后net、train_iter、test_iter留在device上。
    损失函数使用nn.CrossEntropyLoss()，即softmax运算+交叉熵损失函数。
    train_iter、test_iter：训练数据集、测试数据集的迭代器，
        每次返回(批量大小,通道数目,高度,宽度)的Tensor。
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

if __name__ == "__main__":
    conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
    # 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7

    X = torch.rand(1, 1, 32,32)
    fc_features = 512 * 1*1    # c * (输入的高度>>5)*(输入的宽度>>5)

    fc_hidden_units = 4096 # 任意
    net = vgg11(conv_arch, fc_features, fc_hidden_units)

    # named_children获取一级子模块及其名字(named_modules会返回所有子模块,包括子模块的子模块)
    for name, blk in net.named_children(): 
        X = blk(X)
        print(name, 'output shape: ', X.shape)

    # 获取数据和训练模型
    ratio = 8
    small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), 
                    (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]
    net = vgg11(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
    print(net)
    batch_size = 64
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = myUtil.load_data_fashion_mnist(batch_size, resize=(224,224))

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

