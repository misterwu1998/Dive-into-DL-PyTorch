import time
import torch
from torch import nn, optim
import d2lzh_pytorch as d2l
import torch.nn.functional as F
import myUtil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    '''
    构造一个NiN块。
    内含2层1×1卷积，相当于输入的各通道到输出的各通道的全连接。
    块的输入要求形状为(样本数量,in_channels,height,width)，
    输出的形状为(样本数目,
                out_channels,
                下取整(1+(height-1+2*padding[0]-(kernel_size[0]-1))/stride[0]),
                下取整(1+(width-1+2*padding[1]-(kernel_size[1]-1))/stride[1]))。
    '''
    # nn.MaxPool2d层,nn.Conv2d层的输出的高度或宽度
    #     =下取整(
    #     1+(输入的高度或宽度-1+2*padding[·]-dilation[·]*(kernel_size[·]-1))/stride[·]
    #     )
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            # →(样本数目,
            #   out_channels,
            #   下取整(1+(height-1+2*padding[0]-(kernel_size[0]-1))/stride[0]),
            #   下取整(1+(height-1+2*padding[1]-(kernel_size[1]-1))/stride[1]))
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
            # →(样本数目,
            #   out_channels,
            #   下取整(1+(height-1+2*padding[0]-(kernel_size[0]-1))/stride[0]),
            #   下取整(1+(height-1+2*padding[1]-(kernel_size[1]-1))/stride[1]))
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
            # →(样本数目,
            #   out_channels,
            #   下取整(1+(height-1+2*padding[0]-(kernel_size[0]-1))/stride[0]),
            #   下取整(1+(height-1+2*padding[1]-(kernel_size[1]-1))/stride[1]))
        nn.ReLU())
    return blk

def nin(num_out=10):
    '''
    构造一个NiN模型。
    模型的输入要求形状为(样本数目,1（通道数目）,height（不小于67）,width（不小于67）)的Tensor，
    输出是形状为(样本数目,num_out)的Tensor。
    num_out：样本标签类别数目。
    '''
    return nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=0),
            # →(样本数目,
            #   96,
            #   (height-7)>>2,
            #   (width-7)>>2)
        nn.MaxPool2d(kernel_size=3, stride=2),
            # →(样本数目,
            #   96,
            #   (height-11)>>3,
            #   (width-11)>>3)
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
            # →(样本数目,
            #   256,
            #   (height-11)>>3,
            #   (width-11)>>3)
        nn.MaxPool2d(kernel_size=3, stride=2),
            # →(样本数目,
            #   96,
            #   (height-19)>>4,
            #   (width-19)>>4)
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
            # →(样本数目,
            #   384,
            #   (height-19)>>4,
            #   (width-19)>>4)
        nn.MaxPool2d(kernel_size=3, stride=2), 
            # →(样本数目,
            #   96,
            #   (height-35)>>5,
            #   (width-35)>>5)
        nn.Dropout(0.5),
        # 标签类别数是num_out
        nin_block(384, num_out, kernel_size=3, stride=1, padding=1),
            # →(样本数目,
            #   num_out,
            #   (height-35)>>5,
            #   (width-35)>>5)
        d2l.GlobalAvgPool2d(),
            # →(样本数目,num_out,1,1) 
        # 将四维的输出转成二维的输出，其形状为(批量大小, num_out)
        d2l.FlattenLayer()
            # →(样本数目,num_out)
        )

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
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

if __name__ == "__main__":
    net=nin()
    X = torch.rand(1, 1, 67, 67)
    for name, blk in net.named_children(): 
        X = blk(X)
        print(name, 'output shape: ', X.shape)
    batch_size = 128
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = myUtil.load_data_fashion_mnist(batch_size, resize=224)

    lr, num_epochs = 0.002, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
