import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l
import myUtil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Residual(nn.Module):
    '''
    残差块。大致分两条并行线路，“残差”线路的输出是g(X)=f(X)-X，“输入”线路的输出是X，
    块的输出是g(X)+X=f(X)。如果输入通道数不等于输出通道数，
    就需要在两个并行线路上离X最近的位置率先改变通道数。
    块要求输入是(样本数目,in_channels,H,W)的Tensor，H和W≥1；
    输出是(样本数目,out_channels,下取整((H-1)/stride + 1),下取整((W-1)/stride + 1))的Tensor。
    use_1x1conv：是否使用1×1卷积层；只要in_channels≠out_channels，就需要使用，
        否则无法将“输入”线路的输出加到“残差”线路的输出上。
    stride：“残差”线路上的首个卷积层、“输入”线路上可能加入的1×1卷积层所使用的步长参数stride。
    '''
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
            # →(样本数目,
            #   out_channels,
            #   下取整((_-1)/stride + 1),
            #   下取整((_-1)/stride + 1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            # →(样本数目,
            #   out_channels,
            #   ……,
            #   ……)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
                # →(样本数目,
                #   out_channels,
                #   下取整((_-1)/stride + 1),
                #   下取整((_-1)/stride + 1))
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
            # →(样本数目,
            #   out_channels,
            #   下取整((H-1)/stride + 1),
            #   下取整((W-1)/stride + 1))
        Y = self.bn2(self.conv2(Y))
            # →(样本数目,
            #   out_channels,
            #   下取整((H-1)/stride + 1),
            #   下取整((W-1)/stride + 1))
        if self.conv3:
            X = self.conv3(X)
                # →(样本数目,
                #   out_channels,
                #   下取整((H-1)/stride + 1),
                #   下取整((W-1)/stride + 1))
        return F.relu(Y + X)
            # →(样本数目,
            #   out_channels,
            #   下取整((H-1)/stride + 1),
            #   下取整((W-1)/stride + 1))

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    '''
    ResNet模型使用的模块。
    模块要求输入是(样本数目,in_channels,H,W)的Tensor。
    num_residuals：这个模块要使用几个残差块。
    first_block：这个模块是不是ResNet网络中的首个模块；如果是，
        那么in_channels和out_channels必须相等。
    if first_block:
        blk 包含 num_residuals 个残差块，且每个残差块的输出与输入同形状（
        要求 in_channels==out_channels ），整个模块的输出是形状为
        (样本数目,out_channels,H,W)。
    else:
        blk 同样包含 num_residuals 个残差块，其中只有首个残差块将输出的形状改为
        (样本数目,out_channels,(H+1)>>1,(W+1)>>1)（亦整个模块输出的形状）。
    '''
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:#ResNet网络中的（从1计起）第2或第3或…个模块中的首个残差块
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
                # →(样本数目,out_channels,(_+1)>>1,(_+1)>>1)
        else:#ResNet网络中的首个模块中的残差块，或者其余模块中（从1计起）第2或第3或…个残差块。
            blk.append(Residual(out_channels, out_channels))
                # →(样本数目,out_channels,_,_)
    return nn.Sequential(*blk)

def resNet18(num_out=10):
    '''
    构造一个ResNet-18模型。
    模型要求输入是形状为(样本数目,1,H,W)的Tensor，H和W≥1。
    模型的输出是形状为(样本数目,num_out)的Tensor。
    '''
    net = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            # →(样本数目,64,(H+1)>>1,(W+1)>>1)
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            # →(样本数目,64,(H+3)>>2,(W+3)>>2)

    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
        # →(样本数目,64,(H+3)>>2,~)
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
        # →(样本数目,128,(H+7)>>3,~)
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
        # →(样本数目,256,(H+15)>>4,~)
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
        # →(样本数目,512,(H+31)>>5,~)

    net.add_module("global_avg_pool", d2l.GlobalAvgPool2d())
        # →(样本数目, 512, 1, 1)
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(512, num_out))) 
        # →(样本数目, num_out)

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
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
