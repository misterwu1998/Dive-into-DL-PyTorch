import time
import torch
from torch import nn, optim
import d2lzh_pytorch as d2l
import myUtil

class LeNet(nn.Module):
    '''
    返回LeNet：
        输入：(样本数目,1（通道数目）,28,28)的Tensor。
        输出：(样本数目,num_out)的Tensor。
    '''
    def __init__(self,num_out=10):
        super(LeNet, self).__init__()
        # 卷积层块
        # nn.MaxPool2d层,nn.Conv2d层的输出的高度或宽度
        #     =下取整(
        #     1+(输入的高度或宽度-1+2*padding[·]-dilation[·]*(kernel_size[·]-1))/stride[·]
        #     )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), 
                # →(样本数目,6,24,24)
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
                # →(样本数目,6,12,12)
            nn.Conv2d(6, 16, 5),
                # →(样本数目,16,8,8)
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
                # →(样本数目,16,4,4)
        )
        # 全连接层块
        # 此时forward()会将每个样本摊平。
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
                # →(样本数目,120)
            nn.Sigmoid(),
            nn.Linear(120, 84),
                # →(样本数目,84)
            nn.Sigmoid(),
            nn.Linear(84, num_out)
                # →(样本数目,num_out)
        )
    def forward(self, img):
        '''
        img：(样本数目,1（通道数目）,16,16)的Tensor。
        返回：(样本数目,num_out)的Tensor。
        '''
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    def train(self, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
        '''
        使用指定的device训练，完事后当前网络、train_iter、test_iter留在device上。
        损失函数使用nn.CrossEntropyLoss()，即softmax运算+交叉熵损失函数。
        device：torch.device对象。想使用GPU加速，可指定为“
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ”
        optimizer：装载了net中所有参数的优化器；无需关心device。
        '''
        d2l.train_ch5(self,train_iter,test_iter,batch_size,optimizer,device,num_epochs)

def leNet_BN(num_out=10):
    '''
    构造一个带有批量归一化的LeNet。
    返回LeNet：
        输入：(样本数目,1（通道数目）,28,28)的Tensor。
        输出：(样本数目,num_out)的Tensor。
    '''
    return nn.Sequential(
        nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
        nn.BatchNorm2d(6),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2), # kernel_size, stride
        nn.Conv2d(6, 16, 5),
        nn.BatchNorm2d(16),
        nn.Sigmoid(),
        nn.MaxPool2d(2, 2),
        d2l.FlattenLayer(),
        nn.Linear(16*4*4, 120),
        nn.BatchNorm1d(120),
        nn.Sigmoid(),
        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.Sigmoid(),
        nn.Linear(84, num_out))

def train_BN(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    '''
    使用指定的device训练，完事后net、train_iter、test_iter留在device上。
    损失函数使用nn.CrossEntropyLoss()，即softmax运算+交叉熵损失函数。
    net：带有批量归一化的LeNet。
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
    net=LeNet()
    # 数据集
    batch_size=256
    train_iter,test_iter=myUtil.load_data_fashion_mnist(batch_size)

    # for X,y in train_iter:
    #     print(X.size()) # “torch.Size([256, 1, 28, 28])”
    #     break

    # 训练
    # net.train(train_iter,
    #         test_iter,
    #         batch_size,
    #         optim.Adam(net.parameters(),0.001),
    #         torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    #         5)
    # myUtil.saveModel(net,'LeNet202008251233')