import time
import torch
from torch import nn, optim
import torchvision
import d2lzh_pytorch as d2l
import myUtil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    '''
    模型的输入：(样本数目,1（通道数）,195,195)的Tensor
    模型的输出：(样本数目,num_out)的Tensor
    '''
    def __init__(self,num_out=10):
        super(AlexNet, self).__init__()
        # 输出的高度或宽度=下取整(
        #     1+(输入的高度或宽度-1+2*padding[·]-dilation[·]*(kernel_size[·]-1))/stride[·]
        #     )
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # MaxPool2d(kernel_size, stride)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), 
                # →(样本数目,96,(高-7)>>2,(宽-7)>>2)
            nn.ReLU(),
            nn.MaxPool2d(3, 2), 
                # →(样本数目,96,(高-11)>>3,(宽-11)>>3)
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
                # →(样本数目,256,(高-11)>>3,(宽-11)>>3)
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
                # →(样本数目,256,(高-19)>>4,(宽-19)>>4)
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
                # →(样本数目,384,(高-19)>>4,(宽-19)>>4)
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
                # →(样本数目,384,(高-19)>>4,(宽-19)>>4)
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
                # →(样本数目,256,(高-19)>>4,(宽-19)>>4)
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
                # →(样本数目,256,(高-35)>>5,(宽-35)>>5)
        )
        # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        # forward()将此处的输入整形（摊平每个样本）为(样本数目,256*((高-35)>>5)*((宽-35)>>5))，
        # 根据全连接块首个线性层的in_features可令((高-35)>>5)==((宽-35)>>5)==5，
        # 得：高==宽==195。
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
                # →(样本数目,4096)
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
                # →(样本数目,4096)
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, num_out),
                # →(样本数目,num_out)
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
    def train(self, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
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
    net=AlexNet()
    train_iter, test_iter=myUtil.load_data_fashion_mnist(batch_size=1,resize=(195,195))
    for X,y in train_iter:
        y_hat=net(X)
        print('X: ',X.size())
        print('y_hat: ',y_hat.size())
        break
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, 32, optimizer, device, num_epochs)
    print('--end--')