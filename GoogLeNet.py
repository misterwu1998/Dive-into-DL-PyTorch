import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l
import myUtil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Inception(nn.Module):
    '''
    Inception块，大致结构是4条并行线路。具体结构见“./Inception块.svg”。
    块的输入要求形状为(样本数目,in_c,H,W)的Tensor，其中H和W≥1，
    输出是形状为(样本数目,c1+c2[1]+c3[1]+c4,H,W)的Tensor。
    in_c：输入Tensor的通道数。
    c1,c4：线路1、线路4的输出通道数。
    c2,c3：tuple，含2整数，对应线路2、线路3上的2层的输出通道数（这两条线路各有2层需要指定输出通道数）。
    '''
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
            # →(样本数目,c1,H,W)，H和W≥1

        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
            # →(样本数目,c2[0],H,W)，H和W≥1
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
            # →(样本数目,c2[1],H,W)，H和W≥1
        
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
            # →(样本数目,c3[0],H,W)，H和W≥1
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
            # →(样本数目,c3[1],H,W)，H和W≥1
        
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            # →(样本数目,in_c,H,W)，H和W≥1
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)
            # →(样本数目,c4,H,W)，H和W≥1

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
            # →(样本数目,c1,H,W)
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
            # →(样本数目,c2[1],H,W)
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
            # →(样本数目,c3[1],H,W)
        p4 = F.relu(self.p4_2(self.p4_1(x)))
            # →(样本数目,c4,H,W)
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出
            # →(样本数目,c1+c2[1]+c3[1]+c4,H,W)

def googLeNet(num_out=10):
    '''
    构造一个GoogLeNet。
    该模型要求输入是形状为(样本数目,1,H,W)的Tensor，H和W≥1，
    其输出是形状为(样本数目,num_out)的Tensor。
    '''
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            # →(样本数目,64,(H+1)>>1,(W+1)>>1)，H和W≥1
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            # →(样本数目,64,(H+3)>>2,(W+3)>>2)，H和W≥1
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
            # →(样本数目,64,(H+3)>>2,(W+3)>>2)，H和W≥1
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
            # →(样本数目,192,(H+3)>>2,(W+3)>>2)，H和W≥1
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            # →(样本数目,192,(H+7)>>3,(W+7)>>3)，H和W≥1
    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
            # →(样本数目,256,(H+7)>>3,(W+7)>>3)，H和W≥1
        Inception(256, 128, (128, 192), (32, 96), 64),
            # →(样本数目,480,(H+7)>>3,(W+7)>>3)，H和W≥1
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            # →(样本数目,480,(H+15)>>4,(W+15)>>4)，H和W≥1
    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
            # →(样本数目,512,(H+15)>>4,(W+15)>>4)，H和W≥1
        Inception(512, 160, (112, 224), (24, 64), 64),
            # →(样本数目,512,(H+15)>>4,(W+15)>>4)，H和W≥1
        Inception(512, 128, (128, 256), (24, 64), 64),
            # →(样本数目,512,(H+15)>>4,(W+15)>>4)，H和W≥1
        Inception(512, 112, (144, 288), (32, 64), 64),
            # →(样本数目,528,(H+15)>>4,(W+15)>>4)，H和W≥1
        Inception(528, 256, (160, 320), (32, 128), 128),
            # →(样本数目,832,(H+15)>>4,(W+15)>>4)，H和W≥1
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            # →(样本数目,832,(H+31)>>5,(W+31)>>5)，H和W≥1
    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
            # →(样本数目,832,(H+31)>>5,(H+31)>>5)，H和W≥1
        Inception(832, 384, (192, 384), (48, 128), 128),
            # →(样本数目,1024,(H+31)>>5,(H+31)>>5)，H和W≥1
        d2l.GlobalAvgPool2d())
            # →(样本数目,1024,1,1)
    net = nn.Sequential(
        b1, b2, b3, b4, b5, 
            # →(样本数目,1024,1,1)
        d2l.FlattenLayer(), 
            # →(样本数目,1024)
        nn.Linear(1024, num_out))
            # →(样本数目,num_out)
    return net

train=d2l.train_ch5

# def train(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
#     '''
#     使用指定的device训练，完事后net、train_iter、test_iter留在device上。
#     损失函数使用nn.CrossEntropyLoss()，即softmax运算+交叉熵损失函数。
#     train_iter、test_iter：训练数据集、测试数据集的迭代器，
#         每次返回(批量大小,通道数目,高度,宽度)的Tensor。
#     device：torch.device对象。想使用GPU加速，可指定为“
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         ”
#     optimizer：装载了net中所有参数的优化器；无需关心device。
#     '''
#     net = net.to(device)
#     print("training on ", device)
#     loss = torch.nn.CrossEntropyLoss()
#     for epoch in range(num_epochs):
#         train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
#         for X, y in train_iter:
#             X = X.to(device)
#             y = y.to(device)
#             y_hat = net(X)
#             # print('X: ',X.size())
#             # print('y_hat: ',y_hat.size())
#             l = loss(y_hat, y)
#             optimizer.zero_grad()
#             l.backward()
#             optimizer.step()
#             train_l_sum += l.cpu().item()
#             train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
#             n += y.shape[0]
#             batch_count += 1
#         test_acc = evaluate_accuracy(test_iter, net)
#         print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
#             % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

if __name__ == "__main__":
    net=googLeNet()
    batch_size = 128
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = myUtil.load_data_fashion_mnist(batch_size, resize=96)

    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
