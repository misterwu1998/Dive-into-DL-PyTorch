import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l
import myUtil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_block(in_channels, out_channels):
    '''
    构造一个conv_block卷积块。
    该块要求输入是形状为(样本数目,in_channels,H,W)的Tensor。
    输出是形状为(样本数目,out_channels,H,W)的Tensor。
    '''
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels), 
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            # →(样本数目,out_channels,……,……)
    return blk

class DenseBlock(nn.Module):
    '''
    稠密块；要求输入是(样本数目,in_channels,H,W)的Tensor；
    输出是(样本数目,in_channels+num_convs*out_channels,H,W)的Tensor。
    num_convs:需要多少个conv_block。
    out_channels:使用的conv_block的输出通道数。
    '''
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
                # 随着i的递增，conv_block要求输入Tensor的通道数越来越大：
                # in_channels,in_channels+out_channels,in_channels+2*out_channels,
                # …,in_channels+(num_convs-1)*out_channels 。
            net.append(conv_block(in_c, out_channels))
                # →(样本数目,out_channels,……,……)。
        self.net = nn.ModuleList(net) #未连接各conv_block，在该类中的forward()中去连接。
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        # i=0
        for blk in self.net:
            # 此时X的形状为(样本数目,in_channels+i*out_channels,H,W)。
            Y = blk(X)
            # Y的形状为(样本数目,out_channels,H,W)。
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
            # i++
            # 此时X的形状为(样本数目,in_channels+i*out_channels,H,W)。
        return X
            # 此时 i==num_convs ，X形状为(样本数目,in_channels+num_convs*out_channels,H,W)。

def transition_block(in_channels, out_channels):
    '''
    构造一个过渡层，用来控制模型复杂度（每个稠密块都会带来通道数的增加）。
    该层要求输入是(样本数目,in_channels,H,W)的Tensor，其中H和W≥2；
    输出是(样本数目,out_channels,H>>1,W>>1)的Tensor。
    '''
    blk = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
                # →(样本数目,out_channels,_,_)
            nn.AvgPool2d(kernel_size=2, stride=2))
                # →(样本数目,out_channels,(_)>>1,(_)>>1)
    return blk

def denseNet(growth_rate,num_convs_in_dense_blocks,num_channels=64,num_out=10):
    '''
    构造一个DenseNet模型。
    模型要求输入是(样本数目,1,H,W)的Tensor，其中H和W≥(1<<(len(num_convs_in_dense_blocks)+1))-3。
    输出是(样本数目,num_out)的Tensor。
    num_channels:模型中首个二维卷积层的输出通道数。
    growth_rate:稠密块会增大通道数（公式参见DenseBlock类的注释），
        该参数决定DenseBlock构造器的out_channels参数。
    num_convs_in_dense_blocks:list，元素个数即期望给当前DenseNet模型加入稠密块的个数，
        其中元素对应于稠密块的num_convs参数。
    num_out:输出的分类类别数目。
    '''
    net = nn.Sequential(
        nn.Conv2d(1, num_channels, kernel_size=7, stride=2, padding=3),
            # →(样本数目,num_channels,(H+1)>>1,(W+1)>>1)
        nn.BatchNorm2d(num_channels), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            # →(样本数目,num_channels,(H+3)>>2,~)

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        DB = DenseBlock(num_convs=num_convs,
                        in_channels=num_channels,
                        out_channels=growth_rate)
            # →(样本数目,此时的num_channels+此时的num_convs*growth_rate,~,~)
        net.add_module("DenseBlosk_%d" % i, DB)
        # 刚才这个稠密块的输出通道数，即 此时的num_channels+此时的num_convs*growth_rate
        num_channels = DB.out_channels #num_channels更新
        # 在稠密块之间加入通道数减半的过渡层
        if i != len(num_convs_in_dense_blocks) - 1:#刚才这个稠密块不是最后一个
            net.add_module("transition_block_%d" % i, 
                           transition_block(num_channels, num_channels // 2))
                # →(样本数目,num_channels>>1,_>>1,_>>1)
            num_channels = num_channels // 2 #num_channels更新
    # 记len(num_convs_in_dense_blocks)为len。
    # 最后再加一个稠密块。
    # 第i（从0计起）对“稠密块+过渡层”的输出形状为(
    #     样本数目,
    #     (输入通道数[i]+num_convs_in_dense_blocks[i]*growth_rate)>>1,
    #     输入高度>>1,
    #     输入宽度>>1
    # )，其中，通道维度上的迭代关系为：
    # 输入通道数[i+1]==(输入通道数[i]+num_convs_in_dense_blocks[i]*growth_rate)>>1。
    # len-1对“稠密块+过渡层”的输出形状为(
    #     样本数目,
    #     (   输入通道数[0]+growth_rate*(
    #             for i in [0,1,…,len-2]:
    #                 ∑(num_convs_in_dense_blocks[i]<<i))
    #     )>>(len-1),
    #     首个稠密块的输入高度>>(len-1),
    #     首个稠密块的输入宽度>>(len-1)
    # )==(
    #     样本数目,
    #     (   最初的num_channels+growth_rate*(
    #             for i in [0,1,…,len-2]:
    #                 ∑(num_convs_in_dense_blocks[i]<<i))
    #     )>>(len-1),
    #     (H+3)>>(len+1),
    #     (W+3)>>(len+1)
    # )，最后再加一个稠密块，所得形状为(
    #     样本数目,
    #     (   最初的num_channels+growth_rate*(
    #             for i in [0,1,…,len-1]:
    #                 ∑(num_convs_in_dense_blocks[i]<<i))
    #     )>>(len-1),
    #     (H+3)>>(len+1),
    #     (W+3)>>(len+1)
    # )

    net.add_module("BN", nn.BatchNorm2d(num_channels))
    net.add_module("relu", nn.ReLU())
    net.add_module("global_avg_pool", d2l.GlobalAvgPool2d()) 
        # →(  样本数目,
        #     (   最初的num_channels+growth_rate*(
        #             for i in [0,1,…,len-1]:
        #                 ∑(num_convs_in_dense_blocks[i]<<i))
        #     )>>(len-1),
        #     1,
        #     1
        # )
    net.add_module("fc", nn.Sequential(d2l.FlattenLayer(), nn.Linear(num_channels, num_out))) 
        # →(样本数目,num_out)

    return net

train=d2l.train_ch5

if __name__ == "__main__":
    num_channels, growth_rate = 64, 32  # num_channels为当前的通道数
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    net=denseNet(growth_rate,num_convs_in_dense_blocks,num_channels)
    X = torch.rand((1, 1, 96, 96))
    for name, layer in net.named_children():
        X = layer(X)
        print(name, ' output shape:\t', X.shape)
