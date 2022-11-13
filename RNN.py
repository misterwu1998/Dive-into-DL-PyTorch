import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l
import myUtil
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNModel(nn.Module):
    '''
    简单RNN模型。带有输出层（全连接层，不带有交叉熵损失函数）。
    模型输入：
        input:(样本数目,序列长度)的Tensor。
        state:初始状态，(rnn的层数 * rnn的方向数目, 样本数目, hidden_size)的Tensor。
    输出tuple:(
        output:
            (序列长度 * 样本数目, vocab_size)的Tensor（
            在第0维上，同一样本的整个序列是相隔(样本数目)的一组，
            也就是说，处于序列上同一位置的被放在一起）,
        h_n:(self.rnn.num_layers*self.rnn.num_directions,
             样本数目,
             self.rnn.hidden_size
            )的Tensor
    rnn_layer:待封装的循环神经网络层。
    )
    '''
    def __init__(self, vocab_size, rnn_layer):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1) 
            # 替代 rnn_layer.num_directions*rnn_layer.hidden_size ，
            # 即屏蔽掉self.rnn是否双向这一信息。
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        X = d2l.to_onehot(inputs, self.vocab_size) # X是个list
        Y, self.state = self.rnn(torch.stack(X), state)
        # 此时Y形状(序列长度, 样本数目, self.hidden_size)。
        # nn.RNN层的前向计算不涉及输出层，因此需要在这里自己加上全连接层。
        # 全连接层会首先将Y的形状变成(序列长度 * 样本数目, num_hiddens)，
        # 它的输出形状为(序列长度 * 样本数目, vocab_size)。
        # 注意：Tensor对象.view()不会改变任意一对元素之间的相对顺序，
        # 因此Y被view()之后，在第0维上，同一样本的整个序列是相隔(样本数目)的一组，
        # 也就是说，处于序列上同一位置的被放在一起。
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

predict=d2l.predict_rnn_pytorch

def train_and_predict(model, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes, 
                                is_random_iter):
    '''
    model:循环神经网络。具体的输入输出参考 RNN.RNNModel 。
    num_hiddens:在函数体内没有使用该参数，因此删去。
    vocab_size,corpus_indices,idx_to_char,char_to_idx:参见load_data_jay_lyrics()。
    num_steps:取数据时的步长，序列长度。
    clipping_theta:梯度裁剪的阈值。
    pred_period:每逢多少个epoch才用当前的模型预测一次。
    pred_len:预测多少个字符。
    prefixes:list，预测用的前缀们。
    is_random_iter:要不要随机截取序列。
    '''
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    data=None
    if is_random_iter:#要使用随机数据迭代器
        data=d2l.data_iter_random
    else:#不使用随机的，使用相邻的
        data=d2l.data_iter_consecutive
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data(corpus_indices, batch_size, num_steps, device) # 相邻采样
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
            d2l.grad_clipping(model.parameters(), clipping_theta, device)
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
                print(' -', d2l.predict_rnn_pytorch(
                    prefix, pred_len, model, vocab_size, device, idx_to_char,
                    char_to_idx))

if __name__ == "__main__":
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
    num_hiddens=256
    rnn_layer=nn.RNN(input_size=vocab_size,hidden_size=num_hiddens)
    num_steps = 35
    batch_size = 2
    state = None
    X = torch.rand(num_steps, batch_size, vocab_size)
    Y, state_new = rnn_layer(X, state)
    print(Y.shape, len(state_new), state_new[0].shape)

    print('--end--')