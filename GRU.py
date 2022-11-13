import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import d2lzh_pytorch as d2l
import myUtil
import RNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gruModel(vocab_size,gruLayer:nn.GRU):
    '''
    封装一个 nn.GRU 层，返回带有输出层的模型。
    模型的输入输出参见 RNN.RNNModel 。
    '''
    return RNN.RNNModel(vocab_size=vocab_size,rnn_layer=gruLayer)

predict=RNN.predict

train_and_predict=RNN.train_and_predict

if __name__ == "__main__":
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
    
