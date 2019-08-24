import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils

# a = [[5,7,5,3,9],[5,7,5,3,9]]
# a_t = torch.LongTensor(a)
# print('a:',a)
# print('a_t:',a_t)
# a_resort,another = a_t.sort(1,descending=True)
# print('a_resort:',a_resort)
# print('another:',another)

# input_seqs = a_t[another,:]
# print('input_seqs:',input_seqs)

# # batch_sized的作用
# batch_size =3
# max_length = 3
# hidden_size = 2
# n_layers = 1

# tensor_in = torch.FloatTensor([[1,2,3],[1,0,0],[6,0,0]]).resize_(3,3,1)
# tensor_in = Variable(tensor_in)
# seq_lengths = [3,1,1]

# print(tensor_in)

# pack = nn_utils.rnn.pack_padded_sequence(tensor_in,seq_lengths,batch_first=True)
# print('packed:',pack)

# rnn = nn.RNN(1,hidden_size,n_layers,batch_first=True)
# h0 = Variable(torch.randn(n_layers,batch_size,hidden_size))

# out,_ = rnn(pack)
# print('out:',out)

# unpacked = nn_utils.rnn.pad_packed_sequence(out,batch_first=True)
# print('unpacked:',unpacked)
# region =torch.randn(41,5,512)
# one = region[0,:]
# print(one.size())
# print(one)



# import torch
# from torch import nn
# import numpy as np
# import matplotlib.pyplot as plt
 
# class Rnn(nn.Module):
#     def __init__(self, INPUT_SIZE):
#         super(Rnn, self).__init__()
 
#         self.rnn = nn.RNN(
#             input_size=INPUT_SIZE,
#             hidden_size=32,
#             num_layers=1,
#             batch_first=True
#         )
 
#         self.out = nn.Linear(32, 1)
 
#     def forward(self, x, h_state):
#         r_out, h_state = self.rnn(x, h_state)
 
#         outs = []
#         for time in range(r_out.size(1)):
#             outs.append(self.out(r_out[:, time, :]))
#         return torch.stack(outs, dim=1), h_state

 
# # 定义一些超参数
# TIME_STEP = 10
# INPUT_SIZE = 1
# LR = 0.02
 
# # # 创造一些数据
# # steps = np.linspace(0, np.pi*2, 100, dtype=np.float)
# # x_np = np.sin(steps)
# # y_np = np.cos(steps)
# # #
# # # “看”数据
# # plt.plot(steps, y_np, 'r-', label='target(cos)')
# # plt.plot(steps, x_np, 'b-', label='input(sin)')
# # plt.legend(loc='best')
# # plt.show()
 
# # 选择模型
# model = Rnn(INPUT_SIZE)
# print(model)
 
# # 定义优化器和损失函数
# loss_func = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 
# h_state = None # 第一次的时候，暂存为0
 
# for step in range(300):
#     start, end = step * np.pi, (step+1)*np.pi
 
#     steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
#     x_np = np.sin(steps)
#     y_np = np.cos(steps)
 
#     x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
#     y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
 
#     prediction, h_state = model(x, h_state)
#     print(h_state.data)
#     h_state = h_state.data
 
#     loss = loss_func(prediction, y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
# plt.plot(steps, y_np.flatten(), 'r-')
# plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
# plt.show()



import torch
import torch.nn as nn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LSTMTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 32)
        self.lstm = nn.LSTM(32, 5, batch_first=True)
        self.hidden2tag = nn.Linear(5, 10)
        self.logSoftmax = nn.LogSoftmax(dim=2)

    def init_hidden(self, batch_size=1):
        return (torch.empty(batch_size, 1, 5, device=DEVICE).normal_(),
                torch.empty(batch_size, 1, 5, device=DEVICE).normal_())

    def forward(self, sentence, sentence_lengths, hidden):
        sentence_lengths = sentence_lengths.type(torch.LongTensor)
        embeds = self.embedding(sentence.long())
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, 
				 sentence_lengths.to(torch.device('cpu')), batch_first=True)
        hidden0 = [x.permute(1,0,2).contiguous() for x in hidden]
        lstm_out, hidden0 = self.lstm(embeds, hidden0)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, 
					  batch_first=True, total_length=sentence.shape[1])
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.logSoftmax(tag_space)
        return tag_scores, tag_space

def train():
    try:
        print('number of GPUs available:{}'.format(torch.cuda.device_count()))
        print('device name:{}'.format(torch.cuda.get_device_name(0)))
    except:
        pass
    sentence = torch.rand(100, 8, device=DEVICE)
    sentence = torch.abs(sentence * (10)).int()
    sentence_lengths = [sentence.shape[1]] * len(sentence)

    model = LSTMTest()
    model.to(DEVICE)
    model = nn.DataParallel(model)
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(params, lr=0.01)
    batch_size = 6
    for epoch in range(100):
        print(epoch)
        pointer = 0
        while pointer + batch_size <= len(sentence):
            # print(epoch)
            x_batch = sentence[pointer:pointer+batch_size]
            x_length = torch.tensor(sentence_lengths[pointer:pointer+batch_size]).to(DEVICE)
            y = x_batch
            hidden = model.module.init_hidden(batch_size=batch_size)
            y_pred, tag_space = model(x_batch, x_length, hidden)
            loss = criterion(y_pred.view(-1,y_pred.shape[-1]), y.long().view(-1))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            pointer = pointer + batch_size

train()