import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    # def forward(self, padded_input):
    #     total_length = padded_input.size(1) # get the max sequence length
    #     # input_lengths = torch.LongTensor(padded_input.size(0))
    #     print('input_size:',padded_input.size(0),padded_input.size(1),padded_input.size(2))
    #     # 得到该batch中每一个sample的序列长度
    #     # input_lengths = torch.LongTensor([torch.max(padded_input[i, :].data.nonzero()) + 1 for i in range(padded_input.size(0))])
    #     input_lengths = []
    #     for i in range(padded_input.size(0)):
    #         input_lengths.append([torch.max(padded_input[i][j, :].data.nonzero()) + 1 for j in range(padded_input.size(1))])
    #     input_lengths = torch.LongTensor(input_lengths)
    #     print('input_lengths_size:',input_lengths.size())
    #     print('input_lengths:',input_lengths)
    #     # input_lengths = torch.LongTensor([torch.max(padded_input[i, :].data.nonzero()) + 1 )])
    #     # input_lengths, perm_idx = input_lengths.sort(1, descending=True)
    #     # input_seqs = padded_input[perm_idx][:, :input_lengths.max()]
    #     # print(input_lengths)
    #     # print(perm_idx)
    #     packed_input = pack_padded_sequence(padded_input, input_lengths,
    #                                         batch_first=True)
    #     print(packed_input.data.size())
    #     packed_output, _ = self.rnn(packed_input.data)
    #     output, _ = pad_packed_sequence(packed_output, batch_first=True,
    #                                     total_length=padded_input.size(1))
        
    #     return output

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input, is_mixed=False):
        # conv features
        #print('---forward propagation---')
        conv = self.cnn(input)
        # print('conv',conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        # conv = conv.permute(0, 2, 1)  # [b, w, c]
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(self.rnn(conv), dim=2)
        if is_mixed:
            output = output.permute(1, 0, 2)  # [w, b, c]
        return output
