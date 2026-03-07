import torch
import torch.nn as nn

    
class Siamese(nn.Module):
    def __init__(self, model_name, method='method', flat_shape = 64):
        super(Siamese, self).__init__()
        self.model_name = model_name
        self.method = method
        if model_name == 'GRU':
            from nets.GRU import TinyGRU
            self.backbone = TinyGRU()
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'GRUseq':
            from nets.GRU import TinyGRU
            self.backbone = TinyGRU(input_size=1,hidden_size=32)
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'GRUhilbert':
            from nets.GRU import TinyGRU
            self.backbone = TinyGRU(input_size=3,hidden_size=64)
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'LSTM':
            from nets.LSTM import TinyLSTM
            self.backbone = TinyLSTM()
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'LSTM':
            from nets.LSTM import TinyLSTM
            self.backbone = TinyLSTM()
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'LSTMseq':
            from nets.LSTM import TinyLSTM
            self.backbone = TinyLSTM(input_size=1,hidden_size=32)
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'LSTMhilbert':
            from nets.LSTM import TinyLSTM
            self.backbone = TinyLSTM(input_size=3,hidden_size=64)
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'Conv1D':
            from nets.Conv1D import TinyConv1D
            self.backbone = TinyConv1D()
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'Conv1Dhilbert':
            from nets.Conv1D import TinyConv1D
            self.backbone = TinyConv1D(input_size=3)
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'Conv1Dseq':
            from nets.Conv1D import TinyConv1D
            self.backbone = TinyConv1D(input_size=1,output_dim=32)
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'TCN':
            from nets.TCN import TinyTCN
            self.backbone = TinyTCN()
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'TCNhilbert':
            from nets.TCN import TinyTCN
            self.backbone = TinyTCN(input_size=3)
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
        elif model_name == 'TCNseq':
            from nets.TCN import TinyTCN
            self.backbone = TinyTCN(input_size=1,output_dim=32)
            self.fully_connect = torch.nn.Linear(flat_shape, 1)
    def forward(self, x):
        x1, x2 = x
        #------------------------------------------#
        #   我们将两个输入传入到主干特征提取网络
        #------------------------------------------#
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        #-------------------------#
        #   相减取绝对值，取l1距离
        #-------------------------#     
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        #-------------------------#
        #   进行两次全连接
        #-------------------------#
        if self.model_name == 'Vit':
            x = self.fully_connect1(x)
            x = self.fully_connect2(x)
        else:
            x = self.fully_connect(x)
        return x
