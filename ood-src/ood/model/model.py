import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
import torch.nn as nn

DEBUG_MODE=True

class MLP(Module):
    def __init__(self, in_ch, hid_ch, out_ch, return_rep: bool = False):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_ch, hid_ch)
        self.fc2 = torch.nn.Linear(hid_ch, out_ch)
        self.return_rep = return_rep

    def forward(self, x):
        x = F.relu(self.fc1(x))
        if self.return_rep:
            return x, self.fc2(x)
        return self.fc2(x)


class SetConvolution(Module):
    def __init__(self, in_ch, hid_ch, out_ch, num_layers=2, pool_type ='mean'):
        super(SetConvolution, self).__init__()
        self.pool_type = pool_type
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            hid_input_ch = in_ch if i == 0 else hid_ch
            hid_output_ch = out_ch if i == num_layers - 1 else hid_ch
            self.layers.append(nn.Linear(hid_input_ch, hid_output_ch))

    def forward(self, x):
        for layer in self.layers:
            # print("Before forward:", x.shape)
            # import pdb; pdb.set_trace()
            x = layer(x)
            x = F.relu(x)
        if self.pool_type == 'mean':
            # print("Before mean:", x.shape)
            # e.g., when x stands for table encoding,
            # x.shape = [ batchsize(64), table number(4), 64 ]
            x = torch.mean(x, dim= 1)
            # print("After mean:", x.shape)
        elif self.pool_type == 'min':
            x , _ = torch.min(x, dim= 1) # return (val, index)
        else:
            raise NotImplementedError("Unsupported pool type in set convolution!")
        return x


class MSCN(Module):
    def __init__(self, pred_in_ch, pred_hid_ch, pred_out_ch, mlp_hid_ch, return_rep: bool= False):
        super(MSCN, self).__init__()
        self.pred_set_conv = SetConvolution(pred_in_ch, pred_hid_ch, pred_out_ch, num_layers=2)
        self.mlp = MLP(in_ch=pred_out_ch, hid_ch=mlp_hid_ch, out_ch= 1, return_rep=return_rep)

    def forward(self, pred_x):
        pred_x = self.pred_set_conv(pred_x)
        x  = self.mlp(pred_x)
        return x



class MSCNJoin(Module):
    def __init__(self, table_in_ch, table_hid_ch, table_out_ch,
                 pred_in_ch, pred_hid_ch, pred_out_ch,
                 join_in_ch, join_hid_ch, join_out_ch, mlp_hid_ch,
                 return_rep: bool= False):
        super(MSCNJoin, self).__init__()
        self.table_set_cov = SetConvolution(table_in_ch, table_hid_ch, table_out_ch, num_layers=2)
        self.pred_set_conv = SetConvolution(pred_in_ch, pred_hid_ch, pred_out_ch, num_layers=2)
        self.join_set_conv = SetConvolution(join_in_ch, join_hid_ch, join_out_ch, num_layers=2)
        self.mlp = MLP(in_ch=table_out_ch + pred_out_ch + join_out_ch, hid_ch=mlp_hid_ch, out_ch=1, return_rep=return_rep)

    def forward(self, x):
        (table_x, pred_x, join_x) = x
        #print(table_x.shape, pred_x.shape, join_x.shape)

        table_x = self.table_set_cov(table_x)
        pred_x = self.pred_set_conv(pred_x)
        join_x = self.join_set_conv(join_x)
        x = torch.cat([table_x, pred_x, join_x], dim= -1)

        #print(x.shape)
        x = self.mlp(x)
        return x
