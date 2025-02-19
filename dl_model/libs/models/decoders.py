"""
decoders / heads
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F


class CatLinear(nn.Module):
    """
    Concat features (w. dropout) + linear classifier
    """
    def __init__(self, in_channels, input_views, output_dims,
                 dropout_prob=0.0, fc_std=0.01, fc_bias=None):
        super(CatLinear, self).__init__()
        # set up params
        self.in_channels = in_channels
        self.input_views = input_views
        self.output_dims = output_dims
        self.dropout_prob = dropout_prob
        self.fc_std = fc_std
        self.fc_bias = fc_bias
        #self.interv_num = interv_num

        # note: DO NOT use inplace droput
        # (will trigger a weird bug in distributed training)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(self.dropout_prob)
        # print('in_channels = ', self.in_channels)
        # print('input_views = ', self.input_views)
        # self.fc = nn.Linear(self.in_channels, self.output_dims, bias=True)
        self.fc1 = nn.Linear(self.in_channels*self.input_views,
                             self.output_dims, bias=True)
        #self.fc2 = nn.Linear(self.output_dims, self.output_dims, bias=True)
        self.reset_params()

    def reset_params(self):
        # manuall init fc params
        nn.init.normal_(self.fc1.weight, 0.0, self.fc_std)
        #nn.init.normal_(self.fc2.weight, 0.0, self.fc_std)
        if self.fc_bias is None:
            nn.init.constant_(self.fc1.bias, 0.0)
            #nn.init.constant_(self.fc2.bias, 0.0)
        else:
            self.fc1.bias.data = torch.from_numpy(self.fc_bias.copy())
            #self.fc2.bias.data = torch.from_numpy(self.fc_bias.copy())

    def forward(self, x):
        # print(len(x), x[-1].shape)
        if not isinstance(x, tuple):   # only 3 dimensions, batch x l x in_size in lstm
            out = x[:, -1, :]
        else:
            # reshape from n * v, c, ... -> n, v * c, ...
            if x[-1].dim() == 3:
                out = self.avgpool1d(x[-1])
                # out = x[-1]
            else:
                n, c, h, w = x[-1].shape
                x_in = x[-1].view(n // self.input_views,
                                self.input_views * c, h, w)
                # -> n, v * c
                out = self.avgpool(x_in)
            out = out.reshape(out.shape[0], -1)
        out = self.dropout(out)
        out = self.fc1(out)
        #out = nn.ReLU()(out)
        #out = self.fc2(out)
        return out

class CatLinearCTC(nn.Module):
    """
    Concat features (w. dropout) + linear classifier
    """
    def __init__(self, in_channels, input_views, output_dims,
                 dropout_prob=0.0, fc_std=0.01, fc_bias=None):
        super(CatLinearCTC, self).__init__()
        # set up params
        self.in_channels = in_channels
        self.input_views = input_views
        self.output_dims = output_dims
        self.dropout_prob = dropout_prob
        self.fc_std = fc_std
        self.fc_bias = fc_bias
        #self.interv_num = interv_num

        # note: DO NOT use inplace droput
        # (will trigger a weird bug in distributed training)
        self.avgpool = nn.AdaptiveAvgPool2d((512, 1))
        self.dropout = nn.Dropout(self.dropout_prob)
        # print('in_channels = ', self.in_channels)
        # print('input_views = ', self.input_views)
        # self.fc = nn.Linear(self.in_channels, self.output_dims, bias=True)
        self.fc1 = nn.Linear(self.in_channels*self.input_views,
                             self.output_dims, bias=True)
        #self.fc2 = nn.Linear(self.output_dims, self.output_dims, bias=True)
        self.reset_params()

    def reset_params(self):
        # manuall init fc params
        nn.init.normal_(self.fc1.weight, 0.0, self.fc_std)
        #nn.init.normal_(self.fc2.weight, 0.0, self.fc_std)
        if self.fc_bias is None:
            nn.init.constant_(self.fc1.bias, 0.0)
            #nn.init.constant_(self.fc2.bias, 0.0)
        else:
            self.fc1.bias.data = torch.from_numpy(self.fc_bias.copy())
            #self.fc2.bias.data = torch.from_numpy(self.fc_bias.copy())

    def forward(self, x):
        # print(len(x), x[-1].shape)
        if x[-1].dim() == 3:
            n, l, c = x[-1].shape
            in_vec = x[-1]
        elif x[-1].dim() == 4:  # n x c x l x h
            in_vec = x[-1]
            n, c, l, h = in_vec.shape
            # print(in_vec.shape)
            in_vec = in_vec.transpose(1, 2)         # feats: N x C x L x height -> N x L x C x h
            in_vec = self.avgpool(in_vec)           # n x l x c x 1
            in_vec = in_vec.reshape(n, l, c)        # feats: N x L x C x 1 -> N x L x C
        out = torch.stack([F.log_softmax(self.dropout(self.fc1(in_vec[i])), dim=-1) for i in range(n)])
        # print(out.shape)
        # if not isinstance(x, tuple):   # only 3 dimensions, batch x l x in_size in lstm
        #     out = x[:, -1, :]
        # else:
        #     # reshape from n * v, c, ... -> n, v * c, ...
        #     n, c, h, w = x[-1].shape
        #     x_in = x[-1].view(n // self.input_views,
        #                     self.input_views * c, h, w)
        #     # -> n, v * c
        #     out = self.avgpool(x_in)
        #     out = out.view(out.shape[0], -1)
        # out = self.dropout(out)
        # out = self.fc1(out)
        #out = nn.ReLU()(out)
        #out = self.fc2(out)
        return out


class SumLinear(nn.Module):
    """
    Sum of features (w. dropout) + linear classifier
    """
    def __init__(self, in_channels, input_views, output_dims,
                 dropout_prob=0.0, fc_std=0.01, fc_bias=None):
        super(SumLinear, self).__init__()
        # set up params
        self.in_channels = in_channels
        self.input_views = input_views
        self.output_dims = output_dims
        self.dropout_prob = dropout_prob
        self.fc_std = fc_std
        self.fc_bias = fc_bias

        # note: DO NOT use inplace droput
        # (will trigger a weird bug in distributed training)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc = nn.Linear(self.in_channels, self.output_dims, bias=True)

        self.reset_params()

    def reset_params(self):
        # manuall init fc params
        nn.init.normal_(self.fc.weight, 0.0, self.fc_std)
        if self.fc_bias is None:
            nn.init.constant_(self.fc.bias, 0.0)
        else:
            self.fc.bias.data = torch.from_numpy(self.fc_bias.copy())

    def forward(self, x):
        if len(x.shape) == 3:   # only 3 dimensions, batch x l x in_size in lstm
            x_in = x[:, -1, :]
        else:
            # reshape from n * v, c, ... -> n, v * c, ...
            n, c, h, w = x[-1].shape
            x_in = x[-1].view(n // self.input_views,
                            self.input_views * c, h, w)
            # -> n, v * c
        out = self.avgpool(x_in.sum(1))
        out = out.view(out.shape[0], -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out



class MaxLinear(nn.Module):
    """
    Sum of features (w. dropout) + linear classifier
    """
    def __init__(self, in_channels, input_views, output_dims,
                 dropout_prob=0.0, fc_std=0.01, fc_bias=None):
        super(MaxLinear, self).__init__()
        # set up params
        self.in_channels = in_channels
        self.input_views = input_views
        self.output_dims = output_dims
        self.dropout_prob = dropout_prob
        self.fc_std = fc_std
        self.fc_bias = fc_bias

        # note: DO NOT use inplace droput
        # (will trigger a weird bug in distributed training)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(self.dropout_prob)
        self.fc = nn.Linear(self.in_channels, self.output_dims, bias=True)

        self.reset_params()

    def reset_params(self):
        # manuall init fc params
        nn.init.normal_(self.fc.weight, 0.0, self.fc_std)
        if self.fc_bias is None:
            nn.init.constant_(self.fc.bias, 0.0)
        else:
            self.fc.bias.data = torch.from_numpy(self.fc_bias.copy())

    def forward(self, x):
        # reshape from n * v, c, ... -> n, v * c, ...
        n, c, h, w = x[-1].shape
        x_in = x[-1].view(n // self.input_views, self.input_views, c * h * w)
        # -> n, v * c
        out = torch.max(x_in, 1)
        out = out.view(out.shape[0], -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class VaeDecoder(nn.Module):
    """
    VAE decoder
    """
    def __init__(self, in_channels, input_views, output_dims,
                 hidden_dims, dropout_prob=0.0, fc_std=0.01, fc_bias=None):
        super().__init__()
        # set up params
        self.z_dims = in_channels
        self.input_views = input_views
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob
        self.fc_std = fc_std
        self.fc_bias = fc_bias

        # from bottleneck to hidden 400
        self.linear = nn.Linear(self.z_dims, self.hidden_dims)
        self.out = nn.Linear(self.hidden_dims, self.output_dims)

    def forward(self, x):
        z_mu, z_var = x[0], x[1]
        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = torch.add(torch.mul(eps, std), z_mu)
        hidden = F.relu(self.linear(x_sample))
        predicted = torch.sigmoid(self.out(hidden))
        print('pridicted_shape = ', predicted.shape)
        return predicted

