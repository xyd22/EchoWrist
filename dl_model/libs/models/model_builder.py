from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# import your models here
from .resnet_models import resnet18, resnet34, resnet50, resnet101, resnet152
from .rnn_model import myRNN
from .decoders import SumLinear, CatLinear, CatLinearCTC

################################################################################


def build_backbone(network_config):
    """get model
    return a supported network
    """
    model_name = network_config['backbone'].lower()
    if model_name in ['lstm', 'gru']:
        network_config['rnn']['rnn_type'] = model_name
    model = {
        'resnet18':  partial(resnet18,
                             pretrained=network_config['pretrained'],
                             frozen_stages=network_config['frozen_stages'],
                             input_channels=network_config['input_channels']),
        'resnet34':  partial(resnet34,
                             pretrained=network_config['pretrained'],
                             frozen_stages=network_config['frozen_stages'],
                             input_channels=network_config['input_channels']),
        'resnet50':  partial(resnet50,
                             pretrained=network_config['pretrained'],
                             frozen_stages=network_config['frozen_stages'],
                             input_channels=network_config['input_channels']),
        'resnet101': partial(resnet101,
                             pretrained=network_config['pretrained'],
                             frozen_stages=network_config['frozen_stages'],
                             input_channels=network_config['input_channels']),
        'resnet152': partial(resnet152,
                             pretrained=network_config['pretrained'],
                             frozen_stages=network_config['frozen_stages'],
                             input_channels=network_config['input_channels']),
        'lstm': partial(myRNN, rnn_config=network_config['rnn']),
        'gru': partial(myRNN, rnn_config=network_config['rnn']),
        # new models ....
    }[model_name]

    return model


def build_decoder(network_config):
    """Get the head of the network
    return a supported decoder
    """
    model_name = network_config['decoder']
    model = {
        'sumfc':  partial(SumLinear,
                          in_channels=network_config['feat_dim'],
                          input_views=network_config['input_views'],
                          output_dims=network_config['output_dims'],
                          dropout_prob=network_config['dropout_prob'],
                          fc_std=network_config['decoder_fc_std'],
                          fc_bias=network_config['decoder_fc_bias']),
        'catfc':  partial(CatLinear,
                          in_channels=network_config['feat_dim'],
                          input_views=network_config['input_views'],
                          output_dims=network_config['output_dims'],
                          dropout_prob=network_config['dropout_prob'],
                          fc_std=network_config['decoder_fc_std'],
                          fc_bias=network_config['decoder_fc_bias']),
        'catfcctc':  partial(CatLinearCTC,
                          in_channels=network_config['feat_dim'],
                          input_views=network_config['input_views'],
                          output_dims=network_config['output_dims'],
                          dropout_prob=network_config['dropout_prob'],
                          fc_std=network_config['decoder_fc_std'],
                          fc_bias=network_config['decoder_fc_bias']),
        # new decoders ....
    }[model_name]

    return model


def reconNkldiver_loss(outputs, targets, z_mu, z_var):
    # reconstruction loss
    loss = nn.MSELoss()
    print('outputs_shape = ', outputs.shape)
    print('targets_shape = ', targets.shape)
    recon_loss = loss(outputs, targets)
    # KL Divergence loss
    kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)
    return recon_loss + kl_loss


def radious_to_coordinates(finger_radious_batch):
    bone_length = {1: 28.74, 2: 46.60, 3: 31.50, 4: 16.50,
                   5: 80.39, 6: 40.40, 7: 22.36, 8: 11.66,
                   9: 75.05, 10: 45.09, 11: 26.44, 12: 13.03,
                   13: 69.07, 14: 41.80, 15: 25.74, 16: 13.10,
                   17: 67.11, 18: 33.27, 19: 18.55, 20: 12.30}
    batch_size = finger_radious_batch.shape[0]
    coordinates_batch = torch.zeros(
        (batch_size, 60)).to(finger_radious_batch.device)
    for b in range(batch_size):
        # finger_radious shape: (batch_size, 40) -> (batch_size, 20, 2)
        finger_radious_single = torch.reshape(
            finger_radious_batch[b, :], (20, 2))
        length = finger_radious_single.shape[0]
        coordinates = torch.zeros((length, 3))
        for i in range(length):
            radious_xy, radious_z = finger_radious_single[i, :]
            if i in [0, 4, 8, 12, 16]:
                if radious_z == np.pi or radious_z == 0:
                    coordinates[i, :] = torch.tensor(
                        [0, 0, np.cos(radious_z)]) * bone_length[i + 1]
                else:
                    z_axis = torch.cos(radious_z) * bone_length[i + 1]
                    x_axis = torch.sin(radious_z) * \
                        torch.cos(radious_xy) * bone_length[i + 1]
                    y_axis = torch.sin(radious_z) * \
                        torch.sin(radious_xy) * bone_length[i + 1]
                    coordinates[i, :] = torch.tensor([x_axis, y_axis, z_axis])
            else:
                if radious_z == np.pi or radious_z == 0:
                    coordinates[i, :] = coordinates[i - 1, :] + \
                        torch.tensor([0, 0, torch.cos(radious_z)]
                                     ) * bone_length[i + 1]
                else:
                    z_axis = torch.cos(radious_z) * bone_length[i + 1]
                    x_axis = torch.sin(radious_z) * \
                        torch.cos(radious_xy) * bone_length[i + 1]
                    y_axis = torch.sin(radious_z) * \
                        torch.sin(radious_xy) * bone_length[i + 1]
                    coordinates[i, :] = coordinates[i - 1, :] + \
                        torch.tensor([x_axis, y_axis, z_axis])
        coordinates_batch[b, :] = coordinates.reshape((60,))
    return coordinates_batch


# class Point_dis_loss(nn.Module):
#     def __init__(self):
#         super().__init__()

def Point_dis_loss(outputs, targets):
    diff_squared = torch.pow(outputs - targets, 2)
    # print(outputs.shape)
    dis_squared = diff_squared[:, range(0, outputs.shape[1], 3)] + diff_squared[:, range(1, outputs.shape[1], 3)] + diff_squared[:, range(2, outputs.shape[1], 3)]
    # err = torch.pow(dis_squared, 0.5)
    err = dis_squared
    # err = dis_squared ** 2
    return torch.mean(err)


def calculate_dis(outputs, targets):
    diff_squared = torch.pow(outputs - targets, 2)
    # print(outputs.shape)
    dis_squared = diff_squared[:, range(0, outputs.shape[1], 3)] + diff_squared[:, range(1, outputs.shape[1], 3)] + diff_squared[:, range(2, outputs.shape[1], 3)]
    # dis_squared = diff_squared[:, range(0, 40, 2)] + diff_squared[:, range(1, 40, 2)]
    err = torch.pow(dis_squared, 0.5)
    # print(err.shape)
    # err_meta = torch.mean(err[range(0, 20, 4)])
    # err_proximal = torch.mean(err[range(1, 20, 4)])
    # err_distal = torch.mean(err[range(2, 20, 4)])
    # err_tip = torch.mean(err[range(3, 20, 4)])
    # err = dis_squared ** 2
    # return torch.mean(err), err_meta, err_proximal, err_distal, err_tip
    return torch.mean(err), torch.mean(err), torch.mean(err), torch.mean(err), torch.mean(err)

def angleDistLoss(outputs, targets):
    loss = nn.MSELoss()
    # print('outputs_shape = ', outputs.shape)
    # print('targets_shape = ', targets.shape)
    angle_loss = loss(outputs, targets[:, :40])
    distFromAngles = radious_to_coordinates(outputs)
    dist_loss = loss(distFromAngles, targets[:, 40:])
    return angle_loss + dist_loss

def weight4060loss(outputs, targets):
    loss = nn.MSELoss()
    loss0 = torch.mean(torch.abs(outputs - targets), 1)
    # print(loss0.shape)
    weight40 = (loss0 > 40) * 19 + 1
    weight60 = (loss0 > 60) + 1
    return loss((outputs.T * weight40 * weight60).T, (targets.T * weight40 * weight60).T)


def build_loss(network_config):
    """Get the loss function
    """
    if network_config['loss_type'] == 'l1':
        criterion = nn.L1Loss()
    elif network_config['loss_type'] == 'l2':
        criterion = nn.MSELoss()
    elif network_config['loss_type'] == 'huber':
        criterion = nn.SmoothL1Loss()
    elif network_config['loss_type'] == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif network_config['loss_type'] == 'ctc':
        criterion = nn.CTCLoss(blank=network_config['output_dims'] - 1)
    elif network_config['loss_type'] == '4060':
        criterion = weight4060loss
    elif network_config['loss_type'] == 'dis':
        criterion = Point_dis_loss
    elif network_config['loss_type'] == 'frame_weighted':
        criterion = Point_dis_loss
    else:
        raise ValueError('Unsupported loss type!')
    return criterion


class EncoderDecoder(nn.Module):
    """A thin wrapper that builds a full model from network config
    This model will include:
        encoder: backbone network
        decoder: decoder network
        loss: loss for training
    An example network config
    {
      # multi gpu support
      "devices": [0],  # default: single gpu
      "backbone": "resnet18", # backbone network
      "pretrained": False,    # if the backbone is pre-trained
      "frozen_stages": -1,    # freeze part of the network
      "decoder": "catfc",     # decoder for classification
      "feat_dim": 512,        # input feat dim to decoder
      "decoder_fc_std": 0.01, # init fc std for decoder
      "dropout_prob": 0.5,    # dropout ratio for fc in decoder
      "loss_type": 'l1',      # which loss to use
      "input_views": 4        # auto infer from dataset
      "output_dims": 15       # auto infer from dataset
    }
    """

    def __init__(self, network_config):
        super(EncoderDecoder, self).__init__()
        # delayed instaniation
        encoder = build_backbone(network_config)
        self.encoder = encoder()
        self.rnn = None
        if (network_config['backbone'] not in ['lstm', 'gru']) and network_config['rnn']['cat_backbone'] or network_config['crnn_sw']['applied']:
            if network_config['crnn_sw']['applied']:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            else:
                self.avgpool = nn.AdaptiveAvgPool2d((512, 1))
            self.rnn = partial(myRNN, rnn_config=network_config['rnn'])()
            self.dropout = nn.Dropout(network_config['dropout_prob'])

        decoder = build_decoder(network_config)
        self.decoder = decoder()
        # get loss function
        self.criterion = build_loss(network_config)
        # save the config
        self.network_config = network_config

    def weighted(self, n):
        weight = self.network_config["weight"]
        weight_points = self.network_config["weighted_points"]
        output_num = self.network_config["output_dims"]
        kernel = torch.ones([n, output_num])
        for i in range(0, len(weight_points)):
            num = weight_points[i]
            kernel[:, num*2: num*2+2] *= weight[i]
            kernel[:, num*2: num*2+2] *= weight[i]
        return kernel

    def silent_pos(self, n):
        output_num = self.network_config["output_dims"]
        kernel = torch.ones([n, output_num])
        b_pos = self.network_config["silent_pos"]
        # weight_points = self.network_config["weighted_points"]
        for i in range(0, len(b_pos)):
            # num = weight_points[i]
            # kernel[:, num*2 : num*2+2] *= weight[i]
            kernel[:, i] = b_pos[i]
        return kernel
    
    def weighted_loss_function(self, n, action):
        output_num = self.network_config["output_dims"]
        loss_weight = self.network_config["loss_weight"]
        kernel = torch.ones([n, output_num])
        for i in range(0, n):
            if action[i] == 0:
                weight = loss_weight[0]
            else:
                weight = loss_weight[1]
            kernel[i, :] *= weight
        return kernel

    def forward(self, imgs, targets=None, action_label=None):
        # print(imgs.shape)
        if (imgs.dim() == 3):
            # K C H W (single sample testing)
            outputs = self._forward(imgs)
            return outputs
        elif imgs.dim() == 4:
            # N: batch_size, K: view_num, C: channel_num, H: height, W: width
            # N K C H W (batch input for training/testing)

            n, c, h, w = imgs.size()
            # print('img_size = ', imgs.size())
            outputs = self._forward(imgs)  # .view(n * k, c, h, w))
            # if (targets is not None) and self.training:
            if (targets is not None):
                if self.network_config['loss_type'] == 'distance_constraint':
                    loss = angleDistLoss(outputs, targets)
                    return outputs, loss
                elif self.network_config['loss_type'] == 'reconNkldiver':
                    loss = reconNkldiver_loss(
                        outputs, targets, self.z_mu, self.z_var)
                    return outputs, loss
                # elif self.network_config['loss_type'] == 'dis':
                #     loss = Point_dis_loss(outputs, targets)
                #     return outputs, loss
                elif self.network_config['loss_type'] == 'frame_weighted':
                    kernel = self.silent_pos(n).cuda(
                            self.network_config['devices'][0], non_blocking=True)
                    loss_original = self.criterion(outputs, targets)
                    loss_frame_weight = self.criterion(kernel, targets)
                    loss = loss_original + torch.mul(loss_original, loss_frame_weight)
                    return outputs, loss
                else:
                    if self.network_config["if_weight_4060"] > 0:
                        loss0 = self.criterion(outputs, targets)
                        print(outputs.shape, targets.shape, loss0.shape)
                        weight40 = (loss0 > 40) * 19 + 1
                        weight60 = (loss0 > 60) + 1
                        print(weight40, weight40.size())
                        # for w in loss0:
                        #     print(w)
                        loss0 = torch.mul(loss0, weight40)
                        loss = torch.mul(loss0, weight60)
                        #raise KeyboardInterrupt

                    elif self.network_config["if_weight_loss"] > 0:
                        kernel = self.weighted_loss_function(n, action_label).cuda(
                            self.network_config['devices'][0], non_blocking=True)
                        outputs2 = torch.mul(outputs, kernel)
                        targets2 = torch.mul(targets, kernel)
                        loss = self.criterion(outputs2, targets2)
                        #raise KeyboardInterrupt

                    elif self.network_config["if_weight"] > 0:
                        kernel = self.weighted(n).cuda(
                            self.network_config['devices'][0], non_blocking=True)
                        # print(outputs.shape, targets.shape, kernel.shape)
                        outputs2 = torch.mul(outputs, kernel)
                        targets2 = torch.mul(targets, kernel)
                        loss = self.criterion(outputs2, targets2)

                    elif self.network_config["if_weight_mouth"] > 0:
                        kernel = self.weighted_more_on_mouth(n).cuda(
                            self.network_config['devices'][0], non_blocking=True)
                        outputs2 = torch.mul(outputs, kernel)
                        targets2 = torch.mul(targets, kernel)
                        loss = self.criterion(outputs2, targets2)
                    elif self.network_config['loss_type'] == 'ctc':
                        # outputs: N x L x D
                        target_lengths = torch.IntTensor([len(x.split()) for x in targets]).cuda(
                            self.network_config['devices'][0], non_blocking=True)
                        targets_cuda = []
                        for t in targets:
                            targets_cuda += [int(x) for x in t.split()]
                        targets_cuda = torch.IntTensor(targets_cuda).cuda(
                            self.network_config['devices'][0], non_blocking=True)
                        pred_lengths = torch.IntTensor(n).fill_(outputs.shape[1])
                        loss = self.criterion(outputs.transpose(0, 1), targets_cuda, pred_lengths, target_lengths)
                    else:
                        # print(outputs.shape, targets.shape)
                        loss = self.criterion(outputs, targets)
                    return outputs, loss
            else:
                return outputs#, None
        else:
            raise TypeError("Input size mis-match!")

    def _forward(self, imgs):
        # print(imgs.shape, end=' ')
        if self.network_config['crnn_sw']['applied']:
            feats = []
            for i in range(0, imgs.shape[2] - self.network_config['crnn_sw']['window'], self.network_config['crnn_sw']['stride']):
                imgs_piece = imgs[:, :, i: i + self.network_config['crnn_sw']['window'], :]
                # print(imgs_piece.shape)
                feats_piece = self.encoder(imgs_piece)[-1]
                n, c, l, h = feats_piece.shape
                feats_piece = self.avgpool(feats_piece)
                feats_piece = feats_piece.reshape(n, c)
                # feats_piece = self.dropout(feats_piece)
                feats += [feats_piece]
            feats = torch.stack(feats)              # L x n x c
            feats = feats.transpose(0, 1)           # feats: L x n x c -> N x L x C

            # feats = self.rnn(feats)             # N x L x C
            feats = (feats,)

        else:
            feats = self.encoder(imgs)
            # print(imgs.shape, end=' ')
            if self.rnn is not None:
                n, c, l, h = feats[-1].shape
                # print(imgs.shape, feats[0].shape, feats[1].shape, feats[2].shape, feats[3].shape)
                # print(feats)
                # print(feats[-1].shape)
                feats = feats[-1].transpose(1, 2)           # feats: N x C x L x height -> N x L x C x h
                # feats = feats.transpose(2, 3)           # feats: N x L x C x h -> N x L x h x C
                feats = self.avgpool(feats)
                feats = feats.reshape(n, l, -1)      # feats: N x L x C x h -> N x L x (C * h)
                # # print(feats.shape)
                # feats = self.dropout(feats)             # N x L x C
                feats = self.rnn(feats)             # N x L x C
                feats = (feats,)
        # feats = feats.transpose(1, 2)           # N x C x L
        # feats = feats.reshape(n, c, l, 1)
        # feats = feats.transpose(1, 2)
        # feats = feats.transpose(1, 3)
        # print(feats.shape)
        outputs = self.decoder(feats)
        # self.z_mu =feats[0]
        # self.z_var = feats[1]
        return outputs
