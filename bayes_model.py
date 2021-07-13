from copy import deepcopy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from bayes_layer import Conv2dFlipout, LinearFlipout
import bayesian_torch.models.bayesian.resnet_flipout as BayesResNet

logger = logging.getLogger(__name__)


class ModelEMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def forward(self, input):
        return self.module(input)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.parameters(), model.parameters()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            for ema_v, model_v in zip(self.module.buffers(), model.buffers()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(model_v)

    def update_parameters(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)

class BayesBasicBlock(nn.Module):
    expansion=1

    def __init__(self, in_planes, out_planes, stride, dropout=0.0, activate_before_residual=False):
        super(BayesBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.01, eps=1e-3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = Conv2dFlipout(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.01, eps=1e-3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = Conv2dFlipout(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = dropout
        self.equalInOut = (in_planes == out_planes)

        self.convShortcut = (not self.equalInOut) and Conv2dFlipout(in_planes,
                                  self.expansion * out_planes,
                                  kernel_size=1,
                                  stride=stride,
                                  padding=0,
                                  bias=False) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        whole_sampled = {}
        if not self.equalInOut and self.activate_before_residual == True:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))
        
        kl_sum = 0

        out, kl, sampled = self.conv1(out if self.equalInOut else x)
        whole_sampled["conv1"] = sampled
        kl_sum += kl

        out = F.relu(self.bn2(out))
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        out, kl, sampled = self.conv2(out)
        kl_sum += kl
        whole_sampled["conv2"] = sampled

        if not self.equalInOut:
            short, kl, sampled = self.convShortcut(x)
            kl_sum += kl
            whole_sampled["short"] = sampled
        else:
            short = x

        return torch.add(short, out),  kl_sum, whole_sampled

class BayesNetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout=0.0,
                 activate_before_residual=False):
        super(BayesNetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropout, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout,
                    activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropout, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        kl_sum = 0 
        whole_sampled = {}
        out = x
        for id, l in enumerate(self.layer):
            out, kl, sampled = l(out)
            kl_sum += kl
            whole_sampled[id] = sampled
        return out, kl, whole_sampled


class BayesWideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, dropout=0.0, dense_dropout=0.0):
        super(BayesWideResNet, self).__init__()
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BayesBasicBlock
        # 1st conv before any network block
        self.conv1 = Conv2dFlipout(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = BayesNetworkBlock(
            n, channels[0], channels[1], block, 1, dropout, activate_before_residual=True)
        # 2nd block
        self.block2 = BayesNetworkBlock(
            n, channels[1], channels[2], block, 2, dropout)
        # 3rd block
        self.block3 = BayesNetworkBlock(
            n, channels[2], channels[3], block, 2, dropout)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.01, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dense_dropout)
        self.fc = LinearFlipout(channels[3], num_classes)
        self.channels = channels[3]

        self.apply(BayesResNet._weights_init)

    def forward(self, x):
        kl_sum = 0
        whole_sampled = {}

        out, kl, sampled = self.conv1(x)
        kl_sum += kl
        whole_sampled["conv1"] = sampled

        out, kl, sampled = self.block1(out)
        kl_sum += kl
        whole_sampled["block1"] = sampled

        out, kl, sampled = self.block2(out)
        kl_sum += kl
        whole_sampled["block2"] = sampled

        out, kl, sampled = self.block3(out)
        kl_sum += kl
        whole_sampled["block3"] = sampled

        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)

        out, kl, sampled = self.fc(self.drop(out))
        whole_sampled["fc"] = sampled

        kl_sum += kl
        return out, kl, whole_sampled

def build_bayes_wideresnet(args):
    if args.dataset == "cifar10":
        depth, widen_factor = 28, 2
    elif args.dataset == 'cifar100':
        depth, widen_factor = 28, 8

    model = BayesWideResNet(num_classes=args.num_classes,
                       depth=depth,
                       widen_factor=widen_factor,
                       dropout=0,
                       dense_dropout=args.dense_dropout)
    if args.local_rank in [-1, 0]:
        logger.info(f"Model: WideResNet {depth}x{widen_factor}")
        logger.info(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model
