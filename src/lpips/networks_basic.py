from __future__ import absolute_import

import torch
import torch.nn as nn
from torch.autograd import Variable

from . import pretrained_networks as pn
from . import util
from ..models.i3d.pytorch_i3d import InceptionI3d

def spatial_average(in_tens, keepdim=True):
    """Takes the average over non-batch and channel dimensions of the input.

    :param in_tens: 1 x 1 [x T] x H x W
    :param keepdim: Whether to retain the dimensions of the input
    """
    B, C = in_tens.size(0), in_tens.size(1)
    in_tens_reshaped = in_tens.view(B, C, -1)
    in_tens_mean = in_tens_reshaped.mean(dim=-1)
    if keepdim:
        in_tens_mean = in_tens_mean.view(*tuple([1 for _ in range(len(in_tens.size()))]))
    return in_tens_mean

def upsample(in_tens, out_H=64): # assumes scale factor is same for H and W
    in_H = in_tens.shape[2]
    scale_factor = 1.*out_H/in_H

    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)

class PNetLinSqueeze(nn.Module):
    def __init__(self):
        super().__init__()

        self.scaling_layer = ScalingLayer()

        self.chns = [64,128,256,384,384,512,512]
        self.L = len(self.chns)

        self.net = pn.squeezenet(pretrained=False, requires_grad=False)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=True)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=True)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=True)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=True)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=True)
        self.lin5 = NetLinLayer(self.chns[5], use_dropout=True)
        self.lin6 = NetLinLayer(self.chns[6], use_dropout=True)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4, self.lin5, self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = util.normalize_tensor(outs0[kk]), util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1,self.L):
            val += res[l]

        if retPerLayer:
            return val, res
        else:
            return val


class PNetLinAlex(nn.Module):
    def __init__(self):
        super().__init__()

        self.scaling_layer = ScalingLayer()

        self.chns = [64, 192, 384, 256, 256]
        self.L = len(self.chns)

        self.net = pn.alexnet(pretrained=False, requires_grad=False)

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=True)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=True)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=True)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=True)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=True)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

    def forward(self, in0, in1, retPerLayer=False):
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = util.normalize_tensor(outs0[kk]), util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        res = [spatial_average(self.lins[kk].model(diffs[kk]), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1,self.L):
            val += res[l]

        if retPerLayer:
            return val, res
        else:
            return val


class PNetLinI3D(InceptionI3d):
    def __init__(self):
        super().__init__(400, in_channels=3)

        self.endpoint_names = ['Conv3d_2c_3x3', 'Mixed_3c', 'Mixed_4f', 'Mixed_5c']
        self.chns = [192, 480, 832, 1024]
        self.L = len(self.chns)

    def forward(self, in0, in1, retPerLayer=False):
        in0_input, in1_input = in0, in1
        outs0 = self.extract_features(in0_input, self.endpoint_names)
        outs1 = self.extract_features(in1_input, self.endpoint_names)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = util.normalize_tensor(outs0[kk]), util.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1,self.L):
            val += res[l]

        if retPerLayer:
            return val, res
        else:
            return val

class ScalingLayer(nn.Module):
    def __init__(self, num_dims=4):
        """

        :param num_dims: The number of dimensions that the input will have
        """
        super(ScalingLayer, self).__init__()
        assert num_dims == 4 or num_dims == 5

        if num_dims == 4:
            # Batch of images (B x C x H x W)
            self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
            self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])
        elif num_dims == 5:
            # Batch of videos (B x C x T x H x W)
            self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None,None])
            self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None,None])


    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)


class Dist2LogitLayer(nn.Module):
    ''' takes 2 distances, puts through fc layers, spits out value between [0,1] (if use_sigmoid is True) '''
    def __init__(self, chn_mid=32, use_sigmoid=True):
        super(Dist2LogitLayer, self).__init__()

        layers = [nn.Conv2d(5, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, chn_mid, 1, stride=1, padding=0, bias=True),]
        layers += [nn.LeakyReLU(0.2,True),]
        layers += [nn.Conv2d(chn_mid, 1, 1, stride=1, padding=0, bias=True),]
        if(use_sigmoid):
            layers += [nn.Sigmoid(),]
        self.model = nn.Sequential(*layers)

    def forward(self,d0,d1,eps=0.1):
        return self.model.forward(torch.cat((d0,d1,d0-d1,d0/(d1+eps),d1/(d0+eps)),dim=1))

class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super(BCERankingLoss, self).__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        # self.parameters = list(self.net.parameters())
        self.loss = torch.nn.BCELoss()

    def forward(self, d0, d1, judge):
        per = (judge+1.)/2.
        self.logit = self.net.forward(d0,d1)
        return self.loss(self.logit, per)

# L2, DSSIM metrics
class FakeNet(nn.Module):
    def __init__(self, use_gpu=True, colorspace='Lab'):
        super(FakeNet, self).__init__()
        self.use_gpu = use_gpu
        self.colorspace=colorspace

class L2(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            (N,C,X,Y) = in0.size()
            value = torch.mean(torch.mean(torch.mean((in0-in1)**2,dim=1).view(N,1,X,Y),dim=2).view(N,1,1,Y),dim=3).view(N)
            return value
        elif(self.colorspace=='Lab'):
            value = util.l2(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)),
                                    util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.).astype('float')
            ret_var = Variable( torch.Tensor((value,) ) )
            if(self.use_gpu):
                ret_var = ret_var.cuda()
            return ret_var

class DSSIM(FakeNet):

    def forward(self, in0, in1, retPerLayer=None):
        assert(in0.size()[0]==1) # currently only supports batchSize 1

        if(self.colorspace=='RGB'):
            value = util.dssim(1. * util.tensor2im(in0.data), 1. * util.tensor2im(in1.data), range=255.).astype('float')
        elif(self.colorspace=='Lab'):
            value = util.dssim(util.tensor2np(util.tensor2tensorlab(in0.data, to_norm=False)),
                                       util.tensor2np(util.tensor2tensorlab(in1.data, to_norm=False)), range=100.).astype('float')
        ret_var = Variable( torch.Tensor((value,) ) )
        if(self.use_gpu):
            ret_var = ret_var.cuda()
        return ret_var

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network',net)
    print('Total number of parameters: %d' % num_params)
