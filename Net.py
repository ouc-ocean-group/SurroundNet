import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as ttf
import piq
channel = 32

class ASFConv(nn.Module):
    def __init__(self, Ksize, inChannel):
        super(ASFConv, self).__init__()
        self.inChannel = inChannel
        pad = int((Ksize - 1))
        self.Ksize = Ksize
        toPadY = (0, 0, pad, pad)
        self.padderY = nn.ReflectionPad2d(toPadY)
        toPadX = (pad, pad, 0, 0)
        self.padderX = nn.ReflectionPad2d(toPadX)
        self.yKernel = nn.Parameter(torch.ones([self.inChannel, 1, Ksize, 1]))  

        self.masky = nn.Parameter(torch.cat([torch.ones([self.inChannel, 1, Ksize, 1]),torch.zeros([self.inChannel, 1, Ksize-1, 1])], 2),requires_grad=False)
        self.maskx = nn.Parameter(torch.cat([torch.ones([self.inChannel, 1, 1, Ksize]),torch.zeros([self.inChannel, 1, 1, Ksize-1])], 3),requires_grad=False)
        self.m = (((2*self.Ksize-1)**2) / (self.Ksize**2)) 
        self.n = (((2*self.Ksize-1)**2) / (self.Ksize*(2*self.Ksize-1))) 
        
    def forward(self, x):
        b,c,h,w = x.shape
        yKernel = torch.abs(self.yKernel)
        yKernel = yKernel.cumsum(dim=2)
        yKernelFlip = torch.flip(yKernel, [2])[:, :, 1:, :]
        yKernel = torch.cat((yKernel, yKernelFlip), 2)
        yKernel = yKernel / (torch.sum(yKernel, 2, keepdim=True)+1e-8)
        xKernel = yKernel.permute(0, 1, 3, 2)
        xPad = self.padderY(x)
        xPad = self.padderX(xPad)  
        xPad = F.conv2d(xPad, yKernel, bias=None, padding=0, groups=self.inChannel)
        res = F.conv2d(xPad, xKernel, bias=None, padding=0, groups=self.inChannel)   
        
        return res     
                    
                  
class ARBlock(nn.Module):
    def __init__(self, Ksize, inChannel, OChannel):
        super(ARBlock, self).__init__()
        self.ASFConv = ASFConv(Ksize,inChannel)
        
        self.RPath = nn.Sequential(
                     nn.Conv2d(inChannel,OChannel,(3,3),1,padding=(2,2),bias=True,groups=1,dilation=2),
                     nn.ReLU(),
                     nn.Conv2d(OChannel,OChannel,(3,3),1,padding=(2,2),bias=True,groups=1,dilation=2),
                     nn.ReLU(),

        )

        self.IPath = nn.Sequential(
                     nn.Conv2d(inChannel,OChannel,(1,1),1,padding=(0,0),bias=True,groups=1),
                     nn.ReLU(),                  
        )

        self.fusion = nn.Sequential(
                     nn.Conv2d(inChannel*2,OChannel,(1,1),1,padding=(0,0),bias=True,groups=1),
                     nn.ReLU(),
        )

    def forward(self,x):
        b,c,h,w = x.shape
        x = x*255
        logx = torch.log(x+1)    
        CSC = self.ASFConv(x)
        CSC = torch.log(CSC+1)
        
        refl = logx - CSC
        illu = self.IPath(CSC)
        
        refl = self.RPath(refl) + refl
        clear = self.fusion(torch.cat([refl,illu],1))
        return clear,CSC


class ARBlocks(nn.Module):
    def __init__(self,Ksize,inChannel,ouChannel):
        super(ARBlocks, self).__init__()
        self.RBP1 = ARBlock(3,inChannel,inChannel) 
        self.RBP2 = ARBlock(7,inChannel,ouChannel)
        self.RBP3 = ARBlock(11,inChannel,inChannel)
        self.RBP4 = ARBlock(15,inChannel,ouChannel)
    def forward(self,x):
        RBP1,CSC1 = self.RBP1(x)
        RBP2,CSC2 = self.RBP2(x)
        RBP3,CSC3 = self.RBP3(x)
        RBP4,CSC4 = self.RBP4(x)
        return torch.cat([RBP1,RBP2,RBP3,RBP4],1),torch.cat([CSC1,CSC2,CSC3,CSC4],1)

class SECA(nn.Module):
    def __init__(self, channel, k_size=7, shuffle=False):
        super(SECA, self).__init__()
        self.shuffle = shuffle
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
                                nn.Conv1d(1, channel, kernel_size=k_size, padding=(k_size - 1) // 2, bias=True),
                                nn.Conv1d(channel, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=True),
                                )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        if self.shuffle:
            x = x.view(b,channel,int(c/(channel)),h,w).permute(0,2,1,3,4).contiguous().view(b,c,h,w)
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=((kernel_size-1)//2), bias=True,dilation=1)
    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):    
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate 
        self.dense_layers = nn.Sequential(*modules)    
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class LED(nn.Module):
    def __init__(self):
        super(LED, self).__init__()

        self.baseConv1 = nn.Sequential(
                     nn.Conv2d(3,channel,5,1,padding=2,bias=True,dilation=1),
                     nn.ReLU()
        )

        self.baseConv2 = nn.Sequential(
                     nn.Conv2d(channel,3,5,1,padding=2,bias=True,dilation=1),
        )
        
        self.secondConv1 = nn.Sequential(
                     nn.Conv2d(3+3,channel,3,1,padding=1,bias=True,dilation=1),
        )          


        self.RBD1 = RDB(channel,2,int(channel/2)) 
        self.RBD2 = RDB(channel,2,int(channel/2)) 
        self.RBD3 = RDB(channel,2,int(channel/2)) 
        
    def forward(self, x):  
 
        bsConv1 = self.baseConv1(torch.cat([x],1))
        RBD1 = self.RBD1(bsConv1)
        RBD2 = self.RBD2(RBD1)
        base = self.baseConv2(RBD2) + x
              
        scConv1 = self.secondConv1(torch.cat([base,x],1))
        scConv1 = F.relu(scConv1)
        return scConv1,base

class SurroundNet(nn.Module):
    def __init__(self):
        super(SurroundNet, self).__init__()
        self.LED = LED()
        self.ARBlocks = ARBlocks(5,channel,channel)
        self.SECA = SECA(((channel)*4)+channel,9)                     
        self.resConv = nn.Sequential(
                     nn.Conv2d(((channel)*4)+channel,3,5,1,padding=2,bias=True),
        )        

    def forward(self,x):
        b,c,h,w = x.shape
        features,base = self.LED(x)
        ARBlocks,_ = self.ARBlocks(features) #1 
        SECA = self.SECA(torch.cat([ARBlocks,features],1))
        res = self.resConv(SECA)
        res = res+base
        return res

from SSIM import SSIM
class L1CharbonnierLoss(torch.nn.Module):
    def __init__(self):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
 
class SurroundNetLoss(nn.Module):
    def __init__(self):
        super(SurroundNetLoss, self).__init__()
        self.DISTSLoss1 = piq.DISTS()
        self.SSIMLoss1 = SSIM(val_range=1.)  
        self.l1Loss = L1CharbonnierLoss()

    def forward(self, res, highImg, base, LEDImg):
        self.CLoss1 = 1-self.SSIMLoss(res,highImg) + self.l1Loss(res,highImg) + self.DISTSLoss(res,highImg) 
        self.CLoss2 = 1-self.SSIMLoss(base,LEDImg) + self.l1Loss(base,LEDImg) + self.DISTSLoss(base,LEDImg) 

        self.loss =  self.CLoss1 + self.CLoss2 
        return self.loss
          
        
if __name__ == '__main__':
    Net = SurroundNet()
    input = torch.ones([1,3,128,128])
    Net.eval()
    output = Net(input)
    print(output.shape)
