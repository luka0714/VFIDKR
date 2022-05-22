"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren

"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory 
#from .correlation_package.modules.corr import Correlation
# from PWCNet.correlation_package_pytorch0_4.correlation import Correlation #pytorch0.4 version
from PWCNet.correlation_package_pytorch1_0.correlation import Correlation #pytorch0.4 version

import numpy as np





__all__ = [
    'pwc_dc_net', 'pwc_dc_net_old'
    ]

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

import time

class PWCDCNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, md=4):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNet,self).__init__()

        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1aa = conv(16,  16, kernel_size=3, stride=1)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2aa = conv(32,  32, kernel_size=3, stride=1)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3aa = conv(64,  64, kernel_size=3, stride=1)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4aa = conv(96,  96, kernel_size=3, stride=1)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128,128, kernel_size=3, stride=1)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.conv6aa = conv(128,196, kernel_size=3, stride=2)
        self.conv6a  = conv(196,196, kernel_size=3, stride=1)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*md+1)**2
        dd = np.cumsum([128,128,96,64,32],dtype=np.int32).astype(np.int)
        dd = [int(d) for d in dd]

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4]) 
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4]) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4]) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

        W_MAX = 2048
        H_MAX = 1024
        B_MAX = 3
        xx = torch.arange(0, W_MAX).view(1,-1).cuda().repeat(H_MAX,1)
        yy = torch.arange(0, H_MAX).view(-1,1).cuda().repeat(1,W_MAX)
        xx = xx.view(1,1,H_MAX,W_MAX).repeat(B_MAX,1,1,1)
        yy = yy.view(1,1,H_MAX,W_MAX).repeat(B_MAX,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        ## for saving time on allocating a grid in forward
        self.W_MAX = W_MAX
        self.H_MAX = H_MAX
        self.B_MAX = B_MAX
        self.grid = Variable(grid, requires_grad=False)
        # self.mask_base = Variable(torch.cuda.FloatTensor().resize_(B_MAX,).zero_() + 1)


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        # xx = torch.arange(0, W).view(1,-1).cuda().repeat(H,1)
        # yy = torch.arange(0, H).view(-1,1).cuda().repeat(1,W)
        # xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        # yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        # grid = torch.cat((xx,yy),1).float()

        # # if x.is_cuda:
        # #     grid = grid.cuda()
        # vgrid = Variable(grid) + flo
        assert(B <= self.B_MAX and H <= self.H_MAX and W <= self.W_MAX)
        vgrid = self.grid[:B,:,:H,:W] +flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0


        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        # mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = torch.autograd.Variable(torch.cuda.FloatTensor().resize_(x.size()).zero_() + 1, requires_grad = False)
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask


    def forward(self,x, output_more = False):
        # x.shape (1, 6, 512, 704)
        im1 = x[:,:3,:,:]  # (1, 3, 512, 704)
        im2 = x[:,3:,:,:]  # (1, 3, 512, 704)

        # print("\n\n***************************PWC Net details *************** \n\n")
        # start=  time.time()
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))   # (1, 16, 256, 352)
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))   # (1, 16, 256, 352)

        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))   # (1, 32, 128, 176)
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))   # (1, 32, 128, 176)

        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))   # (1, 64, 64, 88)
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))   # (1, 64, 64, 88)

        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))   # (1, 96, 32, 44)
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))   # (1, 96, 32, 44)

        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))   # (1, 128, 16, 22)
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))   # (1, 128, 16, 22)

        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))   # (1, 196, 8, 11)
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))   # (1, 196, 8, 11)


        # print("features " +str(time.time()- start))
        # start=  time.time()
        corr6 = self.corr(c16, c26)    # (1, 81, 8, 11)
        corr6 = self.leakyRELU(corr6)   # (1, 81, 8, 11)
        x = torch.cat((self.conv6_0(corr6), corr6),1)    # (1, 209(128+81), 8, 11)
        x = torch.cat((self.conv6_1(x), x),1)   # (1, 337(128+209), 8, 11)
        x = torch.cat((self.conv6_2(x), x),1)   # (1, 433(96+337), 8, 11)
        x = torch.cat((self.conv6_3(x), x),1)   # (1, 497(64+433), 8, 11)
        x = torch.cat((self.conv6_4(x), x),1)   # (1, 529(32+497), 8, 11)
        flow6 = self.predict_flow6(x)   # (1, 2, 8, 11)  预测当前分辨率下的flow
        up_flow6 = self.deconv6(flow6)   # (1, 2, 16, 22)  对flow进行upsample
        up_feat6 = self.upfeat6(x)   # (1, 2, 16, 22)
        # print("level6 " +str(time.time()- start))
        # start=  time.time()
        
        # 根据flow来warp第二幅特征图
        warp5 = self.warp(c25, up_flow6*0.625)   # (1, 128, 16, 22)
        # print("level5_1 " + str(time.time() - start))
        # start5 = time.time()
        corr5 = self.corr(c15, warp5)   # (1, 81, 16, 22)
        # print("level5_2 " + str(time.time() - start5))
        # start5 = time.time()
        corr5 = self.leakyRELU(corr5)   # (1, 81, 16, 22)

        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)   # (1, 213, 16, 22)
        x = torch.cat((self.conv5_0(x), x),1)   # (1, 341(128+213), 16, 22)
        x = torch.cat((self.conv5_1(x), x),1)   # (1, 469(128+341), 16, 22)
        x = torch.cat((self.conv5_2(x), x),1)   # (1, 565(96+469), 16, 22)
        x = torch.cat((self.conv5_3(x), x),1)   # (1, 629(64+565), 16, 22)
        x = torch.cat((self.conv5_4(x), x),1)   # (1, 661(32+629), 16, 22)

        flow5 = self.predict_flow5(x)   # (1, 2, 16, 22)
        up_flow5 = self.deconv5(flow5)  # (1, 2, 32, 44)
        up_feat5 = self.upfeat5(x)   # (1, 2, 32, 44)
        # print("level5_3 " + str(time.time() - start5))
        # print("level5 " + str(time.time() - start))
        # start = time.time()

        warp4 = self.warp(c24, up_flow5*1.25)  # (1, 96, 32, 44)
        corr4 = self.corr(c14, warp4)   # (1, 81, 32, 44)
        corr4 = self.leakyRELU(corr4)   # (1, 81, 32, 44)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)   # (1, 181, 32, 44)
        x = torch.cat((self.conv4_0(x), x),1)   # (1, 309(128+181), 32, 44)
        x = torch.cat((self.conv4_1(x), x),1)   # (1, 437(128+309), 32, 44)
        x = torch.cat((self.conv4_2(x), x),1)   # (1, 533(96+437), 32, 44)
        x = torch.cat((self.conv4_3(x), x),1)   # (1, 597(64+533), 32, 44)
        x = torch.cat((self.conv4_4(x), x),1)   # (1, 629(32+597), 32, 44)
        flow4 = self.predict_flow4(x)   # (1, 2, 32, 44)
        up_flow4 = self.deconv4(flow4)   # (1, 2, 64, 88)
        up_feat4 = self.upfeat4(x)   # (1, 2, 64, 88)

        # print("level4 " + str(time.time() - start))
        # start = time.time()

        warp3 = self.warp(c23, up_flow4*2.5)   # (1, 64, 64, 88)
        corr3 = self.corr(c13, warp3)   # (1, 81, 64, 88)
        corr3 = self.leakyRELU(corr3)   # (1, 81, 64, 88)

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)   # (1, 149, 64, 88)
        x = torch.cat((self.conv3_0(x), x),1)   # (1, 277(128+149), 64, 88)
        x = torch.cat((self.conv3_1(x), x),1)   # (1, 405(128+277), 64, 88)
        x = torch.cat((self.conv3_2(x), x),1)   # (1, 501(96+405), 64, 88)
        x = torch.cat((self.conv3_3(x), x),1)   # (1, 565(64+501), 64, 88)
        x = torch.cat((self.conv3_4(x), x),1)   # (1, 597(32+565), 64, 88)
        flow3 = self.predict_flow3(x)    # (1, 2, 64, 88)
        up_flow3 = self.deconv3(flow3)    # (1, 2, 128, 176)
        up_feat3 = self.upfeat3(x)    # (1, 2, 128, 176)

        # print("level3 " + str(time.time() - start))
        # start = time.time()

        warp2 = self.warp(c22, up_flow3*5.0)   # (1, 32, 128, 176)
        corr2 = self.corr(c12, warp2)    # (1, 81, 128, 176)
        corr2 = self.leakyRELU(corr2)    # (1, 81, 128, 176)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)   # (1, 117, 128, 176)
        x = torch.cat((self.conv2_0(x), x),1)   # (1, 245(128+117), 128, 176)
        x = torch.cat((self.conv2_1(x), x),1)   # (1, 373(128+245), 128, 176)
        x = torch.cat((self.conv2_2(x), x),1)   # (1, 469(96+373), 128, 176)
        x = torch.cat((self.conv2_3(x), x),1)   # (1, 533(64+469), 128, 176)
        x = torch.cat((self.conv2_4(x), x),1)   # (1, 565(32+533), 128, 176)
        flow2 = self.predict_flow2(x)   # (1, 2, 128, 176)
        # print("level2 " + str(time.time() - start))
        # start = time.time()

        # self.dc_conv1(x).shape: (1, 128, 128, 176)
        # self.dc_conv2(self.dc_conv1(x)).shape: (1, 128, 128, 176)
        # self.dc_conv3(self.dc_conv2(self.dc_conv1(x))).shape: (1, 128, 128, 176)
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))   # (1, 96, 128, 176)

        # self.dc_conv5(x).shape: (1, 64, 128, 176)
        # self.dc_conv6(self.dc_conv5(x)).shape: (1, 32, 128, 176)
        # self.dc_conv7(self.dc_conv6(self.dc_conv5(x))).shape: (1, 2, 128, 176)
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        # flow2.shape: (1, 2, 128, 176), flow3.shape: (1, 2, 64, 88), flow4.shape: (1, 2, 32, 44), flow5.shape: (1, 2, 16, 22), flow6.shape: (1, 2, 8, 11),

        # print("refine " + str(time.time() - start))
        # start = time.time()

        # we don't have the gt for flow, we just fine tune it on flownets
        if not output_more:
            return flow2
        else:
            return [flow2,flow3,flow4,flow5,flow6]
        # if self.training:
        #     return flow2,flow3,flow4,flow5,flow6
        # else:
        #     return flow2



class PWCDCNet_old(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """
    def __init__(self, md=4):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNet_old,self).__init__()

        self.conv1a  = conv(3,   16, kernel_size=3, stride=2)
        self.conv1b  = conv(16,  16, kernel_size=3, stride=1)
        self.conv2a  = conv(16,  32, kernel_size=3, stride=2)
        self.conv2b  = conv(32,  32, kernel_size=3, stride=1)
        self.conv3a  = conv(32,  64, kernel_size=3, stride=2)
        self.conv3b  = conv(64,  64, kernel_size=3, stride=1)
        self.conv4a  = conv(64,  96, kernel_size=3, stride=2)
        self.conv4b  = conv(96,  96, kernel_size=3, stride=1)
        self.conv5a  = conv(96, 128, kernel_size=3, stride=2)
        self.conv5b  = conv(128,128, kernel_size=3, stride=1)
        self.conv6a  = conv(128,196, kernel_size=3, stride=2)
        self.conv6b  = conv(196,196, kernel_size=3, stride=1)

        self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*md+1)**2
        dd = np.cumsum([128,128,96,64,32])

        od = nd
        self.conv6_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv6_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv6_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)        
        self.predict_flow6 = predict_flow(od+dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat6 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+128+4
        self.conv5_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv5_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv5_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od+dd[4]) 
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat5 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+96+4
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od+dd[4]) 
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+64+4
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od+dd[4]) 
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 2, kernel_size=4, stride=2, padding=1) 
        
        od = nd+32+4
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od+dd[4]) 
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1) 
        
        self.dc_conv1 = conv(od+dd[4], 128, kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv2 = conv(128,      128, kernel_size=3, stride=1, padding=2,  dilation=2)
        self.dc_conv3 = conv(128,      128, kernel_size=3, stride=1, padding=4,  dilation=4)
        self.dc_conv4 = conv(128,      96,  kernel_size=3, stride=1, padding=8,  dilation=8)
        self.dc_conv5 = conv(96,       64,  kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64,       32,  kernel_size=3, stride=1, padding=1,  dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)
        
        mask[mask<0.999] = 0
        mask[mask>0] = 1
        
        return output*mask


    def forward(self,x):
        im1 = x[:,:3,:,:]
        im2 = x[:,3:,:,:]
        
        c11 = self.conv1b(self.conv1a(im1))
        c21 = self.conv1b(self.conv1a(im2))
        c12 = self.conv2b(self.conv2a(c11))
        c22 = self.conv2b(self.conv2a(c21))
        c13 = self.conv3b(self.conv3a(c12))
        c23 = self.conv3b(self.conv3a(c22))
        c14 = self.conv4b(self.conv4a(c13))
        c24 = self.conv4b(self.conv4a(c23))        
        c15 = self.conv5b(self.conv5a(c14))
        c25 = self.conv5b(self.conv5a(c24))
        c16 = self.conv6b(self.conv6a(c15))
        c26 = self.conv6b(self.conv6a(c25))
        
        corr6 = self.corr(c16, c26) 
        corr6 = self.leakyRELU(corr6)        
        x = torch.cat((corr6, self.conv6_0(corr6)),1)
        x = torch.cat((self.conv6_1(x), x),1)
        x = torch.cat((x, self.conv6_2(x)),1)
        x = torch.cat((x, self.conv6_3(x)),1)
        x = torch.cat((x, self.conv6_4(x)),1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)
        
        warp5 = self.warp(c25, up_flow6*0.625)
        corr5 = self.corr(c15, warp5) 
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((x, self.conv5_0(x)),1)
        x = torch.cat((self.conv5_1(x), x),1)
        x = torch.cat((x, self.conv5_2(x)),1)
        x = torch.cat((x, self.conv5_3(x)),1)
        x = torch.cat((x, self.conv5_4(x)),1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)
        
        warp4 = self.warp(c24, up_flow5*1.25)
        corr4 = self.corr(c14, warp4)  
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((x, self.conv4_0(x)),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((x, self.conv4_2(x)),1)
        x = torch.cat((x, self.conv4_3(x)),1)
        x = torch.cat((x, self.conv4_4(x)),1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4*2.5)
        corr3 = self.corr(c13, warp3) 
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((x, self.conv3_0(x)),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((x, self.conv3_2(x)),1)
        x = torch.cat((x, self.conv3_3(x)),1)
        x = torch.cat((x, self.conv3_4(x)),1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)
        
        warp2 = self.warp(c22, up_flow3*5.0) 
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((x, self.conv2_0(x)),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((x, self.conv2_2(x)),1)
        x = torch.cat((x, self.conv2_3(x)),1)
        x = torch.cat((x, self.conv2_4(x)),1)
        flow2 = self.predict_flow2(x)
 
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        
        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2





def pwc_dc_net(path=None):

    model = PWCDCNet()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model




def pwc_dc_net_old(path=None):

    model = PWCDCNet_old()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model
