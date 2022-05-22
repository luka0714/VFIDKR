import time
import os
from torch.autograd import Variable
import math
import torch

import random
import numpy as np
import numpy
import networks
from my_args import  args
from torchvision import transforms

import torch.nn.functional as F

from scipy.misc import imread, imsave
from AverageMeter import  *
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True # to speed up the


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)

def ssim(
    X,
    Y,
    data_range=1,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu
    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    # for d in range(len(X.shape) - 1, 1, -1):
    #     X = np.squeeze(X, d)
    #     Y = np.squeeze(Y, d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)

def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out

def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


DO_MiddleBurryOther = True
# MB_Other_DATA = "./MiddleBurySet/other-color-allframes/other-data/"
# MB_Other_RESULT = "./MiddleBurySet/other-result-author/"
# MB_Other_GT = "./MiddleBurySet/other-gt-interp/"

MB_Other_DATA = "./triplets_UCF/"
MB_Other_RESULT = "./triplets_UCF_other/"
MB_Other_GT = "./triplets_UCF/"

# MB_Other_DATA = "./eval-color-twoframes/eval-data/"
# MB_Other_RESULT = "./eval_output/"
# MB_Other_GT = "./MiddleBurySet/other-gt-interp/"

if not os.path.exists(MB_Other_RESULT):
    os.mkdir(MB_Other_RESULT)


# 初始化整个网络
# args.netName: DAIN    args.channels: 3     args.filter_size: 4     args.time_step: 0.5 
model = networks.__dict__[args.netName](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=False)


if args.use_cuda:
    model = model.cuda()

# args.SAVED_MODEL = './model_weights/Deformable2.0_occ_50epoch.pth'
# args.SAVED_MODEL = './model_weights/MEMC-epoch50-1batch/best.pth'
# args.SAVED_MODEL = './model_weights/MEMC+Deforconv2.0-epoch50-1batch/best.pth'
# args.SAVED_MODEL = './model_weights/MEMC+Deforconv2.0-modLoss-epoch35-3batch-nocur_output/best.pth'
# args.SAVED_MODEL = './model_weights/MEMC+Deforconv2.0-new/best.pth'
args.SAVED_MODEL = './model_weights/ablation_nopost/best.pth'


if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")

model = model.eval() # deploy mode



# totalparams = 0
# for param in model.parameters():
#     mulvalue = np.prod(param.size())
#     totalparams += mulvalue

# print(totalparams)


use_cuda=args.use_cuda
save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))

interp_error = AverageMeter()
if DO_MiddleBurryOther:
    subdir = os.listdir(MB_Other_DATA)
    gen_dir = os.path.join(MB_Other_RESULT, unique_id)
    os.mkdir(gen_dir)

    avgpsnr = 0
    avgssim = 0

    tot_timer = AverageMeter()
    proc_timer = AverageMeter()
    end = time.time()
    for dir in subdir:
        print(dir)
        os.mkdir(os.path.join(gen_dir, dir))
        # arguments_strFirst = os.path.join(MB_Other_DATA, dir, "frame10.png")
        # arguments_strSecond = os.path.join(MB_Other_DATA, dir, "frame11.png")
        # arguments_strOut = os.path.join(gen_dir, dir, "frame10i11.png")
        # gt_path = os.path.join(MB_Other_GT, dir, "frame10i11.png")

        # HD
        # arguments_strFirst = os.path.join(MB_Other_DATA, dir, "img1.png")
        # arguments_strSecond = os.path.join(MB_Other_DATA, dir, "img3.png")
        # arguments_strOut = os.path.join(gen_dir, dir, "out-img2.png")
        # gt_path = os.path.join(MB_Other_GT, dir, "img2.png")

        # UCF
        arguments_strFirst = os.path.join(MB_Other_DATA, dir, "im2.png")
        arguments_strSecond = os.path.join(MB_Other_DATA, dir, "im4.png")
        arguments_strOut = os.path.join(gen_dir, dir, "out-im3.png")
        gt_path = os.path.join(MB_Other_GT, dir, "im3.png")

        # davis
        # arguments_strFirst = os.path.join(MB_Other_DATA, dir, "00001.jpg")
        # arguments_strSecond = os.path.join(MB_Other_DATA, dir, "00003.jpg")
        # arguments_strOut = os.path.join(gen_dir, dir, "out-00002.jpg")
        # gt_path = os.path.join(MB_Other_GT, dir, "00002.jpg")

        X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype)  # torch.Size([3, 480, 640])
        X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)  # torch.Size([3, 480, 640])

        y_ = torch.FloatTensor()

        assert (X0.size(1) == X1.size(1))
        assert (X0.size(2) == X1.size(2))

        intWidth = X0.size(2)
        intHeight = X0.size(1)
        channel = X0.size(0)
        if not channel == 3:
            continue

        if intWidth != ((intWidth >> 7) << 7):
            intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
            intPaddingLeft =int(( intWidth_pad - intWidth)/2)
            intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
        else:
            intWidth_pad = intWidth
            intPaddingLeft = 32
            intPaddingRight= 32

        if intHeight != ((intHeight >> 7) << 7):
            intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
            intPaddingTop = int((intHeight_pad - intHeight) / 2)
            intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
        else:
            intHeight_pad = intHeight
            intPaddingTop = 32
            intPaddingBottom = 32

        pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

        torch.set_grad_enabled(False)
        X0 = Variable(torch.unsqueeze(X0,0))
        X1 = Variable(torch.unsqueeze(X1,0))
        X0 = pader(X0)    # (1, 3, 512, 704)   pad之后的x0，x1
        X1 = pader(X1)    # (1, 3, 512, 704)

        if use_cuda:
            X0 = X0.cuda()
            X1 = X1.cuda()
        proc_end = time.time()
        #  torch.stack((X0, X1),dim = 0).shape: torch.Size([2, 1, 3, 512, 704])
        y_s,offset,filter = model(torch.stack((X0, X1),dim = 0))
        # y_s[0], y_s[1].shape: (1, 3, 512, 704)   y_s[0]为两个warp图像的均值， y_s[1]为增强后的输出图像
        # offset[0], offset[1].shape: (1, 2, 512, 704)     filter[0], filter[1].shape: (1, 16, 512, 704)

        y_ = y_s[save_which]   # default:1 即选择增强后的输出图像

        proc_timer.update(time.time() -proc_end)
        tot_timer.update(time.time() - end)
        end  = time.time()
        # print("*****************current image process time \t " + str(time.time()-proc_end )+"s ******************" )
        if use_cuda:
            X0 = X0.data.cpu().numpy()   # (1, 3, 512, 704)  range[0, 1]   第一个参考帧
            y_ = y_.data.cpu().numpy()   # (1, 3, 512, 704)  range[0, 1]   网络得到的增强帧
            offset = [offset_i.data.cpu().numpy() for offset_i in offset]
            filter = [filter_i.data.cpu().numpy() for filter_i in filter]  if filter[0] is not None else None
            X1 = X1.data.cpu().numpy()   # (1, 3, 512, 704)  range[0, 1]   第二个参考帧
        else:
            X0 = X0.data.numpy()
            y_ = y_.data.numpy()
            offset = [offset_i.data.numpy() for offset_i in offset]
            filter = [filter_i.data.numpy() for filter_i in filter]
            X1 = X1.data.numpy()


        # 640*480原始图像大小 (480(height), 640(width), 3)  range[0, 255]
        X0 = np.transpose(255.0 * X0.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        
        # (480, 640, 3)   range[0, 255]
        y_ = np.transpose(255.0 * y_.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
        offset = [np.transpose(offset_i[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0)) for offset_i in offset]
        filter = [np.transpose(
            filter_i[0, :, intPaddingTop:intPaddingTop + intHeight, intPaddingLeft: intPaddingLeft + intWidth],
            (1, 2, 0)) for filter_i in filter]  if filter is not None else None

        # (480, 640, 3)   range[0, 255]
        X1 = np.transpose(255.0 * X1.clip(0,1.0)[0, :, intPaddingTop:intPaddingTop+intHeight, intPaddingLeft: intPaddingLeft+intWidth], (1, 2, 0))
       

        # 将网络输出增强后的图像存到arguments_strOut路径下
        imsave(arguments_strOut, np.round(y_).astype(numpy.uint8))


        rec_rgb =  imread(arguments_strOut)
        gt_rgb = imread(gt_path)

        diff_rgb = 128.0 + rec_rgb - gt_rgb
        avg_interp_error_abs = np.mean(np.abs(diff_rgb - 128.0))

        interp_error.update(avg_interp_error_abs, 1)

        mse = numpy.mean((diff_rgb - 128.0) ** 2)

        PIXEL_MAX = 255.0
        psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

        avgpsnr += psnr

        # ssim
        tensor_rec = transforms.ToTensor()(rec_rgb)
        tensor_rec = tensor_rec.unsqueeze(1)
        tensor_gt = transforms.ToTensor()(gt_rgb)
        tensor_gt = tensor_gt.unsqueeze(1)
        imgssim = ssim(tensor_rec, tensor_gt).item()
        avgssim += imgssim

        # print("interpolation error / PSNR  / SSIM : " + str(round(avg_interp_error_abs,4)) + " / " + str(round(psnr,4)) + " / " + str(round(imgssim,4)))


    metrics = "The average interpolation error / PSNR for all images are : " + str(round(interp_error.avg, 4))
    print(metrics)

    print("avgpsnr: " + str(round(avgpsnr/len(subdir), 4)))
    print("avgssim: " + str(round(avgssim/len(subdir), 4)))
    

