from utils import *
import torch
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
import torch.nn.functional as F
from SAH import *
from utils import *
from argparse import ArgumentParser, Namespace
from architecture.HDNet import *

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default=0)
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--result_path", type=str, default="./results/")
    parser.add_argument("--test_num", type=int, default=5)
    parser.add_argument("--dataset_path",type=str, default = "./datasets/ICVL/")
    parser.add_argument("--pretrained_path", type=str, default ="./model_zoo/hdnet/hdnet.pth")
    parser.add_argument("--mask_path", type=str, default ="./datasets/mask/")
    return parser.parse_args()


args = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
#saving path#
date_time = str(datetime.datetime.now())
date_time = time2file_name(date_time)
# init mask
print("------------load mask------------")
input_mask = None
mask3d_batch_test, input_mask_test = init_mask(args.mask_path, input_mask, args.test_num)
# You can change test_data to your own test data
print("------------load test data------------")
test_data = LoadTest(args.dataset_path)
test_gt = Variable(test_data).cuda().float()

#model
print("------------load model------------")
model = SAH(28,28).cuda()
model.load_state_dict(torch.load(args.checkpoint_path))
hdnet = HDNet().cuda()
hdnet.load_state_dict(torch.load(args.pretrained_path))

print("------------without fine_tune------------")
psnr_list, ssim_list = [], []
model.eval()
with torch.no_grad():
    input_test = init_meas(test_gt, mask3d_batch_test, 'H')
    model_out = hdnet(input_test,input_mask_test)
for k in range(test_gt.shape[0]):
    psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
    ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
    psnr_list.append(psnr_val.detach().cpu().numpy())
    ssim_list.append(ssim_val.detach().cpu().numpy())
pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
psnr_mean = np.mean(np.asarray(psnr_list))
ssim_mean = np.mean(np.asarray(ssim_list))
for i in range(args.test_num):
    print(str(i+1)+":"+"psnr:"+str(psnr_list[i])+",ssim:"+str(ssim_list[i]))
print("mean:"+str(psnr_mean)+"/"+str(ssim_mean))
name = args.result_path + str(psnr_mean)+"_without.mat"
scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_mean': psnr_mean, 'ssim_mean': ssim_mean})

print("------------with fine_tune------------")
psnr_list, ssim_list = [], []
model.eval()
with torch.no_grad():
    input_test = init_meas(test_gt, mask3d_batch_test, 'H')
    model_in = hdnet(input_test,input_mask_test)
    model_out = model(model_in)
for k in range(test_gt.shape[0]):
    psnr_val = torch_psnr(model_out[k, :, :, :], test_gt[k, :, :, :])
    ssim_val = torch_ssim(model_out[k, :, :, :], test_gt[k, :, :, :])
    psnr_list.append(psnr_val.detach().cpu().numpy())
    ssim_list.append(ssim_val.detach().cpu().numpy())
pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
psnr_mean = np.mean(np.asarray(psnr_list))
ssim_mean = np.mean(np.asarray(ssim_list))
for i in range(args.test_num):
  print(str(i+1)+":"+"psnr:"+str(psnr_list[i])+",ssim:"+str(ssim_list[i]))
print("mean:"+str(psnr_mean)+"/"+str(ssim_mean))
name = args.result_path + str(psnr_mean)+"_with.mat"
scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_mean': psnr_mean, 'ssim_mean': ssim_mean})

