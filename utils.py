import h5py
import scipy.io as sio
import os
import numpy as np
import torch
import logging
import random
from ssim_torch import ssim
import cv2
import glob
import re 
def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def TVloss(x,batch_size,weight=0.001):
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h =  (x.size()[2]-1) * x.size()[3]
    count_w = x.size()[2] * (x.size()[3] - 1)
    h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
    return weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def normalize_image(image, min_value=0, max_value=1):
    min_val = np.min(image)
    max_val = np.max(image)

    normalized_image = (image - min_val) / (max_val - min_val) * (max_value - min_value) + min_value
    
    return normalized_image

def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    scene_list.sort()
    #target_size=[28,1392,1300]
    print('training sences:', len(scene_list))
    #for i in range(len(scene_list)):
    for i in range(5):
        scene_path = path + scene_list[i]
        img = h5py.File(scene_path)['rad'][:]
        img = img.astype(np.float32)
        img = normalize_image(img)
        c,h,w = img.shape
        if w < 1300:
            pad_width = 1300 - w
            img = np.pad(img, ((0, 0), (0, 0), (0, pad_width)), mode='constant', constant_values=0)
            print(img.shape)
            print('Sence {} is miss. {}'.format(i, scene_list[i]))
        imgs.append(img[0:28,:,:])
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))
    imgs = np.array(imgs)
    imgs = torch.from_numpy(imgs)
    return imgs
def arguement_1(x):
    """
    :param x: c,h,w
    :return: c,h,w
    """
    rotTimes = random.randint(0, 3)
    vFlip = random.randint(0, 1)
    hFlip = random.randint(0, 1)
    # Random rotation
    for j in range(rotTimes):
        x = torch.rot90(x, dims=(1, 2))
    # Random vertical Flip
    for j in range(vFlip):
        x = torch.flip(x, dims=(2,))
    # Random horizontal Flip
    for j in range(hFlip):
        x = torch.flip(x, dims=(1,))
    return x

def arguement_2(generate_gt):
    c, h, w = generate_gt.shape[1],256,256
    divid_point_h = 128
    divid_point_w = 128
    output_img = torch.zeros(c,h,w).cuda()
    output_img[:, :divid_point_h, :divid_point_w] = generate_gt[0]
    output_img[:, :divid_point_h, divid_point_w:] = generate_gt[1]
    output_img[:, divid_point_h:, :divid_point_w] = generate_gt[2]
    output_img[:, divid_point_h:, divid_point_w:] = generate_gt[3]
    return output_img


def shuffle_crop_icvl(train_data, batch_size, crop_size=256, argument=True): #(n,28,1304, 1392)
    if argument:
        gt_batch = []
        # The first half data use the original data.
        index = np.random.choice(range(len(train_data)), batch_size//2)
        processed_data = np.zeros((batch_size//2, 28 ,crop_size, crop_size), dtype=np.float32)
        for i in range(batch_size//2):
            img = train_data[index[i]]
            _,h, w = img.shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = img[:,x_index:x_index + crop_size, y_index:y_index + crop_size]
        processed_data = torch.from_numpy(processed_data).cuda().float()
        for i in range(processed_data.shape[0]):
            gt_batch.append(arguement_1(processed_data[i]))

        # The other half data use splicing.
        processed_data = np.zeros((4, 28, 128, 128), dtype=np.float32)
        for i in range(batch_size - batch_size // 2):
            sample_list = np.random.randint(0, len(train_data), 4)
            for j in range(4):
                x_index = np.random.randint(0, h-crop_size//2)
                y_index = np.random.randint(0, w-crop_size//2)
                processed_data[j] = train_data[sample_list[j]][:,x_index:x_index+crop_size//2,y_index:y_index+crop_size//2]
            gt_batch_2 = torch.from_numpy(processed_data).cuda()  # [4,28,128,128]
            gt_batch.append(arguement_2(gt_batch_2))
        gt_batch = torch.stack(gt_batch, dim=0)
        return gt_batch
    else:
        index = np.random.choice(range(len(train_data)), batch_size)
        processed_data = np.zeros((batch_size,28, crop_size, crop_size), dtype=np.float32)
        for i in range(batch_size):
            _, h, w = train_data[index[i]].shape
            x_index = np.random.randint(0, h - crop_size)
            y_index = np.random.randint(0, w - crop_size)
            processed_data[i, :, :, :] = train_data[index[i]][:,x_index:x_index + crop_size, y_index:y_index + crop_size]
        gt_batch = torch.from_numpy(processed_data)
        return gt_batch

# We find that this calculation method is more close to DGSMP's.
def torch_psnr(img, ref):  # input [28,256,256]
    img = (img*256).round()
    ref = (ref*256).round() 
    nC = img.shape[0]
    psnr = 0
    for i in range(nC):
        mse = torch.mean((img[i, :, :] - ref[i, :, :]) ** 2)
        psnr += 10 * torch.log10((255*255)/mse)
    return psnr / nC

def torch_ssim(img, ref):  # input [28,256,256]
    return ssim(torch.unsqueeze(img, 0), torch.unsqueeze(ref, 0))

def gen_meas_torch(data_batch, mask3d_batch, Y2H=True, mul_mask=False):
    nC = data_batch.shape[1]
    temp = shift(mask3d_batch * data_batch, 2)
    meas = torch.sum(temp, 1)
    if Y2H:
        meas = meas / nC * 2
        H = shift_back(meas)
        if mul_mask:
            HM = torch.mul(H, mask3d_batch)
            return HM
        return H
    return meas

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output

def shift_back(inputs, step=2):  # input [bs,256,310]  output [bs, 28, 256, 256]
    [bs, row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col - (nC - 1) * step).cuda().float()
    for i in range(nC):
        output[:, i, :, :] = inputs[:, :, step * i:step * i + col - (nC - 1) * step]
    return output

def init_mask(mask_path, mask_type, batch_size):
    mask3d_batch = generate_masks(mask_path, batch_size)
    if mask_type == 'Phi':
        shift_mask3d_batch = shift(mask3d_batch)
        input_mask = shift_mask3d_batch
    elif mask_type == 'Phi_PhiPhiT':
        Phi_batch, Phi_s_batch = generate_shift_masks(mask_path, batch_size)
        input_mask = (Phi_batch, Phi_s_batch)
    elif mask_type == 'Mask':
        input_mask = mask3d_batch
    elif mask_type == None:
        input_mask = None
    return mask3d_batch, input_mask

def init_meas(gt, mask, input_setting):
    if input_setting == 'H':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=False)
    elif input_setting == 'HM':
        input_meas = gen_meas_torch(gt, mask, Y2H=True, mul_mask=True)
    elif input_setting == 'Y':
        input_meas = gen_meas_torch(gt, mask, Y2H=False, mul_mask=True)
    return input_meas

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs


def generate_masks(mask_path, batch_size):
    mask = sio.loadmat(mask_path + '/mask.mat')
    mask = mask['mask']
    mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask3d = np.transpose(mask3d, [2, 0, 1])
    mask3d = torch.from_numpy(mask3d)
    [nC, H, W] = mask3d.shape
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    return mask3d_batch

def generate_shift_masks(mask_path, batch_size):
    mask256 = sio.loadmat(mask_path)['mask']
    r, c, nC, step = 256,256,28,2
    mask=np.zeros((r, c + step * (nC - 1)))
    mask_3d_shift = np.tile(mask[:, :, np.newaxis], (1, 1, nC))
    for i in range(nC):
        mask_3d_shift[:, i:i+256, i]=mask256
    mask_3d_shift = np.transpose(mask_3d_shift, [2, 0, 1])
    mask_3d_shift = torch.from_numpy(mask_3d_shift)
    [nC, H, W] = mask_3d_shift.shape
    Phi_batch = mask_3d_shift.expand([batch_size, nC, H, W]).cuda().float()
    Phi_s_batch = torch.sum(Phi_batch**2,1)
    Phi_s_batch[Phi_s_batch==0] = 1
    return Phi_batch, Phi_s_batch


def transform(inputs,index,wavenum,x=0,y=0):#[1,28,256,256]
    inputs=inputs.cpu().detach().numpy()
    inputs=inputs.squeeze()
    outputs=np.zeros(inputs.shape)
    if index==0:#Horizontal
        for i in range(wavenum):
            outputs[i,:,:]=cv2.flip(inputs[i,:,:],1,dst=None)
    if index==1:#Vertical
        for i in range(wavenum):
            outputs[i,:,:]=cv2.flip(inputs[i,:,:],0,dst=None)
    if index==2:#Rotate angle
        w = inputs.shape[2]
        h = inputs.shape[1]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w//2,h//2), 45, 1)
        #rotate
        for i in range(wavenum):
            outputs[i,:,:]=cv2.warpAffine(inputs[i,:,:],M,(w,h))
    if index==3:#Rotate angle
        w = inputs.shape[2]
        h = inputs.shape[1]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w//2,h//2), 90, 1)
        #rotate
        for i in range(wavenum):
            outputs[i,:,:]=cv2.warpAffine(inputs[i,:,:],M,(w,h))
    if index==4:#Rotate angle
        w = inputs.shape[2]
        h = inputs.shape[1]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w//2,h//2), 180, 1)
        #rotate
        for i in range(wavenum):
            outputs[i,:,:]=cv2.warpAffine(inputs[i,:,:],M,(w,h))
    if index==5:#Rotate angle
        w = inputs.shape[2]
        h = inputs.shape[1]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w//2,h//2), 270, 1)
        #rotate
        for i in range(wavenum):
            outputs[i,:,:]=cv2.warpAffine(inputs[i,:,:],M,(w,h))
    if index==6:
        img_info=inputs.shape
        height=img_info[1]
        width=img_info[2]
        mat_translation=np.float32([[1,0,x],[0,1,y]])  
        for i in range(wavenum):
            outputs[i,:,:]=cv2.warpAffine(inputs[i,:,:],mat_translation,(width,height))  
    if index==7:
        for i in range(wavenum):
            outputs[i,:,:]=cv2.flip(inputs[i,:,:],-1,dst=None)
    if index==8:#45
        w = inputs.shape[2]
        h = inputs.shape[1]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w//2,h//2), 45, 1)
        #rotate
        for i in range(wavenum):
            outputs[i,:,:]=cv2.warpAffine(inputs[i,:,:],M,(w,h))
    if index==9:#135
        w = inputs.shape[2]
        h = inputs.shape[1]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w//2,h//2), 135, 1)
        #rotate
        for i in range(wavenum):
            outputs[i,:,:]=cv2.warpAffine(inputs[i,:,:],M,(w,h))
    if index==10:#225
        w = inputs.shape[2]
        h = inputs.shape[1]
        #rotate matrix
        M = cv2.getRotationMatrix2D((w//2,h//2), 225, 1)
        #rotate
        for i in range(wavenum):
            outputs[i,:,:]=cv2.warpAffine(inputs[i,:,:],M,(w,h))
    outputs=torch.from_numpy(outputs)
    outputs=torch.unsqueeze(outputs,0)
    return outputs

def checkpoint(model, epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))

def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

#choose 5 patch 
def LoadTest(path_test): 
    imgs = []
    scene_path1 = path_test + "10.mat"
    img = h5py.File(scene_path1)['rad'][:]
    img = img.astype(np.float32)
    img = normalize_image(img)
    imgs.append(img[0:28,800:800+256,500:500+256])
    scene_path2 = path_test + "168.mat"
    img = h5py.File(scene_path2)['rad'][:]
    img = img.astype(np.float32)
    img = normalize_image(img)
    imgs.append(img[0:28,300:300+256,700:700+256])
    scene_path3 = path_test + "90.mat"
    img = h5py.File(scene_path3)['rad'][:]
    img = img.astype(np.float32)
    img = normalize_image(img)
    imgs.append(img[0:28,700:700+256,200:200+256])
    scene_path4 = path_test + "130.mat"
    img = h5py.File(scene_path4)['rad'][:]
    img = img.astype(np.float32)
    img = normalize_image(img)
    imgs.append(img[0:28,256:256+256,735:735+256])
    scene_path5 = path_test + "190.mat"
    img = h5py.File(scene_path5)['rad'][:]
    img = img.astype(np.float32)
    img = normalize_image(img)
    imgs.append(img[0:28,1100:1100+256,0:0+256])
    imgs = np.array(imgs)
    imgs = torch.from_numpy(imgs)
    return imgs