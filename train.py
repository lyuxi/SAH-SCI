#Load pre-trained model: HDNET (example)
from argparse import ArgumentParser, Namespace
from utils import *
import torch
import scipy.io as scio
import time
import os
import numpy as np
from torch.autograd import Variable
import datetime
import torch.nn.functional as F
from architecture.HDNet import HDNet
from SAH import *

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default=0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--dataset_path", type=str, default ="./datasets/ICVL/")
    parser.add_argument("--mask_path", type=str, default ="./datasets/mask/")
    parser.add_argument("--checkpoint_path", type=str, help= "Load pretrained SAH",default =None)
    parser.add_argument("--pretrained_path", type=str, default ="./model_zoo/hdnet/hdnet.pth")
    parser.add_argument("--model_path", type=str, help= "Save SAH model", default ="./checkpoint")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epoch_sam_num", type=int, default=1000)
    parser.add_argument("--epoch_num", type=int, default=500)
    return parser.parse_args()

def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #saving path#
    date_time = str(datetime.datetime.now())
    date_time = time2file_name(date_time)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # init mask
    print("------------load mask------------")
    input_mask = None
    mask3d_batch_train, input_mask_train = init_mask(args.mask_path, input_mask, args.batch_size)
    mask3d_batch_train = mask3d_batch_train.cuda()
    # dataset
    print("------------load data------------")
    train_set = LoadTraining(args.dataset_path)
    #model
    print("------------load model------------")
    model = SAH(28,28).cuda()
    if args.checkpoint_path != None:
        model.load_state_dict(torch.load(args.checkpoint_path))
    hdnet = HDNet().cuda()
    hdnet.load_state_dict(torch.load(args.pretrained_path))

    # optimizing
    learning_rate = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999),eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250,350,450], gamma=0.5)  
    mse = torch.nn.MSELoss().cuda()
    initial_epoch = 0
    print("------------start training------------")
    logger = gen_log(args.model_path)
    logger.info("beta rate:{}, alpha:{}.\n".format(args.beta, args.alpha))
    for epoch in range(initial_epoch,args.epoch_num):
        logger.info("epoch:{}\n".format(epoch))
        model.train()
        epoch_loss = epoch_loss_1 = epoch_loss_2 = epoch_loss_3 = epoch_loss_4 =  0
        begin = time.time()
        batch_num = int(np.floor(args.epoch_sam_num / args.batch_size))
        for i in range(batch_num):
            gt_batch = shuffle_crop_icvl(train_set, args.batch_size)
            gt = Variable(gt_batch).cuda().float()
            input_meas_loss = init_meas(gt, mask3d_batch_train, 'Y')
            input_meas = init_meas(gt, mask3d_batch_train, 'H')
            optimizer.zero_grad()
            with torch.no_grad():
                model_input = hdnet(input_meas,input_mask_train)
            model_out = model(model_input)
            output_meas = init_meas(model_out, mask3d_batch_train, 'Y')
            "----first update-----"
            loss1 = torch.sqrt(mse(output_meas,input_meas_loss))
            model_out_trans = torch.zeros_like(model_out)
            for j in range (0,args.batch_size):
                x1 = y1 = 0
                index = np.random.randint(0,11)
                if index == 6:
                    x1 = np.random.randint(-50,50)
                    y1 = np.random.randint(-50,50)
                model_out_trans[j,:,:,:] = transform(model_out[j,:,:,:],index,28,x1,y1).cuda().to(torch.float32) #Tf(y)
            mask3d_batch_train_trans, _ = init_mask(args.mask_path, input_mask, args.batch_size)
            input_trans_meas = init_meas(model_out_trans, mask3d_batch_train_trans, 'H').cuda()
            with torch.no_grad():
                model_out_trans_in = hdnet(input_trans_meas,input_mask_train)
            model_out_trans_out = model(model_out_trans_in)
            "----second update-----"
            loss2 = torch.sqrt(mse(model_out_trans_out, model_out_trans))
            "----third update-----"
            z = model_out 
            PhiNx = init_meas(z, mask3d_batch_train, 'H')
            with torch.no_grad():
                PhiNx_in = hdnet(PhiNx,input_mask_train)
            z_prime = model(PhiNx_in)
            "-----fourth update-----"
            res = (z_prime-z) 
            mask = torch.ones_like(res)
            mask = F.dropout(mask, 0.5) * 0.5
            mask = torch.where(mask == 0, -1 * mask, mask)
            r = (mask * res.detach())
            PhiNx = init_meas(model_out + r, mask3d_batch_train, 'H')
            with torch.no_grad():
                PhiNx_in = hdnet(PhiNx,input_mask_train)
            x_output_r = model(PhiNx_in)
            loss3 = torch.sqrt(mse(x_output_r,(model_out - r)))
            losstv = TVloss(model_out,args.batch_size)
            loss = loss1 + args.alpha * loss2 +  args.beta * loss3 + losstv
            epoch_loss += loss.data
            epoch_loss_1 += loss1.data
            epoch_loss_2 += args.alpha *loss2.data
            epoch_loss_3 += args.beta * loss3.data
            epoch_loss_4 += losstv.data
            loss.backward()
            optimizer.step()
        end = time.time()
        logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} ,x_loss:{:.6f},ei_loss :{:.6f},res_loss:{:.6f}, tv_loss:{:.6f} time: {:.2f}".
                    format(epoch, epoch_loss / batch_num,epoch_loss_1/batch_num, epoch_loss_2 / batch_num, epoch_loss_3 / batch_num,epoch_loss_4 / batch_num,(end - begin)))
        scheduler.step()
        if epoch > 50 and epoch % 30 == 0:
            checkpoint(model, epoch, args.model_path, logger)


if __name__ == '__main__':
    args = parse_args()
    main(args)
