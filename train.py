from ast import arg
import torch
import torch.nn as nn               # Linear
import torch.nn.functional as F     # relu, softmax
import torch.optim as optim         # Adam Optimizer
from torch.distributions import Categorical # Categorical import from torch.distributions module
import torch.multiprocessing as mp # multi processing
import time 
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt ###for plot
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random
import os

from torchinfo import summary

import models
import utils
import data_load
#input sample of size 69 × 240
#latent space 3 × 8 × 256 tensor

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--model_type', type=str, default='AE') 
parser.add_argument('--datasetPath', type=str, default='/input/MotionInfillingData/train_data')
parser.add_argument('--ValdatasetPath', type=str, default='/input/MotionInfillingData/valid_data')
parser.add_argument('--saveDir', type=str, default='/personal/GiHoonKim/reproduce_infilling')
parser.add_argument('--gpu', type=str, default='0', help='gpu')
parser.add_argument('--numEpoch', type=int, default=200, help='input batch size for training')
parser.add_argument('--batchSize', type=int, default=80, help='input batch size for training')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
args = parser.parse_args()


def main(args):
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    saveUtils = utils.saveData(args)
    
    writer = SummaryWriter(saveUtils.save_dir_tensorBoard)
    
    
    if args.model_type == 'VAE':
        model = models.Convolutional_VAE().to(device)
    else:
        model = models.Convolutional_AE().to(device)
    
    saveUtils.save_log(str(args))
    saveUtils.save_log(str(summary(model, (1,1,69,240))))
    
    train_dataloader, train_dataset = data_load.get_dataloader(args.datasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=True, dataset_mean=None, dataset_std=None)
    valid_dataloader, valid_dataset = data_load.get_dataloader(args.ValdatasetPath , args.batchSize, IsNoise=False, \
                                                                            IsTrain=False, dataset_mean=None, dataset_std=None)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.L1Loss()
    
    print_interval = 100
    print_num = 0
    for num_epoch in range(args.numEpoch):
        
        total_loss = 0
        total_root_loss = 0

        total_v_loss = 0
        total_root_v_loss = 0
        
        if train_dataset.masking_length_mean < 120 and num_epoch is not 0 and num_epoch%10 == 0:
            train_dataset.masking_length_mean = train_dataset.masking_length_mean + 10
            valid_dataset.masking_length_mean = train_dataset.masking_length_mean
            train_dataloader = DataLoader(train_dataset, batch_size=args.batchSize, shuffle=True, drop_last=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=args.batchSize, shuffle=True, drop_last=True)
            
            log = "Current train_dataset.masking_length_mean: %d" % train_dataset.masking_length_mean
            print(log)
            saveUtils.save_log(log)
            
        for iter, item in enumerate(train_dataloader):
            print_num +=1
            
            masked_input, gt_image = item
            masked_input = masked_input.to(device, dtype=torch.float)
            gt_image = gt_image.to(device, dtype=torch.float)
            
            pred = model(masked_input)
            
            train_loss = loss_function(pred, gt_image)

            #total_loss += train_loss.item()
            #train_loss_root = loss_function(pred[:, :, -1, :], gt_image[:, :, -1, :]) + loss_function(pred[:, :, -2, :], gt_image[:, :, -2, :]) +loss_function(pred[:, :, -3, :], gt_image[:, :, -3, :])
            train_loss_root = loss_function(pred[:, :, -1, :], gt_image[:, :, -1, :])
            total_train_loss = train_loss + train_loss_root * 5

            total_root_loss += train_loss_root.item() * 5
            total_loss += total_train_loss.item()
            
            optimizer.zero_grad()
            total_train_loss.backward()
            #train_loss.backward()
            optimizer.step()
            
            if iter % print_interval == 0 and iter != 0:
                train_iter_loss =  total_loss*0.01
                train_root_iter_loss = total_root_loss*0.01
                log = "Train: [Epoch %d][Iter %d] [Train Loss: %.4f] [Train Root Loss %.4f]" % (num_epoch, iter, train_iter_loss, train_root_iter_loss)
                print(log)
                saveUtils.save_log(log)
                writer.add_scalar("Train Loss/ iter", train_iter_loss, print_num)
                writer.add_scalar("Train Root Loss/ iter", train_root_iter_loss, print_num)
                total_loss = 0
                
        #validation per epoch ############
        for iter, item in enumerate(valid_dataloader):
            model.eval()
            masked_input, gt_image = item
            masked_input = masked_input.to(device, dtype=torch.float)
            gt_image = gt_image.to(device, dtype=torch.float)
            
            with torch.no_grad():
                pred = model(masked_input)

            val_loss = loss_function(pred, gt_image.detach())
            #val_loss_root = loss_function(pred[:, :, -1, :], gt_image[:, :, -1, :]) + loss_function(pred[:, :, -2, :], gt_image[:, :, -2, :]) +loss_function(pred[:, :, -3, :], gt_image[:, :, -3, :])
            val_loss_root = loss_function(pred[:, :, -1, :], gt_image[:, :, -1, :])
            
            total_val_loss = val_loss + val_loss_root * 5
            total_v_loss += total_val_loss.item()
            total_root_v_loss += val_loss_root.item() * 5
            model.train()
            
        #pred = data_load.De_normalize_data_dist(pred.detach().squeeze(1).permute(0,2,1).cpu().numpy(), 0.0, 1.0)
        #gt_image = data_load.De_normalize_data_dist(gt_image.detach().squeeze(1).permute(0,2,1).cpu().numpy(), 0.0, 1.0)
        #masked_input = data_load.De_normalize_data_dist(masked_input.detach().squeeze(1).permute(0,2,1).cpu().numpy(), 0.0, 1.0)
        
        saveUtils.save_result(pred, gt_image, masked_input, num_epoch)
        valid_epoch_loss = total_v_loss/len(valid_dataloader)
        valid_epoch_root_loss = total_root_v_loss/len(valid_dataloader)
        log = "Valid: [Epoch %d] [Valid Loss: %.4f] [Valid Root Loss: %.4f]" % (num_epoch, valid_epoch_loss, valid_epoch_root_loss)
        print(log)
        saveUtils.save_log(log)
        writer.add_scalar("Valid Loss/ Epoch", valid_epoch_loss, num_epoch)  
        writer.add_scalar("Valid Root Loss/ Epoch", valid_epoch_root_loss, num_epoch)     
        saveUtils.save_model(model, num_epoch) # save model per epoch
        #validation per epoch ############
        
        
        
if __name__ == "__main__":
    main(args)