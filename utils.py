import os
import os.path
import torch
import sys
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt



class saveData():
    def __init__(self, args):
        self.args = args
        #Generate Savedir folder
        self.save_dir = os.path.join(args.saveDir, args.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        #Generate Savedir/model
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        #Generate Savedir/validation
        self.save_dir_validation = os.path.join(self.save_dir, 'validation')
        if not os.path.exists(self.save_dir_validation):
            os.makedirs(self.save_dir_validation)

        #Generate Savedir/checkpoint
        self.save_dir_checkpoint = os.path.join(self.save_dir, 'checkpoint')
        if not os.path.exists(self.save_dir_checkpoint):
            os.makedirs(self.save_dir_checkpoint)

        #Generate Savedir/tensorBoard
        self.save_dir_tensorBoard = os.path.join(self.save_dir, 'tensorBoard')
        if not os.path.exists(self.save_dir_tensorBoard):
            os.makedirs(self.save_dir_tensorBoard)

        #Generate Savedir/log.txt
        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')
    
    def save_log(self, log):
        sys.stdout.flush()
        self.logFile.write(log + '\n')
        self.logFile.flush()
        
    def save_model(self, model, epoch):
        torch.save(
            model.state_dict(),
            self.save_dir_model + '/model_' + str(epoch) + '.pt')
    
    def save_result(self, pred, gt, masked_input, epoch):
        pred = pred.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        gt = gt.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        masked_input = masked_input.detach().squeeze(1).permute(0,2,1).cpu().numpy()
        #style_input = style_input.detach().squeeze(1).permute(0,2,1).cpu().numpy()

        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "_Results", pred)
        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "_GT", gt)
        np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "_Masked_Input", masked_input)
        #np.save(self.save_dir_validation + '/epoch_' + str(epoch) + "_Style_Input", style_input)

        cmap = plt.get_cmap('jet') 
        
        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(pred[i], cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("prediction", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'_prediction_'+str(i)+'.png')
            
        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(gt[i], cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("gt", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'_gt_'+str(i)+'.png')  
            
        for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(masked_input[i], cmap=cmap)
            #plt.matshow(np.zeros(masked_input[i].shape), cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("masked_input", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'_masked_input_'+str(i)+'.png')

        '''for i in range(1): 
            plt.figure(figsize=(2,4))
            plt.matshow(style_input[i], cmap=cmap)
            #plt.matshow(np.zeros(masked_input[i].shape), cmap=cmap)
            plt.clim(-100, 50)
            #plt.axis('off')
            plt.title("style_input", fontsize=25)
            plt.savefig(self.save_dir_validation + '/epoch_'+ str(epoch) +'_style_input_'+str(i)+'.png')'''
        
        plt.close('all')