import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random



def remove_foot_contacts(data): # chaneel 73 -> 69, 69 is baseline 
    assert data.shape[0] == 73
    return np.delete(data, obj=list(range(data.shape[0] - 4, data.shape[0])), axis=0)

'''def calculate_data_dist(data):
    assert data.shape[2] == 73
    temp = data.reshape(-1, 73)
    remove_foot_contacts
    return remove_foot_contacts(np.mean(temp, axis = 0)), remove_foot_contacts(np.std(temp, axis = 0))


def De_normalize_data_dist(result, dataset_mean, dataset_std):
    
    #row wise normalization (69, 240)
    for idx, dist in enumerate(zip(dataset_mean, dataset_std)):
        mean, std = dist
        if std == 0.0:
            std = 1
        result[idx, :] = result[idx, :] * std + mean
    
    return result
        
        
def normalize_data_dist(gt_image, dataset_mean, dataset_std):
    
    #row wise normalization (69, 240)
    for idx, dist in enumerate(zip(dataset_mean, dataset_std)):
        mean, std = dist
        if std == 0.0:
            std = 1
        gt_image[idx, :] = (gt_image[idx, :] - mean) / std
    
    return gt_image'''

class MotionLoader(Dataset):
        def __init__(self, root, IsNoise=False, IsTrain=True, dataset_mean=None, dataset_std=None):
            super(MotionLoader, self).__init__()
            #loda all the data as array
            print("#### MotionLoader ####")
            print("####### load data from {} ######".format(root))
            
            self.IsNoise = IsNoise
            self.IsTrain = IsTrain
            self.masking_length_mean = 10
            data_list = os.listdir(root)
            for idx , name in enumerate(data_list):
                file_path = os.path.join(root, name)
                if idx == 0 :
                    self.data = np.load(file_path)['clips'] #(clip num, 240, 73) (2846, 240, 73)
                else:
                    self.data = np.concatenate((self.data, np.load(file_path)['clips']), axis=0) # concat all the data (# of data , 240 , 73)
            '''if self.IsTrain == True:
                self.dataset_mean, self.dataset_std = calculate_data_dist(self.data)
                #print("dataset_mean", self.dataset_mean.shape)
                #print("dataset_std", self.dataset_std.shape)
            else:
                self.dataset_mean = dataset_mean
                self.dataset_std = dataset_std'''
                
            print("####### total Lentgh is {} ######".format(self.__len__()))
            
        def mean_std(self):
            return self.dataset_mean, self.dataset_std    
        
        def __getitem__(self, idx):
            #load item #processing
            gt_image = self.remove_foot_contacts(self.data[idx]) # remove_foot_contacts  (240 , 73) --> (240 , 69)
            #switch (240 , 69) --> (69, 240)
            gt_image = np.transpose(gt_image)
            #gt_image = normalize_data_dist(gt_image, 0.0, 1.0)
            #get masked input
            
            #masked_input, gt_image = self.masking_input(gt_image, self.IsNoise)
            #masking_length = 60
            #if self.masking_length_mean < 120 and epoch is not 0 and epoch%10 == 0:
            #     self.masking_length_mean = self.masking_length_mean + 10
            
            masking_length = round(np.random.normal(self.masking_length_mean, 20.0)) # std 20
            
            if masking_length <= 0 : 
                masking_length = 1
            if masking_length >= 240 :
                masking_length = 239
                
            orig_height = gt_image.shape[0] #69
            orig_width = gt_image.shape[1] #240
            #test denoising
            #masked_input = gt_image.copy() + np.random.randn(orig_height, orig_width) * 0.1 # deep copy
            
            # for maksing
            
            masked_input = gt_image.copy()
            mask_width = masking_length #+ self.noise(False) #55 ~ 65
            if self.IsTrain == True:
                #In training step, randomly masking
                masking = np.zeros((orig_height, mask_width))# generate zeros matrix for masking: orig_height x mask_width
                index = random.randint(0, orig_width - mask_width)# sampling the start point of masking 
                masked_input[: , index : index+mask_width] = masking # masking
            else:
                #In test phase, center of the data are masked
                masking = np.zeros((orig_height, mask_width))# generate zeros matrix for masking: orig_height x mask_width
                #index = random.randint(0, orig_width - mask_width)# sampling the start point of masking 
                index = 120 - mask_width // 2
                masked_input[: , index : index+mask_width] = masking # masking
            
            #print(np.sum(masked_input) == np.sum(gt_image))
            #return maksed_input and gt CHW #(69, 240) --> (1, 69, 240)
            return np.expand_dims(masked_input, axis=0), np.expand_dims(gt_image, axis=0) # it will (batch, 1, 69, 240)
        
        def __len__(self):
            return len(self.data)
    
        
        def remove_foot_contacts(self, data): # chaneel 73 -> 69, 69 is baseline 
            assert data.shape[1] == 73
            return np.delete(data, obj=list(range(data.shape[1] - 4, data.shape[1])), axis=1)
        
        
        def noise(self, option = True):
            if option == True:
                return random.randint(-5, 5)
            else:
                return 0

        '''def masking_input(self, gt_image, masking_length = 60, IsNoise = False):
            # input sample of size 69 × 240
            # in paper 10 ~ 120
            # I will set 60
            orig_height = gt_image.shape[0] #69
            orig_width = gt_image.shape[1] #240
            masked_input = gt_image.copy() # deep copy
            mask_width = masking_length + self.noise(IsNoise) #55 ~ 65
            
            masking = np.zeros((orig_height, mask_width))# generate zeros matrix for masking: orig_height x mask_width
            index = random.randint(0, orig_width - mask_width)# sampling the start point of masking 
            masked_input[: , index : index+mask_width] = masking # masking
            print("in func: ",np.sum(masked_input) == np.sum(gt_image))
            return masked_input, gt_image  '''      
        
def get_dataloader(dataroot, batch_size, IsNoise=False, IsTrain=True, dataset_mean=None, dataset_std=None):
    dataset = MotionLoader(dataroot, IsNoise=IsNoise, IsTrain=IsTrain, dataset_mean=dataset_mean, dataset_std=dataset_std)
    print("# of dataset:", len(dataset))
    
    if IsTrain == True:
        IsSuffle = True
    else:
        IsSuffle = False
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=IsSuffle, drop_last=True)
    if IsTrain == True:
        #mean, std = dataset.mean_std()
        return dataloader, dataset
    else:
        return dataloader, dataset


if __name__ == "__main__":
    print("START")
    data_root = 'C:/Users/VML/Desktop/2022_Spring/Motion_Graphics/Final_project/downloadCode/valid_data/'
    batch_size = 32
    datalodaer, _ = get_dataloader(data_root , 32)
    datalodaer2, _ = get_dataloader(data_root , 32)
    
    for iter, item0 in enumerate(zip(datalodaer, datalodaer2)): 
        #print(item.shape)
        item , item2 = item0
        masked_input, gt_image = item
        masked_input2, gt_image2 = item2
        print("masked_input.sum(): ", masked_input.sum())
        print("masked_input2.sum(): ", masked_input2.sum()) 
        print(iter)
        print(masked_input.shape)
        print(gt_image.shape)
        if iter == 5 :
            break
    
    print("END")
    
    '''for iter, item in enumerate(datalodaer2): 
        
        
        masked_input, gt_image = item
        print("masked_input.sum(): ", masked_input.sum())
        #print("masked_input2.sum(): ", masked_input2.sum()) 
        print(iter)
        print(masked_input.shape)
        print(gt_image.shape)
        if iter == 5 :
            break
    
    print("END")'''