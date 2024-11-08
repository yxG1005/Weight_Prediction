import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UMRL(nn.Module):
    """
    the influence of food
    """
    def __init__(self, configs):
        super(UMRL, self).__init__()
        self.food_individual = configs.food_individual
        self.breakfast = configs.breakfast
        self.lunch = configs.lunch
        self.supper = configs.supper
        self.meal_num = [self.breakfast, self.lunch, self.supper].count(1)
        print("number of meals",self.meal_num)

        if self.food_individual:
            self.food_linear = nn.ModuleList()
            for i in range(self.meal_num):
                self.food_linear.append(nn.Linear(512, 1))
        else:
            self.food_linear = nn.Linear(512, 1)

    def forward(self, x):#input  b × seq_len × 512

        meal_list = []
        for i in range(self.meal_num):
            meal_list.append(x[:, :, i*512: (i+1)*512])

        output = torch.zeros([x.size(0), x.size(1), self.meal_num],dtype=x.dtype).to(x.device)#[32,seq_len,meal_num]

        
        if self.food_individual: 
            for i in range(self.meal_num):       
                output[:,:,i:i+1] = self.food_linear[i](meal_list[i])#[32,seq_len,1]

        else:
            for i in range(self.meal_num):     
                output[:,:,i:i+1] = self.food_linear(meal_list[i])#[32,seq_len,1]

        return output

class early_fusion_UMRL(nn.Module):
    """
    the influence of food
    """
    def __init__(self, configs):
        super(early_fusion_UMRL, self).__init__()
        # self.food_img_individual = configs.food_img_individual
        # self.food_txt_individual = configs.food_txt_individual
        self.breakfast = configs.breakfast
        self.lunch = configs.lunch
        self.supper = configs.supper
        self.meal_num = [self.breakfast, self.lunch, self.supper].count(1)
        print("number of meals",self.meal_num)


        self.food_img_linear = nn.Linear(512, 1)
        self.food_txt_linear = nn.Linear(512, 1)

    def forward(self, x):#input  b × seq_len × 512

        img_meal_list = []
        for i in range(self.meal_num, self.meal_num * 2):
            img_meal_list.append(x[:, :, i*512: (i+1)*512])

        txt_meal_list = []
        for i in range(self.meal_num):
            txt_meal_list.append(x[:, :, i*512: (i+1)*512])

        img_output = torch.zeros([x.size(0), x.size(1), self.meal_num],dtype=x.dtype).to(x.device)#[32,seq_len,meal_num]
        txt_output = torch.zeros([x.size(0), x.size(1), self.meal_num],dtype=x.dtype).to(x.device)#[32,seq_len,meal_num]
        

        for i in range(self.meal_num):       
            img_output[:,:,i:i+1] = self.food_img_linear(img_meal_list[i])#[32,seq_len,1]    
        for i in range(self.meal_num):     
            txt_output[:,:,i:i+1] = self.food_txt_linear(txt_meal_list[i])

        output = 0.5 * img_output + 0.5 * txt_output
        return output



class Model(nn.Module):
    """
    linear base
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.features = configs.features
        self.image = configs.image
        self.text = configs.text
        self.fusion = configs.fusion

        self.Linear = nn.Linear(self.seq_len, self.pred_len)

        if ((self.features == "M" and self.image) or (self.features == "M" and self.text)) and self.fusion!="early":
            self.food_mapping = UMRL(configs)

        if self.features == "M" and self.image and self.text and self.fusion == "early":
            self.food_mapping = early_fusion_UMRL(configs)


    def forward(self, x):
        #[batch_size, seq_len, channels]
        weight = x[:, :, -1:]
        seq_last = weight[:,-1:,:].detach()    
        weight = weight - seq_last


        if (self.features == "M" and self.image) or (self.features == "M" and self.text):
            food_output = self.food_mapping(x)

            weight = torch.cat((food_output, weight), axis=2)

        weight = self.Linear(weight.permute(0,2,1)).permute(0,2,1)

        y = weight[:, :, -1:] + seq_last 
        y = torch.cat((weight[:, :, :-1], y) ,axis=2) 


        return y # [Batch, Output length, Channel]  
    

