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
        self.meal_num = [self.breakfast, self.lunch, self.supper].count(1)#有几餐参与训练
        print("有几餐参与训练",self.meal_num)

        if self.food_individual:
            print("individual")
            self.food_linear = nn.ModuleList()
            for i in range(self.meal_num):#三餐中有几餐参与训练
                self.food_linear.append(nn.Linear(512, 1))
        else:
            self.food_linear = nn.Linear(512, 1)

    def forward(self, x):#input  b × seq_len × 512

        meal_list = []
        for i in range(self.meal_num):
            meal_list.append(x[:, :, i*512: (i+1)*512])#每一餐都是 [32,seq_len,512]

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
        self.food_img_individual = configs.food_img_individual
        self.food_txt_individual = configs.food_txt_individual
        self.breakfast = configs.breakfast
        self.lunch = configs.lunch
        self.supper = configs.supper
        self.meal_num = [self.breakfast, self.lunch, self.supper].count(1)#有几餐参与训练
        print("有几餐参与训练",self.meal_num)

        if self.food_img_individual:
            print("individual")
            self.food_img_linear = nn.ModuleList()
            for i in range(self.meal_num):#三餐中有几餐参与训练
                self.food_img_linear.append(nn.Linear(512, 1))
        else:
            self.food_img_linear = nn.Linear(512, 1)

        if self.food_txt_individual:
            print("individual")
            self.food_txt_individual = nn.ModuleList()
            for i in range(self.meal_num):#三餐中有几餐参与训练
                self.food_txt_linear.append(nn.Linear(512, 1))
        else:
            self.food_txt_linear = nn.Linear(512, 1)

    def forward(self, x):#input  b × seq_len × 512

        img_meal_list = []
        for i in range(self.meal_num, self.meal_num * 2):
            img_meal_list.append(x[:, :, i*512: (i+1)*512])#每一餐都是 [32,seq_len,512]

        txt_meal_list = []
        for i in range(self.meal_num):
            txt_meal_list.append(x[:, :, i*512: (i+1)*512])#每一餐都是 [32,seq_len,512]

        img_output = torch.zeros([x.size(0), x.size(1), self.meal_num],dtype=x.dtype).to(x.device)#[32,seq_len,meal_num]
        txt_output = torch.zeros([x.size(0), x.size(1), self.meal_num],dtype=x.dtype).to(x.device)#[32,seq_len,meal_num]
        
        if self.food_img_individual: 
            for i in range(self.meal_num):       
                img_output[:,:,i:i+1] = self.food_img_linear[i](img_meal_list[i])#[32,seq_len,1]
        else:
            for i in range(self.meal_num):       
                img_output[:,:,i:i+1] = self.food_img_linear(img_meal_list[i])#[32,seq_len,1]    

        if self.food_txt_individual: 
            for i in range(self.meal_num):       
                txt_output[:,:,i:i+1] = self.food_txt_linear[i](txt_meal_list[i])
        else:
            for i in range(self.meal_num):     
                txt_output[:,:,i:i+1] = self.food_txt_linear(txt_meal_list[i])
        # print("txt_output", txt_output)
        # print("img_output", img_output)
        output = 0.5 * img_output + 0.5 * txt_output
        # print("output", output)
        return output

# class early_fusion_food_bais_indiv(nn.Module):
#     """
#     the influence of food
#     """
#     def __init__(self, configs):
#         super(early_fusion_food_bais_indiv, self).__init__()
#         self.food_img_individual = configs.food_img_individual
#         self.food_txt_individual = configs.food_txt_individual
#         self.breakfast = configs.breakfast
#         self.lunch = configs.lunch
#         self.supper = configs.supper
#         self.meal_num = [self.breakfast, self.lunch, self.supper].count(1)#有几餐参与训练
#         print("有几餐参与训练",self.meal_num)

#         self.food_img_linear = nn.Linear(512, 1)

#     def forward(self, x):#input  b × seq_len × 512

#         img_meal_list = []
#         for i in range(self.meal_num):
#             img_meal_list.append(x[:, :, i*512: (i+1)*512])#每一餐都是 [32,seq_len,512]

#         txt_meal_list = []
#         for i in range(self.meal_num, self.meal_num * 2):
#             txt_meal_list.append(x[:, :, i*512: (i+1)*512])#每一餐都是 [32,seq_len,512]

#         img_output = torch.zeros([x.size(0), x.size(1), self.meal_num],dtype=x.dtype).to(x.device)#[32,seq_len,meal_num]
#         txt_output = torch.zeros([x.size(0), x.size(1), self.meal_num],dtype=x.dtype).to(x.device)#[32,seq_len,meal_num]
        
#         for i in range(self.meal_num):   
#             img_output[:,:,i:i+1] = self.food_img_linear(img_meal_list[i])#[32,seq_len,1]  

#         for i in range(self.meal_num):
#             txt_output[:,:,i:i+1] = self.food_img_linear(txt_meal_list[i])
#         print("txt_output", txt_output)
#         print("img_output", img_output)
#         output = (txt_output + img_output) / 2
#         print("output", output)

#         return output



    
# class early_fusion_average_food_bais(nn.Module):
#     """
#     the influence of food
#     """
#     def __init__(self, configs):
#         super(early_fusion_average_food_bais, self).__init__()
#         self.food_individual = configs.food_individual
#         self.breakfast = configs.breakfast
#         self.lunch = configs.lunch
#         self.supper = configs.supper
#         self.meal_num = [self.breakfast, self.lunch, self.supper].count(1)#有几餐参与训练
#         print("有几餐参与训练",self.meal_num)

#         if self.food_individual:
#             print("individual")
#             self.food_linear = nn.ModuleList()
#             for i in range(self.meal_num):#三餐中有几餐参与训练
#                 self.food_linear.append(nn.Linear(512, 1))
#         else:
#             self.food_linear = nn.Linear(512, 1)

#     def forward(self, x):#input  b × seq_len × 512


#         img_meal_list = []
#         for i in range(self.meal_num):
#             img_meal_list.append(x[:, :, i*512: (i+1)*512])#每一餐都是 [32,seq_len,512]

#         txt_meal_list = []
#         for i in range(self.meal_num, self.meal_num * 2):
#             txt_meal_list.append(x[:, :, i*512: (i+1)*512])#每一餐都是 [32,seq_len,512]

#         meal_list = []
#         for i in range(self.meal_num):
#             meal_list.append((img_meal_list[i] + txt_meal_list[i])/2)


#         output = torch.zeros([x.size(0), x.size(1), self.meal_num],dtype=x.dtype).to(x.device)#[32,seq_len,meal_num]

        
#         if self.food_individual: 
#             for i in range(self.meal_num):       
#                 output[:,:,i:i+1] = self.food_linear[i](meal_list[i])#[32,seq_len,1]

#         else:
#             for i in range(self.meal_num):     
#                 output[:,:,i:i+1] = self.food_linear(meal_list[i])#[32,seq_len,1]

#         return output







class Model(nn.Module):
    """
    linear base
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual
        self.features = configs.features
        self.image = configs.image
        self.text = configs.text
        self.variation = configs.variation
        self.mix_variation = configs.mix_variation
        self.fusion = configs.fusion
        
    
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
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


            if self.variation :
                variation = x[:, :, -2:-1]
                weight = torch.cat((food_output, variation, weight), axis=2)

            else:
                weight = torch.cat((food_output, weight), axis=2)

        elif self.features == "M" and self.variation:
            weight = torch.cat((x[:, :, :-1], weight), axis = 2)

        # print("送入Nlinear的数据是", weight)

        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            weight = self.Linear(weight.permute(0,2,1)).permute(0,2,1)



        y = weight[:, :, -1:] + seq_last #输出的最后一列需要加回减掉的最后一行

        y = torch.cat((weight[:, :, :-1], y) ,axis=2) #再将输出的前面几列和已经加回的那一部分cat

        #可能交互一下变量
        if self.features == "M" and self.mix_variation:
            y = self.mix_linear(y)
        

        # print("y size", y, y.shape)         

        return y # [Batch, Output length, Channel]  
    

