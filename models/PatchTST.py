__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp

class UMRL(nn.Module):
    """
    the influence of food
    """
    def __init__(self, configs):
        super(UMRL, self).__init__()
        # self.seq_len = configs.seq_len
        # self.pred_len = configs.pred_len
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
            # print("food_individual")
            for i in range(self.meal_num):       
                # print("meal_list[{}]".format(i), meal_list[i].shape)
                output[:,:,i:i+1] = self.food_linear[i](meal_list[i])#[32,seq_len,1]

        else:
            for i in range(self.meal_num):     
                # print("meal_list[{}]".format(i), meal_list[i].shape)
                output[:,:,i:i+1] = self.food_linear(meal_list[i])#[32,seq_len,1]
                # print("output[:,:,i:i+1]", output[:,:,i:i+1])
        # print("food_output.shape", output, output.shape)
        return output
    

class Model(nn.Module):
    def __init__(self, configs, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=False, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        
        super().__init__()
        
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.patchtst_individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin#是否标准化1/0
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        self.image = configs.image
        self.text = configs.text
        self.variation = configs.variation
        self.features = configs.features
        self.mix_variation = configs.mix_variation
        self.fusion = configs.fusion
 
        
        # model
        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
    
        if ((self.features == "M" and self.image) or (self.features == "M" and self.text)) and self.fusion=="NO":
            self.food_mapping = UMRL(configs)
 
    def forward(self, x):           # x: [Batch, Input length, Channel]

        if (self.features == "M" and self.image) or (self.features == "M" and self.text):
            weight = x[:, :, -1:]#体重部分

            food_output = self.food_mapping(x)

            if self.variation:
                variation = x[:, :, -2:-1]
                x = torch.cat((food_output, variation, weight), axis=2)

            #将食物mapping后的output和weight concat在一起
            else:
                x = torch.cat((food_output, weight), axis=2)


        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            print("x", x.shape)
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x