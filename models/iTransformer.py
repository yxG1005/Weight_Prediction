import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
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
    

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.image = configs.image
        self.text = configs.text
        self.features = configs.features
        self.fusion = configs.fusion

        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    None,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        if ((self.features == "M" and self.image) or (self.features == "M" and self.text)) and self.fusion=="NO":
            self.food_mapping = UMRL(configs)

    def forecast(self, x_enc, x_mark_enc):
        if (self.features == "M" and self.image) or (self.features == "M" and self.text):
            weight = x_enc[:, :, -1:]

            food_output = self.food_mapping(x_enc)
            x_enc = torch.cat((food_output, weight), axis=2)

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates 
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out


    def forward(self, x_enc, x_mark_enc, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]