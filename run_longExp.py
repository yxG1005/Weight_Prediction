import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)

parser = argparse.ArgumentParser(description='Weight Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='NLinear',
                    help='model name, options: [NLinear, PatchTST, iTransformer]')

# data loader
parser.add_argument('--image', type=int, required=True, default=0, help='whether to add images')
parser.add_argument('--text', type=int, required=True, default=0, help='whether to add texts(ingredients)')
parser.add_argument('--text_from_img', action='store_true', default=False, help='whether get text(ingredients) predicted from img')
parser.add_argument('--breakfast', type=int, required=True, default=0, help='whether to add breakfast')
parser.add_argument('--lunch', type=int, required=True, default=0, help='whether to add lunch')
parser.add_argument('--supper', type=int, required=True, default=0, help='whether to add supper')
parser.add_argument('--root_path', type=str, default='dataset', help='root path of the data file')
parser.add_argument('--image_root', type=str, default='/share/test/yxgui/', help='change to your path ')
parser.add_argument('--feature_path', type=str, default='/share/ckpt/guiyinxuan/', help='change to your path ')
parser.add_argument('--data_path', type=str, default='data.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S]; M:multivariate predict multivariate, S:univariate predict univariate')
parser.add_argument('--target', type=str, default='weight', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='iTransformer/lmm', help='location of model checkpoints')
parser.add_argument('--checkpoints_root', type=str, default='/share/ckpt/guiyinxuan/LTSF-ckpt', help='checkpoints root path, change to your path')

# forecasting task
parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')
parser.add_argument('--fusion', required=True, type=str, default="early", help='options:[early, NO]; early: early fusion; NO: not fusion')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--patchtst_individual', type=int, default=1, help='individual head; True 1 False 0')


# Models
parser.add_argument('--food_individual', action='store_true', default=False, help='weight of food in DLinear')
parser.add_argument('--lamda', default=0.25, type=float, help='balance weight loss and food loss')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=4, help='encoder input size') # only used in PatachTST, equals to (number of diets)+1
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')#transformer模型维度
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')


# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')


args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)


Exp = Exp_Main


if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments

        #NLinear
        if args.model == "NLinear":
            setting = '{}_f{}_s{}_p{}_food_{}{}{}_l_{}'.format(
                args.model,
                args.features,
                args.seq_len,
                args.pred_len,
                args.breakfast,
                args.lunch,
                args.supper,
                args.lamda)
            if args.features == "S":
                setting = '{}_f{}_s{}_p{}'.format(
                    args.model,
                    args.features,
                    args.seq_len,
                    args.pred_len)   
        
        
        # iTransformer or PatchTST
        elif args.model == "iTransformer" or args.model == "PatchTST":
            setting = '{}_f{}_s{}_p{}_depth{}_d_model{}_nhead{}_dff{}_food_{}{}{}_l_{}'.format(
                args.model,
                args.features,
                args.seq_len,
                args.pred_len,
                args.e_layers,
                args.d_model,
                args.n_heads,
                args.d_ff,
                args.breakfast,
                args.lunch,
                args.supper,
                args.lamda)
            if args.features == "S":
                setting = '{}_f{}_s{}_p{}_depth{}_d_model{}_nhead{}_dff{}'.format(
                    args.model,
                    args.features,
                    args.seq_len,
                    args.pred_len,
                    args.e_layers,
                    args.d_model,
                    args.n_heads,
                    args.d_ff)              
         



        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        if not args.train_only:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

        torch.cuda.empty_cache()
else:
    ii = 0
    if args.model == "NLinear":
        setting = '{}_f{}_s{}_p{}_food_{}{}{}_l_{}'.format(
            args.model,
            args.features,
            args.seq_len,
            args.pred_len,
            args.breakfast,
            args.lunch,
            args.supper,
            args.lamda)
        if args.features == "S":
            setting = '{}_f{}_s{}_p{}'.format(
                args.model,
                args.features,
                args.seq_len,
                args.pred_len)   
    
    
    # iTransformer or PatchTST
    elif args.model == "iTransformer" or args.model == "PatchTST":
        setting = '{}_f{}_s{}_p{}_depth{}_d_model{}_nhead{}_dff{}_food_{}{}{}_l_{}'.format(
            args.model,
            args.features,
            args.seq_len,
            args.pred_len,
            args.e_layers,
            args.d_model,
            args.n_heads,
            args.d_ff,
            args.breakfast,
            args.lunch,
            args.supper,
            args.lamda)
        if args.features == "S":
            setting = '{}_f{}_s{}_p{}_depth{}_d_model{}_nhead{}_dff{}'.format(
                args.model,
                args.features,
                args.seq_len,
                args.pred_len,
                args.e_layers,
                args.d_model,
                args.n_heads,
                args.d_ff)              
         
        exp = Exp(args)  # set experiments


        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
