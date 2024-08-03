import random
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import  iTransformer, NLinear, PatchTST
from utils.tools import EarlyStopping, EarlyStopping_original, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import utils.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
# from utils.picture import draw


import numpy as np
import pandas as pandas
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')




class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        r_path =  "tensorboard/{}-{}".format(str(self.args.seq_len), str(self.args.pred_len))
        if self.args.features == "S":
            modal = "S"
        elif self.args.image:
            modal = "img"
            # if self.args.lamda_ablation:
            #     modal = "img_lamda_ablation"
        elif self.args.text and self.args.text_from_img:
            modal = "lmm"
            # if self.args.lamda_ablation:
            #     modal = "lmm_lamda_ablation"
            if self.args.fusion == "early":
                modal = "img_lmm_early"
        else:
            modal = "txt" 
            # if self.args.lamda_ablation:
            #     modal = "txt_lamda_ablation"
            if self.args.fusion == "early":
                modal = "img_txt_early"
        log_dir = os.path.join( r_path, self.args.model, 
                                            modal + '_food_' + str(self.args.breakfast)
                                               + str(self.args.lunch) + str(self.args.supper) +"_l_" +str(self.args.lamda) )
        # if self.args.lamda_ablation:
        #     log_dir = os.path.join( r_path, self.args.model, 
        #                                     modal + '_food_' + str(self.args.breakfast)
        #                                        + str(self.args.lunch) + str(self.args.supper) +"_l_" +str(self.args.lamda) )
        if modal == "S":
            log_dir = os.path.join( r_path, self.args.model, modal )       
        
        print("tensorboard_dir", log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)

    def _build_model(self):
        model_dict = {
            'iTransformer': iTransformer,
            'NLinear': NLinear,
            'PatchTST': PatchTST,
        }
 
        # if self.args.is_training == False and self.args.fusion == "late" :
        #     image_model = model_dict[self.args.model].Model(self.args).float()
        #     text_model = model_dict[self.args.model].Model(self.args).float()
        #     return image_model, text_model
        
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        # if self.args.features == 'M' and self.args.image:
        #     food_model = Food.food_bais(self.args).float()
        #     return model, food_model

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # real_total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
   

                    elif self.args.model == 'iTransformer':
                        outputs = self.model(batch_x, x_mark_enc=None)

                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
  
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
                # real_total_loss.append(real_loss)
        total_loss = np.average(total_loss)
        # real_total_loss = np.average(real_total_loss)
        self.model.train()

        return total_loss

    def train(self, setting):
        lamda = self.args.lamda
        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        fix_seed = 2021
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            # for name, layer in self.model.named_parameters():
            #     if name == "Linear.weight":
            #         print(name, layer.cpu().data.numpy())

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)



                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # print("dec_inp", dec_inp.shape)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # print("dec_inp", dec_inp.shape)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    elif self.args.model == 'iTransformer':
                        outputs = self.model(batch_x, x_mark_enc=None)

                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    if self.args.features == 'M':
                        batch_y = batch_y[:, -self.args.pred_len:, -2:].to(self.device)  

                        vari = batch_y[:, :, :1].repeat(1, 1, outputs.shape[2] - 1)
                        batch_y = torch.cat((vari, batch_y[:, :, -1:]), axis=2)

                    #loss ablation
                    if self.args.features == 'M':
                        m_list = []
                        meal_num = [self.args.breakfast, self.args.lunch, self.args.supper].count(1)
                        for l in range(0, meal_num):
                            m_list.append((outputs[:, :, l:l+1] - batch_y[:, :, l:l+1]) ** 2)
        

                        w = (outputs[:, :, -1:]-batch_y[:, :, -1:]) ** 2
                        L_weight = w
                        L_diet = (1/3) * (sum(m_list))
                        loss = lamda * L_weight + (1- lamda) * L_diet
                        loss = loss.sum() / (outputs.shape[0] * outputs.shape[1])
                        # print(loss)

                    #original loss
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss= self.vali(test_data, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(vali_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            self.writer.add_scalar("train_loss", train_loss, epoch)
            self.writer.add_scalar("vali_loss", vali_loss, epoch)
            self.writer.add_scalar("test_loss", test_loss, epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    


    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        criterion = self._select_criterion()
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints , setting , 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        inverse_preds = []
        inverse_trues = []
   

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_x_mark = None
                batch_y_mark = None

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)

                    elif self.args.model == 'iTransformer':
                        outputs = self.model(batch_x, x_mark_enc=None)

                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' or 'M' else 0
 
                
                inverse_pred = outputs[:, -self.args.pred_len:, f_dim:]
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                inverse_true = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                

                outputs = outputs.detach().cpu().numpy()
                inverse_pred = inverse_pred.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                inverse_true = inverse_true.detach().cpu().numpy()
                

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inverse_preds.append(inverse_pred)
                inverse_trues.append(inverse_true)

                # inputx.append(batch_x.detach().cpu().numpy())
                inputx.append(batch_x)
     
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            

        preds = np.concatenate(preds, axis=0)

        trues = np.concatenate(trues, axis=0)
        inputx = torch.concat(inputx, axis=0)

        inverse_preds = np.concatenate(inverse_preds, axis=0)
        inverse_trues= np.concatenate(inverse_trues, axis=0)



        index_dict = test_data.get_message()[0]
        old_csv = './dataset/{}_test_weight_{}.csv'.format(self.args.seq_len + self.args.pred_len, self.args.scale)
        result_path = self.metric_pred(old_csv, "weight_pred", test_data, inverse_preds, inverse_trues, inputx.shape[1], inverse_preds.shape[1], index_dict)

        mae, mse = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('\n')
        f.write('\n')
        f.close()

        GT_col = "original_weight" if test_data.scale==1 else "weight"
        self.draw_sample("weight_pred", result_path, GT_col, inputx.shape[1])

        

        print("------------------long term prediction --------------------")
        longterm_predict = self.longterm_predict(test_data, inputx, preds)
        result_path = self.metric_pred(result_path, "weight_longterm", test_data, longterm_predict, inverse_trues, inputx.shape[1], inverse_preds.shape[1], index_dict)
        self.draw_sample("weight_longterm", result_path, GT_col, inputx.shape[1])

        return


    """
    长期预测 没有真实的食物，全部用历史给的食物，以给出的天数为循环周期循环
    """
    def longterm_predict(self, test_data, inputx, preds):
        # self.food_model.eval()
        i=0#i是df_test的行数
        k=0#k是第几个样本
        outputs = []
        longterm_dict = dict()
        sample_len = self.args.seq_len + self.args.pred_len
        df_test = pandas.read_csv(os.path.join(test_data.root_path,
                    str(self.args.seq_len + self.args.pred_len) +'_test_weight_{}.csv'.format(self.args.scale)))
        while i<df_test.shape[0]:
            # print("-----一个新人-----")
            # print("input", inputx.shape[0])#[1, 3, 1537]
            # print("pred", preds[k])

            concurrent_ID = df_test.iloc[i][0]
            temp_df = df_test[df_test['ID']==concurrent_ID]
            # print("temp_df",temp_df.shape[0])

            input = inputx[k].unsqueeze(0) 
            history_GT_food = inputx[k, :, :-2].unsqueeze(0)#一个人的给模型的唯一GT_food 1 * seq_len * 512
            # print("history_GT_food", history_GT_food, history_GT_food.shape)

            user_dict = dict()
            for j in range(input.shape[1]):
                if str(j + 1) not in user_dict:
                    user_dict[str(j + 1)] = []
                user_dict[str(j + 1)].append(input[0, j:j+1, -1])

            with torch.no_grad():
                dec_inp = torch.zeros_like(input[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([input[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                food = history_GT_food
                # print("一个人的唯一GT food", food)
                for item in range(temp_df.shape[0] + 1 - sample_len):#一个individual能贡献的样本数
                    # print("item",item)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            pass
                    else:
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                                # print("input", input) # 1 × seq_len × dim
                                output = self.model(input)
                                # print("output", output)
                        elif self.args.model == "iTransformer":#itransformer
                            output = self.model(input, x_mark_enc=None)
                        else:
                            output = self.model(input, None, dec_inp, None)

                    f_dim = -1 if self.args.features == 'MS' or 'M' else 0
                    output = output[:, -self.args.pred_len:, f_dim:]
                    # print("long output", output.shape)

                    outputs.append(output.detach().cpu().numpy())

                    # print("input", input)
                    # print("output", output)
                    for j in range(output.shape[1]):
                        if str(input.shape[1] + item + 1 + j) not in user_dict:
                            user_dict[str(input.shape[1] + item + 1 + j)] = []
                        user_dict[str(input.shape[1] + item + 1 + j)].append(output[0, j:j+1, -1])

                    new_input = []
                    variation_input = []
                    for j in range(item+2, item + 2 + input.shape[1]):#input不是output
                        pred_value = sum(user_dict[str(j)])/len(user_dict[str(j)])
                        # print("pred_value", pred_value)
                        new_input.append(pred_value)
                        if self.args.variation:
                            variation_input.append(pred_value - sum(user_dict[str(j-1)])/len(user_dict[str(j-1)]))
                    input = torch.stack(new_input, axis=0).unsqueeze(0)
                    if variation_input!=[]:
                        variation = torch.stack(variation_input, axis=0).unsqueeze(0)
                        # print("variation", variation)
                        input = torch.cat((variation, input), axis=2)
                    #input: [1, seq_len, 1]
                    if (self.args.features=='M' and self.args.image) or (self.args.features=='M' and self.args.text):
                        if k + item + 1 < inputx.shape[0]:
                            new_image = inputx[k + item + 1].unsqueeze(0)[:, :, :-2]#【1, seq_len, 512*n】
                            food = torch.cat((food[:, 1:, :], food[:, :1, :]), axis = 1)
                            # print("新food", food)
                            input = torch.cat((food, input), axis=2)

                    # print("k + item + 1", k + item + 1)
                k += temp_df.shape[0] + 1 - sample_len
                i = i+temp_df.shape[0]

        outputs = np.concatenate(outputs, axis=0)

        return outputs


    def metric_pred(self, old_csv, tar_col, test_data, inverse_preds, inverse_trues, seq_len, pred_len, index_dict, strategy = 'mean'):# mean/max/min/first/last
        # print("metric----")
        if not os.path.exists('./result'):
            os.makedirs('./result')

    

        f_dim = -1 if self.args.features == 'MS' or 'M' else 0
        real_PD = inverse_preds[:, -self.args.pred_len:, f_dim:]   
        real_GT = inverse_trues[:, -self.args.pred_len:, f_dim:] 

        # print("seq_len", seq_len)
        # message = test_data.get_message()
        # index_dict = index_dict
        start_key = 0 #1281
        start_index = index_dict[str(start_key)]

        assert strategy in ['mean', 'max', 'min', 'first', 'last']#first:较远的输入序列预测的结果  last:较近的输入序列预测的结果

        temp_d = dict()#形如 "test文件中行数":[pred1, pred2, pred3...] 方便后续取mean/max..

        for i in range(real_PD.shape[0]):
            pred = real_PD[i]
            # print("pred", pred)
            # print("pred", pred.dtype)
            real_key = i + start_key
            test_fi_index = index_dict[str(real_key)] - start_index + seq_len
            # print(test_fi_index)
            for j in range(pred_len):
                if str(test_fi_index + j) not in temp_d:
                    temp_d[str(test_fi_index + j)] = np.array([], dtype = np.float32)
                temp_d[str(test_fi_index + j)] = np.append(temp_d[str(test_fi_index + j)], pred[j][0])

        # print("11temp_d") 
        # for item in temp_d:         
        #     print(item, temp_d[item])
        
        df = pandas.read_csv(old_csv)
        # if 'height' in df.columns:
        #     df = df.drop(columns=['date','height','age','sex','breakfast','lunch','supper'])

        individual = []
        sum = 0
        individual.append(0)
        p = 0
        while p < df.shape[0]:
            concurrent_ID = df.iloc[p][0]
            temp_df = df[df['ID']==concurrent_ID]
            individual.append(sum + (temp_df.shape[0] - (self.args.seq_len + self.args.pred_len) + 1))
            sum += temp_df.shape[0] - (self.args.seq_len + self.args.pred_len) + 1
            p = p+temp_df.shape[0]
        # print(individual)
        # 计算error
        if (test_data.scale):
            j = 0
            mae_ = 0
            mse_ = 0
            while j<df.shape[0]:
                concurrent_ID = df.iloc[j][0]
                temp_df = df[df['ID']==concurrent_ID]   

                initial_weight =  temp_df.iloc[0]['original_weight']
                max_weight = (initial_weight+2)
                min_weight = (initial_weight-8)

                for k in range(j + seq_len, j+temp_df.shape[0]):
                    temp_d[str(k)] = temp_d[str(k)]*10 + min_weight
                    for item in temp_d[str(k)]:
                        mae_ += np.abs(item - df.iloc[k]['original_weight'])
                        mse_ += (item - df.iloc[k]['original_weight'] )** 2
                   

                j = j+temp_df.shape[0]
            rmae_ = mae_/(inverse_preds.shape[0]* inverse_preds.shape[1])
            rmse_ = mse_/(inverse_preds.shape[0]* inverse_preds.shape[1])
            for item in temp_d:          
                print(item, temp_d[item])
            print("rmse_",rmse_)
            print("rmae_",rmae_)
            



        final_preds = np.zeros([df.shape[0]], dtype = np.float32) 

        for i in range(df.shape[0]):
            if str(i) in temp_d:
                pred_list = temp_d[str(i)]
                if strategy == 'mean':
                    final_pred =np.mean(pred_list)
                elif strategy == 'max':
                    final_pred = max(pred_list)
                elif strategy =='min':
                    final_pred = min(pred_list)
                elif strategy == 'first':
                    final_pred = pred_list[0]
                else:
                    final_pred = pred_list[-1]
                final_preds[i] = final_pred
        # print(final_preds)

        # df.insert(loc=3, column='weight_pred', value=final_preds)
        df[tar_col] = final_preds

        scale = 'T' if test_data.get_scale() else 'F'

        r_path =  "result/{}-{}".format(str(self.args.seq_len), str(self.args.pred_len))
        if self.args.features == "S":
            modal = "S"
        elif self.args.image:
            modal = "img"
            # if self.args.lamda_ablation:
            #     modal = "img_lamda_ablation"
        elif self.args.text and self.args.text_from_img:
            modal = "lmm"
            # if self.args.lamda_ablation:
            #     modal = "lmm_lamda_ablation"
            if self.args.fusion == "early":
                modal = "img_lmm_early"
        else:
            modal = "txt" 
            # if self.args.lamda_ablation:
            #     modal = "txt_lamda_ablation"
            if self.args.fusion == "early":
                modal = "img_txt_early"
        os.makedirs(r_path, exist_ok=True)
        result_path = os.path.join(r_path, self.args.model + "_" + modal + "_"
                                     + strategy + '_food_' + str(self.args.breakfast) + str(self.args.lunch) + str(self.args.supper)+"_l_"+ str(self.args.lamda) + '.csv')
        # if self.args.lamda_ablation:
        #     result_path = os.path.join(r_path, self.args.model + "_" + modal + "_"
        #                              + strategy + '_food_' + str(self.args.breakfast) + str(self.args.lunch) + str(self.args.supper) +"_l_" + str(self.args.lamda) + '.csv')
        if modal == "S":
            result_path = os.path.join(r_path, self.args.model + "_" + modal + "_" + strategy + '.csv')   


        print("result_path", result_path)
        #" + "_G_" + str(self.args.gamma) +'_tanh' "
        #all_history_food
        df.to_csv(result_path, index=0)

        # if tar_col == "weight_longterm":
        #     real_PD = np.delete(real_PD, note_list, axis=0)
        #     real_GT = np.delete(real_GT, note_list, axis=0)
        #     print("real_PD", real_PD.shape)
        #     print(real_GT.shape)
        # real_mae, real_mse, rrmse, rmape, rmspe, rrse, rcorr = metric(real_PD, real_GT) 
        real_mae, real_mse = metric(real_PD, real_GT) 
        print('rmse:{}, rmae:{}'.format(real_mse, real_mae))  
        # print('rmse:{:.3f}, rmae:{:.3f}'.format(real_mse, real_mae))  
        print('rmse / rmae:  {:.3f} / {:.3f}'.format(real_mse, real_mae))  


        individual_mse = []
        individual_mae = []
        for j in range(len(individual)-1):
            indiv_mae, indiv_mse = metric(real_PD[individual[j]: individual[j+1], :, :],
                                                                        real_GT[individual[j]: individual[j+1], :, :]) 
            individual_mse.append(indiv_mse)
            individual_mae.append(indiv_mae)

        individual_mse = ["{:.3f}".format(number) for number in individual_mse]
        individual_mae = ["{:.3f}".format(number) for number in individual_mae]
        print("indiv_mse", individual_mse)
        print("indiv_mae", individual_mae)
  
        return result_path
         


    def draw_sample(self, tar_col, result_path, GT_col, seq_len):
        df = pandas.read_csv(result_path)
        # GT_col = "original_weight" if self
        sample_len = self.args.seq_len + self.args.pred_len
        indv_errs = []
        id_l = []
        i=0
        while i<df.shape[0]:
            concurrent_ID = df.iloc[i][0]
            id_l.append(concurrent_ID)
            temp_df = df[df['ID']==concurrent_ID]
            i += temp_df.shape[0]

            temp_df = df[df['ID']==concurrent_ID][seq_len:]
            err_l = metrics.MAE(temp_df[tar_col].values, temp_df[GT_col].values)
            indv_errs.append(err_l)
        # print(indv_errs)
        mwe = np.mean(indv_errs)
        print("mmwwee", mwe, round(mwe, 3))



