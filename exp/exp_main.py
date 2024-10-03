import random
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import  iTransformer, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import utils.metrics as metrics
from torch.utils.tensorboard import SummaryWriter

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

    def _build_model(self):
        model_dict = {
            'NLinear': NLinear,
            'iTransformer': iTransformer,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()

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
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(batch_x)

                elif self.args.model == 'iTransformer':
                    outputs = self.model(batch_x, x_mark_enc=None)

                f_dim = -1
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
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

        path = os.path.join(self.args.checkpoints_root, self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                

                if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(batch_x)
                elif self.args.model == 'iTransformer':
                    outputs = self.model(batch_x, x_mark_enc=None)

                f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if self.args.features == 'M':
                    batch_y = batch_y[:, -self.args.pred_len:, -2:].to(self.device)  

                    vari = batch_y[:, :, :1].repeat(1, 1, outputs.shape[2] - 1)
                    batch_y = torch.cat((vari, batch_y[:, :, -1:]), axis=2)

                    m_list = []
                    meal_num = [self.args.breakfast, self.args.lunch, self.args.supper].count(1)
                    for l in range(0, meal_num):
                        m_list.append((outputs[:, :, l:l+1] - batch_y[:, :, l:l+1]) ** 2)

                    w = (outputs[:, :, -1:]-batch_y[:, :, -1:]) ** 2
                    L_weight = w
                    L_diet = (1/3) * (sum(m_list))
                    loss = lamda * L_weight + (1- lamda) * L_diet
                    loss = loss.sum() / (outputs.shape[0] * outputs.shape[1])

                else: 
                    loss = criterion(outputs, batch_y)  #loss for "S"

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

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

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, test=1):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints_root, self.args.checkpoints, setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'Linear' in self.args.model or 'TST' in self.args.model:
                    outputs = self.model(batch_x)

                elif self.args.model == 'iTransformer':
                    outputs = self.model(batch_x, x_mark_enc=None)

                f_dim = -1 if self.args.features == 'M' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x)
     
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = torch.concat(inputx, axis=0)

        index_dict = test_data.index_d

        print("------------------long term prediction --------------------")
        old_csv = './dataset/{}_test_weight.csv'.format(self.args.seq_len + self.args.pred_len)
        longterm_preds = self.longterm_predict(test_data, inputx)
        result_path = self.metric_pred(old_csv, "weight_longterm", longterm_preds, trues, index_dict)

        return


    def metric_pred(self, old_csv, tar_col, preds, trues, index_dict, strategy="mean"):# mean/max/min/first/last
        if not os.path.exists('./result'):
            os.makedirs('./result')

        f_dim = -1 if self.args.features == 'M' else 0
        real_PD = preds[:, -self.args.pred_len:, f_dim:]   
        real_GT = trues[:, -self.args.pred_len:, f_dim:] 

        assert strategy in ['mean', 'max', 'min', 'first', 'last']

        temp_d = dict()#record prediction results for each day, so then computing max/min/mean conveniently
        for i in range(real_PD.shape[0]):
            pred = real_PD[i]
            test_fi_index = index_dict[str(i)] + self.args.seq_len
            for j in range(self.args.pred_len):
                if str(test_fi_index + j) not in temp_d:
                    temp_d[str(test_fi_index + j)] = np.array([], dtype = np.float32)
                temp_d[str(test_fi_index + j)] = np.append(temp_d[str(test_fi_index + j)], pred[j][0])

        df = pandas.read_csv(old_csv)
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

        df[tar_col] = final_preds

        r_path =  "result/{}-{}".format(str(self.args.seq_len), str(self.args.pred_len))
        if self.args.features == "S":
            modal = "S"
        if self.args.image:
            modal = "img"
        if self.args.text and self.args.text_from_img:
            modal = "lmm"
            if self.args.fusion == "early":
                modal = "img_lmm_early"
        if self.args.text and not self.args.text_from_img:
            modal = "txt" 
            if self.args.fusion == "early":
                modal = "img_txt_early"
        os.makedirs(r_path, exist_ok=True)
        result_path = os.path.join(r_path, self.args.model + "_" + modal + "_"
                                     + strategy + '_food_' + str(self.args.breakfast) + str(self.args.lunch) + str(self.args.supper)+"_l_"+ str(self.args.lamda) + '.csv')
        if modal == "S":
            result_path = os.path.join(r_path, self.args.model + "_" + modal + "_" + strategy + '.csv')   


        print("result_path", result_path)
        if 'date' in df.columns:
            df = df.drop(columns=['date', 'breakfast', 'lunch', 'supper', 'variation'])
        df.to_csv(result_path, index=0)

        real_mae, real_mse = metric(real_PD, real_GT) 
        print('rmse:{}, rmae:{}'.format(real_mse, real_mae))  
        print('rmse / rmae:  {:.3f} / {:.3f}'.format(real_mse, real_mae))  

        return result_path
         


    """
    Long term forecast
    Food information follows a cycle of seq_1en 
    eg setting 3-3
    first 3 days food are ground-truth, food of 4-6 / 7-10 /... days are same as 1-3 days
    """
    def longterm_predict(self, test_data, inputx):
        i=0
        k=0
        outputs = []
        sample_len = self.args.seq_len + self.args.pred_len
        df_test = pandas.read_csv(os.path.join(test_data.root_path,
                    str(self.args.seq_len + self.args.pred_len) +'_test_weight.csv'))
        while i<df_test.shape[0]:
            current_ID = df_test.iloc[i][0]
            temp_df = df_test[df_test['ID']==current_ID]

            input = inputx[k].unsqueeze(0) 
            history_GT_food = inputx[k, :, :-2].unsqueeze(0)

            user_dict = dict()
            for j in range(input.shape[1]):
                if str(j + 1) not in user_dict:
                    user_dict[str(j + 1)] = []
                user_dict[str(j + 1)].append(input[0, j:j+1, -1])

            with torch.no_grad():
                food = history_GT_food
                for item in range(temp_df.shape[0] + 1 - sample_len):#一个individual能贡献的样本数
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        output = self.model(input)
                    elif self.args.model == "iTransformer":#itransformer
                        output = self.model(input, x_mark_enc=None)

                    f_dim = -1 if self.args.features == 'M' else 0
                    output = output[:, -self.args.pred_len:, f_dim:]
                    outputs.append(output.detach().cpu().numpy())

                    for j in range(output.shape[1]):
                        if str(input.shape[1] + item + 1 + j) not in user_dict:
                            user_dict[str(input.shape[1] + item + 1 + j)] = []
                        user_dict[str(input.shape[1] + item + 1 + j)].append(output[0, j:j+1, -1])

                    new_input = []
                    for j in range(item+2, item + 2 + input.shape[1]):
                        pred_value = sum(user_dict[str(j)])/len(user_dict[str(j)])
                        new_input.append(pred_value)
                    input = torch.stack(new_input, axis=0).unsqueeze(0)

                    if (self.args.features=='M' and self.args.image) or (self.args.features=='M' and self.args.text):
                        if k + item + 1 < inputx.shape[0]:
                            food = torch.cat((food[:, 1:, :], food[:, :1, :]), axis = 1)
                            input = torch.cat((food, input), axis=2)

                k += temp_df.shape[0] + 1 - sample_len
                i = i+temp_df.shape[0]

        outputs = np.concatenate(outputs, axis=0)

        return outputs