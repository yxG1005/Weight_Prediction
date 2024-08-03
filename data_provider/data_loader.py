import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from utils.timefeatures import time_features
import warnings
from PIL import Image
import clip
import json

warnings.filterwarnings('ignore')



class Dataset_Weight(Dataset):
    def __init__(self, root_path, image_root, flag='train', size=None,
                 features='S', data_path='new_8_all_feat.csv', feature_path = '/share/ckpt/yxgui',
                 target='end_weight', scale=1, timeenc=0, freq='h', train_only=False, image = 1, text = 1,
                  text_from_img = 0, b=1, l=1, s=1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 5
            self.label_len = 5
            self.pred_len = 5
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.sample_len =  self.seq_len + self.pred_len
        self.real_sample_len = self.seq_len + self.pred_len
        # init
        self.flag = flag
        assert self.flag in ['train', 'val', 'test']
        self.type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = self.type_map[self.flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.image = image
        self.text = text
        self.text_from_img = text_from_img
        self.breakfast = b
        self.lunch = l
        self.supper = s

        self.image_root = image_root
        self.root_path = root_path
        self.data_path = data_path
        self.feature_path = feature_path
        self.index_d = dict()
        


        self.__split_data__()
        self.__read_data__()


    
    def cmp(self, x):
        if x < self.sample_len:
            return 0
        if x >= self.sample_len:
            return 1
        
    def __get_raw_data__(self):
        #对原有数据进行一定处理后写入 samplelen_total_weight.csv
        total_weight_path = os.path.join(self.root_path,
                                          str(self.sample_len)+'_total_weight.csv')
        # if not os.path.exists(total_weight_path):
        # print("数据集0,",self.data_path)
        total = pd.read_csv(os.path.join(self.root_path, self.data_path))#anno_6.csv 总数据集

        # le = LabelEncoder()
        # total['sex'] = le.fit_transform(total['sex'])

        cols = list(('ID', 'date', 'breakfast', 'lunch', 'supper','start_weight', 'weight'))
        df = total[cols]
        df["variation"] = df['weight'] - df['start_weight'] #变化值列

        # print(df_new)
        # print(cols)

        i=0
        d_dict = dict()
        while i<df.shape[0]:
            concurrent_ID = df.iloc[i][0]
            temp_df = df[df['ID']==concurrent_ID]
            i = i+temp_df.shape[0]
            
            d_dict[concurrent_ID] = temp_df.shape[0]
            

        d_dict_sorted = sorted(d_dict.items(), key=lambda x: self.cmp(x[1]))
        # print(d_dict_sorted)


        new_df_2 = pd.DataFrame()
        for item in d_dict_sorted:
            ID = item[0]
            temp_df = df[df['ID']==ID]
            new_df_2 = pd.concat([new_df_2, temp_df], ignore_index=True)

        new_df_2.to_csv(total_weight_path, index=0)

        return total_weight_path


    def __split_data__(self):
        """split to samplelen_train.csv / samplelen_val.csv / samplelen_test.csv 
        """

        total_weight_path = self.__get_raw_data__()
        print("total_weight_path",total_weight_path)

        df_raw = pd.read_csv(total_weight_path) #对个人信息进行分桶处理
        # print("分桶处理后", df_raw[['height'] + ['age']])

        if self.scale:
            df_raw = self.instance_norm(df_raw, 2, -8)
            # print("标准化后", df_raw)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        i=0
        sum1 = 0
        while i<df_raw.shape[0]:
            concurrent_ID = df_raw.iloc[i][0]
            temp_df = df_raw[df_raw['ID']==concurrent_ID]
            
            if temp_df.shape[0] >= self.sample_len:
                sum1 += temp_df.shape[0] + 1 - self.sample_len
            i = i+temp_df.shape[0]

        # print("sum1", sum1)

        self.borders=[0]*3
        self.sample_num = [0]*3
        img_list =  []
        i=0
        sample_sum = 0
        while i<df_raw.shape[0]:
            concurrent_ID = df_raw.iloc[i][0]
            temp_df = df_raw[df_raw['ID']==concurrent_ID]

            k = i+temp_df.shape[0]
            if temp_df.shape[0] < self.sample_len:
                # print(concurrent_ID, temp_df.shape[0])
                self.borders[0] = k
            if temp_df.shape[0] >= self.sample_len:
                sample_sum += temp_df.shape[0]+1-self.sample_len
                if sample_sum <= sum1 * 0.7 and sample_sum + df_raw[df_raw['ID']==df_raw.iloc[k][0]].shape[0] > sum1 * 0.7:
                    self.borders[1] = k
                    self.sample_num[1] = sample_sum
                if sample_sum <= sum1 * 0.8 and sample_sum + df_raw[df_raw['ID']==df_raw.iloc[k][0]].shape[0] > sum1 * 0.8:
                    self.borders[2] = k
                    self.sample_num[2] = sample_sum
            
            i = i+temp_df.shape[0]

        
        self.borders.append(df_raw.shape[0])
        self.sample_num.append(sample_sum)

        # print("borders", self.borders)
        # print("sample_num",self.sample_num)

        for key in self.type_map:#train/val/test
            print("key", key)
            border1 = self.borders[self.type_map[key]]
            border2 = self.borders[self.type_map[key] + 1]

            df_part = df_raw[border1: border2]
            df_part.to_csv(os.path.join(self.root_path, str(self.sample_len) +
                                        '_{}_weight_{}.csv'.format(key, self.scale)), index=0)

    


    def get_meal_feature(self, set, temp_df, k, meal):#从图片获取某一餐的feature并取平均
        
        feature_meal_list = self.get_meal_feature_list(temp_df, k, meal)#某一餐的feature_list

        if len(feature_meal_list)!=0:
            img_feature = torch.stack(feature_meal_list,0)
        elif k == 0:
            m = 1
            while len(feature_meal_list) == 0:
                feature_meal_list = self.get_meal_feature_list(temp_df, k + m, meal)
                m += 1
            img_feature = torch.stack(feature_meal_list,0)
            if set == "train": 
                self.meal_lack += 1
                print(self.meal_lack)
        else:
            m = 1
            while len(feature_meal_list) == 0:
                feature_meal_list = self.get_meal_feature_list(temp_df, k - m, meal)
                m += 1
            img_feature = torch.stack(feature_meal_list,0)  
            if set == "train": 
                self.meal_lack += 1
                print(self.meal_lack)


        img_feature = torch.mean(img_feature,0)#[1,512]
        img_feature = img_feature.float()
    
        return img_feature  


    
    #获取某一餐的在目前数据集（train/val/test）的所有特征 并存储在 sample_len-flag-meal.pkl中
    def get_txt_feature(self, meal):
        save_path = os.path.join(self.feature_path, 'LTSF-txt-npy')
        os.makedirs(save_path, exist_ok=True)
        npy_path = os.path.join(save_path, f"{self.sample_len}_{meal}.npy")
        # npy_path = os.path.join(save_path, f"{16}_{meal}.npy")
        self.meal_lack = 0 #记录某一餐总缺失数

        #如果不存在这个文件, 读取这种sample长度的所有该餐进文件
        if not os.path.exists(npy_path):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('RN101', device)

            meal_list = []#存放这一餐的所有feanture的list
            # train_empty_sum = 0#计数测试的时候空的token
            #循环train/val/test 数据集
            for set in ['train', 'val', 'test']:
                df = pd.read_csv(os.path.join(self.root_path,
                            str(self.sample_len) + '_{}_weight_{}.csv'.format(set, self.scale)))
                p = 0#校验
                i = 0
                while i < df.shape[0]:
                    concurrent_ID = df.iloc[i][0]
                    temp_df = df[df['ID']==concurrent_ID]
                    print('\n',concurrent_ID)
                    if temp_df.shape[0] >= self.sample_len:

                        for k in range(temp_df.shape[0]):
                            p += 1
                            meal_token = self.get_meal_token(temp_df, k, meal) #某一餐token list 可能为空

                            #为空的处理
                            if len(meal_token) == 0:
                                if k == 0:
                                    m = 1
                                    while len(meal_token) == 0:
                                        meal_token = self.get_meal_token(temp_df, k + m, meal)
                                        m += 1
                                else:
                                    m = 1
                                    while len(meal_token) == 0:
                                        meal_token = self.get_meal_token(temp_df, k - m, meal)
                                        m += 1
                                if set == "train": 
                                    self.meal_lack += 1
                                    print(self.meal_lack)

                            
                            text_input = torch.cat(meal_token).to(device) # 原料数 × 77
                            # print("text_input", text_input.shape)
                            with torch.no_grad():
                                text_feature = model.encode_text(text_input) #原料数 × 512 
                                # print("text_feature", text_feature.shape)
                            text_feature = torch.mean(text_feature,0).unsqueeze(0).float() #1 × 512
                            # print("text_feature", text_feature.shape)

                            meal_list.append(text_feature.cpu().numpy())       

                    i += temp_df.shape[0]
                # print("p", p)

            
            #三个集循环结束
            # print("{}的长度".format(meal), len(meal_list))
            meal_x = np.concatenate(meal_list, axis=0)
            # print("{}的形状".format(meal), meal_x.shape) #按理来说是 长度 * 512

            #存入npy文件中
            np.save(npy_path, meal_x)

        else:
            print("存在文件！")

        #从文件中读取特征
        meal_x = np.load(npy_path)
        # print("load以后{}的形状".format(meal), meal_x.shape) #按理来说是 长度 * 512

        #返回的是某一餐按照train val test 的顺序 排的所有特征  形状应该是 行数 × 512
        # print("self.set_type",self.set_type)
        # print("borders", self.borders)
        start = self.borders[self.set_type] - self.borders[0]
        end = self.borders[self.set_type + 1] - self.borders[0]
        # print("start", start)
        # print("end", end)
        # if self.set_type == 2:
        #     start += (11- self.seq_len) * 3
        #meal_x[self.border1 - self.borders[0]: self.border2 - self.borders[0]]
        return meal_x[start: end]

    #做完纯text的实验以后，需要做用大模型从图片得到ingredients，然后作为食物信息
    # 这个函数是得到所有的文本特征
    def get_txt_feature_from_img(self, meal):
        #load 图片地址：图片 对应的ingredients 字典
        ingrs_img_path = "dataset/predict_ingr.json"
        with open(ingrs_img_path) as f:
            ingrs_img_dict = json.load(f)

        save_path = save_path = os.path.join(self.feature_path, "LTSF-txt-from-img-npy")
        os.makedirs(save_path, exist_ok=True)
        npy_path = os.path.join(save_path, f"{self.sample_len}_{meal}_from_img.npy")
        # npy_path = os.path.join(save_path, f"{16}_{meal}_from_img.npy")
        self.meal_lack = 0 #记录某一餐总缺失数

        #如果不存在这个文件, 读取这种sample长度的所有该餐进文件
        if not os.path.exists(npy_path):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('RN101', device)

            meal_list = []#存放这一餐的所有feanture的list

            #循环train/val/test 数据集
            for set in ['train', 'val', 'test']:
                df = pd.read_csv(os.path.join(self.root_path,
                            str(self.sample_len) + '_{}_weight_{}.csv'.format(set, self.scale)))
                print(f"现在在读取{set}的{meal}")
                p = 0#校验
                i = 0
                while i < df.shape[0]:
                    concurrent_ID = df.iloc[i][0]
                    temp_df = df[df['ID']==concurrent_ID]

                    if temp_df.shape[0] >= self.sample_len:

                        for k in range(temp_df.shape[0]):
                            p += 1
                            meal_token = self.get_meal_token_from_img(ingrs_img_dict, temp_df, k, meal) #某一餐token list 可能为空

                            #为空的处理
                            if len(meal_token) == 0:
                                if k == 0:
                                    m = 1
                                    while len(meal_token) == 0:
                                        meal_token = self.get_meal_token_from_img(ingrs_img_dict, temp_df, k + m, meal)
                                        m += 1
                                else:
                                    m = 1
                                    while len(meal_token) == 0:
                                        meal_token = self.get_meal_token_from_img(ingrs_img_dict, temp_df, k - m, meal)
                                        m += 1
                                if set == "train": 
                                    self.meal_lack += 1
                                    print(self.meal_lack)
                                
                            #用文本编码器编码成feature
                            text_input = torch.cat(meal_token).to(device) # 原料数 × 77
                            # print("text_input", text_input.shape)
                            with torch.no_grad():
                                text_feature = model.encode_text(text_input) #原料数 × 512 
                                # print("text_feature", text_feature.shape)
                            text_feature = torch.mean(text_feature,0).unsqueeze(0).float() #1 × 512
                            # print("text_feature", text_feature.shape)

                            meal_list.append(text_feature.cpu().numpy())       

                    i += temp_df.shape[0]
                # print("p", p)

            
            #三个集循环结束
            meal_x = np.concatenate(meal_list, axis=0)

            #存入npy文件中
            np.save(npy_path, meal_x)

        else:
            print("exist file!")

        #从文件中读取特征
        meal_x = np.load(npy_path)
        # print("meal_x", meal_x)
        # print("load以后{}的形状".format(meal), meal_x.shape) #按理来说是 长度 * 512

        #返回的是某一餐按照train val test 的顺序 排的所有特征  形状应该是 行数 × 512
        # print("self.set_type",self.set_type)
        # print("borders", self.borders)
        start = self.borders[self.set_type] - self.borders[0]
        end = self.borders[self.set_type + 1] - self.borders[0]
        # print("start", start)
        # print("end", end)
        # if self.set_type == 2:
        #     start += (11- self.seq_len) * 3
        #meal_x[self.border1 - self.borders[0]: self.border2 - self.borders[0]]
        return meal_x[start: end]   

    #获取某一餐的在目前数据集（train/val/test）的所有图片特征 并存储在 sample_len-flag-meal.pkl中
    #
    def get_img_feature(self, meal):
        save_path = save_path = os.path.join(self.feature_path, "LTSF-img-npy")
        os.makedirs(save_path, exist_ok=True)
        npy_path = os.path.join(save_path, f"{self.sample_len}_{meal}.npy")
        # npy_path = os.path.join(save_path, f"{16}_{meal}.npy")
        self.meal_lack = 0 #记录某一餐总缺失数

        #如果不存在这个文件, 读取这种sample长度的所有该餐进文件
        if not os.path.exists(npy_path):

            meal_list = []#存放这一餐的所有feanture的list

            #循环train/val/test 数据集
            for set in ['train', 'val', 'test']:
                df = pd.read_csv(os.path.join(self.root_path,
                            str(self.sample_len) + '_{}_weight_{}.csv'.format(set, self.scale)))
                p = 0#校验
                i = 0
                while i < df.shape[0]:
                    concurrent_ID = df.iloc[i][0]
                    temp_df = df[df['ID']==concurrent_ID]
                    print('\n',concurrent_ID)
                    if temp_df.shape[0] >= self.sample_len:

                        for k in range(temp_df.shape[0]):
                            p += 1
                            #得到某餐的feature
                            meal_feature = self.get_meal_feature(set, temp_df, k, meal)
                            meal_list.append(meal_feature.cpu().numpy())

                    i += temp_df.shape[0]
                # print("p", p)

            
            #三个集循环结束
            print("{}的长度".format(meal), len(meal_list))
            meal_x = np.concatenate(meal_list, axis=0)
            print("{}的形状".format(meal), meal_x.shape) #按理来说是 长度 * 512

            #存入npy文件中
            np.save(npy_path, meal_x)

        else:
            print("exist file!")

        #从文件中读取特征
        meal_x = np.load(npy_path)
        # print("load以后{}的形状".format(meal), meal_x.shape) #按理来说是 长度 * 512

        #返回的是某一餐按照train val test 的顺序 排的所有特征  形状应该是 行数 × 512
        # print("self.set_type",self.set_type)
        # print("borders", self.borders)
        start = self.borders[self.set_type] - self.borders[0]
        end = self.borders[self.set_type + 1] - self.borders[0]
        # print("start", start)
        # print("end", end)
        # if self.set_type == 2:
        #     start += (11- self.seq_len) * 3
        #meal_x[self.border1 - self.borders[0]: self.border2 - self.borders[0]]
        return meal_x[start: end]


    def __read_data__(self):
        self.scaler = StandardScaler()
        self.mm_scalar = MinMaxScaler()
        df_train = pd.read_csv(os.path.join(self.root_path,
                            str(self.sample_len) + '_train_weight_{}.csv'.format(self.scale)))

        df_origin = pd.read_csv(os.path.join(self.root_path,
                            str(self.sample_len) + '_{}_weight_{}.csv'.format(self.flag, self.scale)))

      
        # print("现在是", self.flag)
        print(self.index_d)


        i=0
        sample_sum = 0
        self.len = 0
        while i < df_origin.shape[0]:
            concurrent_ID = df_origin.iloc[i][0]
            temp_df = df_origin[df_origin['ID']==concurrent_ID]

            if temp_df.shape[0] >= self.sample_len:
                j = i
                for step in range(temp_df.shape[0] + 1 - self.real_sample_len):
                    self.index_d[str(sample_sum + step) ] = j 
                    j += 1
                sample_sum += temp_df.shape[0] + 1 - self.real_sample_len

            i += temp_df.shape[0]
        self.len = sample_sum
        print(self.index_d)


        cols = list(df_origin.columns)
        if self.features == 'S':
            print(cols)
            print(self.target)
            cols.remove(self.target)
        # if self.features == 'M' or self.features == 'MS':
        #     if not self.user_profile:
        #         cols.remove('height')
        #         cols.remove('age')
        #         cols.remove('sex')

        cols.remove('date')
        cols.remove('start_weight')
        cols.remove('ID')
        cols.remove('breakfast')
        cols.remove('lunch')
        cols.remove('supper')
        cols.remove("variation")
        

        if self.features == 'M' or self.features == 'MS':
            df_origin = df_origin[['ID'] + ['date'] + ['variation'] + cols]
            cols_data = df_origin.columns[-2:]
            df_data = df_origin[cols_data]
            train_data = df_train[cols_data]
            print("df_data", df_data)
        elif self.features == 'S':
            df_origin = df_origin[['ID'] + ['date'] + cols + [self.target]]
            df_data = df_origin[[self.target]]
            train_data = df_train[[self.target]]

        self.col = df_data.columns

        data = df_data.values


        # #min/max 标准化身高、体重、年龄
        # if self.user_profile:
        #     train_profile = train_data[["height"]+["age"]+["sex"]]
        #     df_profile = df_data[["height"]+["age"]+["sex"]]
        #     self.mm_scalar.fit(train_profile.values)
        #     profile = self.mm_scalar.transform(df_profile.values)
        #     data = data[:, -1:]
        #     data = np.concatenate((profile, data), axis=1)

        
        # print("total_data", data, data.shape)

        df_stamp = df_origin[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # df_raw.insert(loc=3, column='weight_norm', value=data)


        self.data_x = data
        self.data_y = data
        # print("img_x",border1-borders[0], border2-borders[0])
        
        if self.image and self.features == 'M':
            print("img")
            meal_list_x = []
            if self.breakfast:
                breakfast_img_feature = self.get_img_feature("breakfast")
                # print(f"breakfast {self.flag}", breakfast_img_feature.shape, type(breakfast_img_feature))
                # print("breakfast 总缺失数", self.meal_lack)
                meal_list_x.append(breakfast_img_feature)
            if self.lunch:
                lunch_img_feature = self.get_img_feature("lunch")
                # print(f"lunch {self.flag}", lunch_img_feature.shape, type(lunch_img_feature))
                # print("lunch 总缺失数", self.meal_lack)
                meal_list_x.append(lunch_img_feature)
            if self.supper:
                supper_img_feature = self.get_img_feature("supper")
                # print(f"supper {self.flag}", supper_img_feature.shape, type(supper_img_feature))
                # print("supper 总缺失数", self.meal_lack)
                meal_list_x.append(supper_img_feature)      
            img = np.concatenate(meal_list_x, axis=1)
            self.data_x = np.concatenate((img, self.data_x), axis=1)
            self.data_y = np.concatenate((img, self.data_y), axis=1)


        #文本特征
        if self.text and self.features == 'M' and self.text_from_img == False:
            print("txt")
            meal_list_x = []
            if self.breakfast:
                breakfast_txt_feature = self.get_txt_feature("breakfast")
                # print(f"breakfast {self.flag}", breakfast_txt_feature.shape, type(breakfast_txt_feature))
                # print("breakfast 总缺失数", self.meal_lack)
                meal_list_x.append(breakfast_txt_feature)
            if self.lunch:
                lunch_txt_feature = self.get_txt_feature("lunch")
                # print(f"lunch {self.flag}", lunch_txt_feature.shape,  type(lunch_txt_feature))
                # print("lunch 总缺失数", self.meal_lack)
                meal_list_x.append(lunch_txt_feature)
            if self.supper:
                supper_txt_feature = self.get_txt_feature("supper")
                # print(f"supper {self.flag}", supper_txt_feature.shape,  type(supper_txt_feature))
                # print("supper 总缺失数", self.meal_lack)
                meal_list_x.append(supper_txt_feature)
            txt = np.concatenate(meal_list_x, axis=1)
            self.data_x = np.concatenate((txt, self.data_x), axis=1)
            self.data_y = np.concatenate((txt, self.data_y), axis=1)

        #文本特征 从LMM得到 得到文本特征的函数调用get_txt_feature_from_img
        if self.text and self.features == 'M' and self.text_from_img:
            meal_list_x = []
            if self.breakfast:
                breakfast_txt_feature = self.get_txt_feature_from_img("breakfast")
                # print(f"breakfast {self.flag}", breakfast_txt_feature.shape, type(breakfast_txt_feature))
                # print("breakfast 总缺失数", self.meal_lack)
                meal_list_x.append(breakfast_txt_feature)
            if self.lunch:
                lunch_txt_feature = self.get_txt_feature_from_img("lunch")
                # print(f"lunch {self.flag}", lunch_txt_feature.shape,  type(lunch_txt_feature))
                # print("lunch 总缺失数", self.meal_lack)
                meal_list_x.append(lunch_txt_feature)
            if self.supper:
                supper_txt_feature = self.get_txt_feature_from_img("supper")
                # print(f"supper {self.flag}", supper_txt_feature.shape,  type(supper_txt_feature))
                # print("supper 总缺失数", self.meal_lack)
                meal_list_x.append(supper_txt_feature)           
            txt = np.concatenate(meal_list_x, axis=1)
            self.data_x = np.concatenate((txt, self.data_x), axis=1)
            self.data_y = np.concatenate((txt, self.data_y), axis=1)


        print("data_x", self.data_x.shape, type(self.data_x))
        # print("data_y", type(self.data_y))
        self.data_stamp = data_stamp


    def __getitem__(self, index):

        
        real_index = self.index_d[str(index)]

        # print("real_index", real_index)

        s_begin = real_index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len
        # print("s_begin", s_begin, "s_end", s_end)
        # print("r_begin", r_begin, "r_end", r_end)
        seq_x = self.data_x[s_begin : s_end]
        seq_y = self.data_y[r_begin : r_end]
        # seq_img_x = self.img_x[s_begin : s_end]
        # seq_img_y = self.img_y[r_begin : r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
      



        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_message(self):
        return self.index_d, self.sample_num[self.set_type]
    
    def get_col(self):
        return self.col
    
    def get_scale(self):
        return self.scale 

    def get_meal_feature_list(self, temp_df, k, meal):#get feature list of a meal from images
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess =clip.load("RN101", device=device)
        model = model
        meal_list = self.get_img_addr(temp_df, k, meal)

        
        feature_meal_list=[]#将图片转化为tensor然后转化为feature存进这个数组
        with torch.no_grad():
            for meal_addr in meal_list:
                try: 
                    img_open = Image.open(meal_addr)
                    if img_open.mode != 'RGB':
                        img_open = img_open.convert('RGB')
                    img = preprocess(img_open).unsqueeze(0).cuda()
                    img_feature = model.encode_image(img)
                    # print("img_feature", type(img_feature), img_feature.shape)
                    feature_meal_list.append(img_feature)
                except IOError: 
                    pass

            # print("feature_meal_list",len(feature_meal_list))

            return feature_meal_list
    
    def get_img_addr(self, temp_df, k, meal):
        meal_list = []
        addr, tag = str.split(temp_df.iloc[k][meal], "|||")
        for x in str.split(addr, ";"):
            if x!='':
                meal_list.append(os.path.join(self.image_root, 'DietDiary', x))
        return meal_list

    def get_meal_token(self, temp_df, k, meal):#从文本获取某一顿餐（meal）的token list
        #拿到某一餐的标注 有可能是空数组（无标注）
        meal_list = self.get_annotation(temp_df, k, meal)
        print("k={}天的{}有几种食材".format(k, meal), len(meal_list))
        #转化为token list : text_inputs
        text_inputs = [clip.tokenize(ingr) for ingr in meal_list]

        return text_inputs
    
    def get_annotation(self, temp_df, k, meal):#返回一行的某一餐标注
        meal_list = []
        addr, tag = str.split(temp_df.iloc[k][meal], "|||")
        for x in str.split(tag, " "):
            if x!='':
                meal_list.append(x)

        return meal_list
    
    #从图像用FoodLMM得到ingredients列表，然后转成token. dict是图像地址-ingredients字典
    def get_meal_token_from_img(self, dict, temp_df, k, meal):
        #拿到某一餐的地址list
        meal_list = self.get_img_addr(temp_df, k, meal)
        print("k={}天的{}有多少张图".format(k, meal), len(meal_list))
        
        #一餐所有ingredients都存进这个数组
        meal_ingrs = []
        for meal_addr in meal_list:
            print(meal_addr)
            meal_addr = meal_addr.split(self.image_root)[1]
            if meal_addr in dict:#字典里有这个图的ingredients
                meal_ingrs.extend(dict[meal_addr])#合并一餐的所有图的ingredients列表
        print("meal_ingrs", meal_ingrs)

        #对得到的ingredients数组去重
        meal_ingrs = list(dict.fromkeys(meal_ingrs))

        # meal_ingrs里是一餐的标注(从图片得到) 有可能是空数组（无标注）
        print("k={}天的{}有几种食材".format(k, meal), len(meal_ingrs))
        #转化为token list : text_inputs
        text_inputs = [clip.tokenize(ingr) for ingr in meal_ingrs]

        return text_inputs
        
