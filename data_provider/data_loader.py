import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
from PIL import Image
import clip
import json

warnings.filterwarnings('ignore')



class Dataset_Weight(Dataset):
    def __init__(self, root_path, image_root, flag='train', size=None,
                 features='S', data_path='data.csv', feature_path = '/share/ckpt/yxgui',
                 target='end_weight', train_only=False, image = 1, text = 1,
                  text_from_img=0, b=1, l=1, s=1):

        if size == None:
            self.seq_len = 3
            self.pred_len = 3
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]
        self.sample_len =  self.seq_len + self.pred_len
        self.flag = flag
        assert self.flag in ['train', 'val', 'test']
        self.type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = self.type_map[self.flag]

        self.features = features
        self.target = target
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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = self.get_clip()
        
        self.__split_data__()
        self.__read_data__()
        

    def get_clip(self):
        model, preprocess = clip.load("RN101", self.device)
        return model, preprocess
    
    def cmp(self, x):
        if x < self.sample_len:
            return 0
        if x >= self.sample_len:
            return 1
        
    def __get_raw_data__(self):
        #process original data and write into 'samplelen_total_weight.csv'
        total_weight_path = os.path.join(self.root_path,
                                          str(self.sample_len)+'_total_weight.csv')
        total = pd.read_csv(os.path.join(self.root_path, self.data_path))#read data.csv

        cols = list(('ID', 'date', 'breakfast', 'lunch', 'supper','start_weight', 'weight'))
        df = total[cols]
        df["variation"] = df['weight'] - df['start_weight'] 

        #record number of data of each user
        i=0
        d_dict = dict()
        while i<df.shape[0]:
            current_ID = df.iloc[i][0]
            temp_df = df[df['ID']==current_ID]
            i = i+temp_df.shape[0]
            d_dict[current_ID] = temp_df.shape[0]

        #sort the records according to relative value to sample_len
        #only users having records larger than sample_len will be add to dataset of setting 'seqlen_predlen'
        d_dict_sorted = sorted(d_dict.items(), key=lambda x: self.cmp(x[1]))

        #write the filted data into 'samplelen_total_weight.csv'
        new_df = pd.DataFrame()
        for item in d_dict_sorted:
            ID = item[0]
            temp_df = df[df['ID']==ID]
            new_df = pd.concat([new_df, temp_df], ignore_index=True)
        new_df.to_csv(total_weight_path, index=0)

        return total_weight_path


    def __split_data__(self):
        """
        split data.csv to samplelen_train.csv / samplelen_val.csv / samplelen_test.csv 
        """
        total_weight_path = self.__get_raw_data__() #get 'samplelen_total_weight.csv'
        df_raw = pd.read_csv(total_weight_path) 


        #df_raw.columns: ['date', ...(other features), target feature]
        i=0
        sum1 = 0 #total number of sample (eg: 10days records of a user can contribute (10+1-6)=5 samples for setting 3-3)
        while i<df_raw.shape[0]:
            current_ID = df_raw.iloc[i][0]
            temp_df = df_raw[df_raw['ID']==current_ID]
            
            if temp_df.shape[0] >= self.sample_len:
                sum1 += temp_df.shape[0] + 1 - self.sample_len
            i = i+temp_df.shape[0]

        #borders: bounding of split different sets for train/val/test
        self.borders=[0]*3
        self.sample_num = [0]*3
        i=0
        sample_sum = 0
        while i<df_raw.shape[0]:
            current_ID = df_raw.iloc[i][0]
            temp_df = df_raw[df_raw['ID']==current_ID]

            k = i+temp_df.shape[0]
            if temp_df.shape[0] < self.sample_len:
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

        #write train/val/test data into (train/val/test)_weight.csv
        for key in self.type_map:#train/val/test
            border1 = self.borders[self.type_map[key]]
            border2 = self.borders[self.type_map[key] + 1]

            df_part = df_raw[border1: border2]
            df_part.to_csv(os.path.join(self.root_path, str(self.sample_len) +
                                        '_{}_weight.csv'.format(key)), index=0)
            
    def __read_data__(self):
        df_train = pd.read_csv(os.path.join(self.root_path,
                            str(self.sample_len) + '_train_weight.csv'))

        df_origin = pd.read_csv(os.path.join(self.root_path,
                            str(self.sample_len) + '_{}_weight.csv'.format(self.flag)))

        i=0
        sample_sum = 0
        self.len = 0
        while i < df_origin.shape[0]:
            current_ID = df_origin.iloc[i][0]
            temp_df = df_origin[df_origin['ID']==current_ID]

            if temp_df.shape[0] >= self.sample_len:
                j = i
                for step in range(temp_df.shape[0] + 1 - self.sample_len):
                    self.index_d[str(sample_sum + step) ] = j 
                    j += 1
                sample_sum += temp_df.shape[0] + 1 - self.sample_len
            i += temp_df.shape[0]
        self.len = sample_sum

        cols = list(df_origin.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')
        cols.remove('start_weight')
        cols.remove('ID')
        cols.remove('breakfast')
        cols.remove('lunch')
        cols.remove('supper')
        cols.remove("variation")
        

        if self.features == 'M':
            df_origin = df_origin[['ID'] + ['date'] + ['variation'] + cols]
            cols_data = df_origin.columns[-2:]
            df_data = df_origin[cols_data]
        elif self.features == 'S':
            df_origin = df_origin[['ID'] + ['date'] + cols + [self.target]]
            df_data = df_origin[[self.target]]

        self.col = df_data.columns
        data = df_data.values

        self.data_x = data
        self.data_y = data
        
        if self.image and self.features == 'M':#image
            # print("img")
            meal_list_x = []
            if self.breakfast:
                breakfast_img_feature = self.get_img_feature("breakfast")
                meal_list_x.append(breakfast_img_feature)
            if self.lunch:
                lunch_img_feature = self.get_img_feature("lunch")
                meal_list_x.append(lunch_img_feature)
            if self.supper:
                supper_img_feature = self.get_img_feature("supper")
                meal_list_x.append(supper_img_feature)      
            img = np.concatenate(meal_list_x, axis=1)
            self.data_x = np.concatenate((img, self.data_x), axis=1)
            self.data_y = np.concatenate((img, self.data_y), axis=1)


        if self.text and self.features == 'M' and self.text_from_img == False:#ingredients text
            # print("txt")
            meal_list_x = []
            if self.breakfast:
                breakfast_txt_feature = self.get_txt_feature("breakfast")
                meal_list_x.append(breakfast_txt_feature)
            if self.lunch:
                lunch_txt_feature = self.get_txt_feature("lunch")
                meal_list_x.append(lunch_txt_feature)
            if self.supper:
                supper_txt_feature = self.get_txt_feature("supper")
                meal_list_x.append(supper_txt_feature)
            txt = np.concatenate(meal_list_x, axis=1)
            self.data_x = np.concatenate((txt, self.data_x), axis=1)
            self.data_y = np.concatenate((txt, self.data_y), axis=1)


        if self.text and self.features == 'M' and self.text_from_img:#ingredients from LMM
            meal_list_x = []
            if self.breakfast:
                breakfast_txt_feature = self.get_txt_feature_from_img("breakfast")
                meal_list_x.append(breakfast_txt_feature)
            if self.lunch:
                lunch_txt_feature = self.get_txt_feature_from_img("lunch")
                meal_list_x.append(lunch_txt_feature)
            if self.supper:
                supper_txt_feature = self.get_txt_feature_from_img("supper")
                meal_list_x.append(supper_txt_feature)           
            txt = np.concatenate(meal_list_x, axis=1)
            self.data_x = np.concatenate((txt, self.data_x), axis=1)
            self.data_y = np.concatenate((txt, self.data_y), axis=1)


    def __getitem__(self, index):
        real_index = self.index_d[str(index)]
        s_begin = real_index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin : s_end]
        seq_y = self.data_y[r_begin : r_end]

        return seq_x, seq_y

    #get feature of a meal, if multiple images, avarage the feature of each image
    def get_meal_feature(self, set, temp_df, k, meal):
        feature_meal_list = self.get_meal_feature_list(temp_df, k, meal)#feature list of a meal
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
        else:
            m = 1
            while len(feature_meal_list) == 0:
                feature_meal_list = self.get_meal_feature_list(temp_df, k - m, meal)
                m += 1
            img_feature = torch.stack(feature_meal_list,0)  
            if set == "train": 
                self.meal_lack += 1

        img_feature = torch.mean(img_feature,0)#[1,512]
        img_feature = img_feature.float()
    
        return img_feature  


    def get_txt_feature(self, meal):
        save_path = os.path.join(self.feature_path, 'LTSF-txt-npy')
        os.makedirs(save_path, exist_ok=True)
        npy_path = os.path.join(save_path, f"{self.sample_len}_{meal}.npy")

        self.meal_lack = 0 

        if not os.path.exists(npy_path):
            print("not exist file!")
            meal_list = []
           
            for set in ['train', 'val', 'test']:
                df = pd.read_csv(os.path.join(self.root_path,
                            str(self.sample_len) + '_{}_weight.csv'.format(set)))
                i = 0
                while i < df.shape[0]:
                    current_ID = df.iloc[i][0]
                    temp_df = df[df['ID']==current_ID]
                    if temp_df.shape[0] >= self.sample_len:

                        for k in range(temp_df.shape[0]):
                            meal_token = self.get_meal_token(temp_df, k, meal) #某一餐token list 可能为空
                            #meal_token is empty
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
                            
                            text_input = torch.cat(meal_token).to(self.device) 
                            with torch.no_grad():
                                text_feature = self.clip_model.encode_text(text_input)
                            text_feature = torch.mean(text_feature,0).unsqueeze(0).float()

                            meal_list.append(text_feature.cpu().numpy())       

                    i += temp_df.shape[0]

            meal_x = np.concatenate(meal_list, axis=0)#end train/val/test loop
            np.save(npy_path, meal_x)

        else:
            print("exist file!")

        #read feature from npy file
        meal_x = np.load(npy_path)
        start = self.borders[self.set_type] - self.borders[0]
        end = self.borders[self.set_type + 1] - self.borders[0]
        return meal_x[start: end]


    def get_txt_feature_from_img(self, meal):
        ingrs_img_path = "dataset/predict_ingr.json" # ingredients prediction from FoodLMM
        with open(ingrs_img_path) as f:
            ingrs_img_dict = json.load(f)

        save_path = save_path = os.path.join(self.feature_path, "LTSF-txt-from-img-npy")
        os.makedirs(save_path, exist_ok=True)
        npy_path = os.path.join(save_path, f"{self.sample_len}_{meal}_from_img.npy")
        self.meal_lack = 0 

        if not os.path.exists(npy_path):
            print("not exist file!")
            meal_list = []
            for set in ['train', 'val', 'test']:
                df = pd.read_csv(os.path.join(self.root_path,
                            str(self.sample_len) + '_{}_weight.csv'.format(set)))
                i = 0
                while i < df.shape[0]:
                    current_ID = df.iloc[i][0]
                    temp_df = df[df['ID']==current_ID]

                    if temp_df.shape[0] >= self.sample_len:

                        for k in range(temp_df.shape[0]):
                            meal_token = self.get_meal_token_from_img(ingrs_img_dict, temp_df, k, meal) #某一餐token list 可能为空

                            #meal_token is empty
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
                                
                            text_input = torch.cat(meal_token).to(self.device)
                            with torch.no_grad():
                                text_feature = self.clip_model.encode_text(text_input) 
                            text_feature = torch.mean(text_feature,0).unsqueeze(0).float()

                            meal_list.append(text_feature.cpu().numpy())       

                    i += temp_df.shape[0]

            meal_x = np.concatenate(meal_list, axis=0)
            np.save(npy_path, meal_x)

        else:
            print("exist file!")

        #read feature from npy file
        meal_x = np.load(npy_path)
        start = self.borders[self.set_type] - self.borders[0]
        end = self.borders[self.set_type + 1] - self.borders[0]
        return meal_x[start: end]   


    def get_img_feature(self, meal):
        save_path = os.path.join(self.feature_path, "LTSF-img-npy")
        os.makedirs(save_path, exist_ok=True)
        npy_path = os.path.join(save_path, f"{self.sample_len}_{meal}.npy")
        self.meal_lack = 0 #record number of missing data of a meal

        if not os.path.exists(npy_path):
            print("not exist file!")

            meal_list = []
            for set in ['train', 'val', 'test']:
                df = pd.read_csv(os.path.join(self.root_path,
                            str(self.sample_len) + '_{}_weight.csv'.format(set)))
                i = 0
                while i < df.shape[0]:
                    current_ID = df.iloc[i][0]
                    temp_df = df[df['ID']==current_ID]
                    if temp_df.shape[0] >= self.sample_len:
                        for k in range(temp_df.shape[0]):
                            meal_feature = self.get_meal_feature(set, temp_df, k, meal) #get feature of a meal
                            meal_list.append(meal_feature.cpu().numpy())

                    i += temp_df.shape[0]

            
            meal_x = np.concatenate(meal_list, axis=0)#train/val/test loop end
            np.save(npy_path, meal_x)#write into npy file

        else:
            print("exist file!")

        #read feature from npy file
        meal_x = np.load(npy_path)
        start = self.borders[self.set_type] - self.borders[0]
        end = self.borders[self.set_type + 1] - self.borders[0]
        return meal_x[start: end]

    def __len__(self):
        return self.len

    def get_meal_feature_list(self, temp_df, k, meal):#get feature list of a meal from images
        meal_list = self.get_img_addr(temp_df, k, meal)
        feature_meal_list=[]
        with torch.no_grad():
            for meal_addr in meal_list:
                try: 
                    img_open = Image.open(meal_addr)
                    if img_open.mode != 'RGB':
                        img_open = img_open.convert('RGB')
                    img = self.preprocess(img_open).unsqueeze(0).cuda()
                    img_feature = self.clip_model.encode_image(img)
                    feature_meal_list.append(img_feature)
                except IOError: 
                    pass
            return feature_meal_list
    
    def get_img_addr(self, temp_df, k, meal):
        meal_list = []
        addr, tag = str.split(temp_df.iloc[k][meal], "|||")
        for x in str.split(addr, ";"):
            if x!='':
                meal_list.append(os.path.join(self.image_root, 'DietDiary', x))
        return meal_list

    def get_meal_token(self, temp_df, k, meal):#get token list of a meal from ingredients text
        meal_list = self.get_annotation(temp_df, k, meal)
        text_inputs = [clip.tokenize(ingr) for ingr in meal_list]

        return text_inputs
    
    def get_annotation(self, temp_df, k, meal):
        meal_list = []
        addr, tag = str.split(temp_df.iloc[k][meal], "|||")
        for x in str.split(tag, " "):
            if x!='':
                meal_list.append(x)

        return meal_list
    
    def get_meal_token_from_img(self, dict, temp_df, k, meal):
        meal_list = self.get_img_addr(temp_df, k, meal)

        meal_ingrs = []
        for meal_addr in meal_list:
            meal_addr = meal_addr.split(self.image_root)[1]
            if meal_addr in dict:
                meal_ingrs.extend(dict[meal_addr])

        meal_ingrs = list(dict.fromkeys(meal_ingrs))#remove duplication
        text_inputs = [clip.tokenize(ingr) for ingr in meal_ingrs]

        return text_inputs
        
