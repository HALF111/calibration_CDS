import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        # 加上'all'的flag
        assert flag in ['train', 'test', 'val', 'all']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        # deal with 'all'
        if self.set_type == 3:
            border1 = border1s[0]
            border2 = border2s[-1]
        else:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
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

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        # add 'all'
        assert flag in ['train', 'test', 'val', 'all']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        if self.set_type == 3:
            border1 = border1s[0]
            border2 = border2s[-1]
        else:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'all']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'all': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        if self.set_type == 3:
            border1 = border1s[0]
            border2 = border2s[-1]
        else:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
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

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)





class Dataset_Custom_Test(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_nearest_data=False, use_further_data=False, adapt_start_pos = 1,  # 也别忘记use_nearest_data和adapt_start_pos参数
                 test_train_num=1):  # 别忘记加上test_train_num参数！！
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # 注意：别忘记要加这一句话！！
        self.test_train_num = test_train_num
        # 以及这一句话，加上use_nearest_data参数！！！
        self.use_nearest_data = use_nearest_data
        self.use_further_data = use_further_data
        self.adapt_start_pos = adapt_start_pos

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # 别忘记这里也要做修改！！
        # border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        if not self.use_nearest_data and not self.use_further_data:
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - (self.seq_len + self.pred_len + self.test_train_num - 1)]
        else:
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - (self.seq_len + self.test_train_num + self.adapt_start_pos - 1)]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
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

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if not self.use_nearest_data and not self.use_further_data:
            s_begin = index
            # s_end = s_begin + self.seq_len
            # 注意这里要做一些修改！！！
            s_end = s_begin + (self.seq_len + self.pred_len + self.test_train_num - 1)
            
            r_begin = s_end - self.label_len
            # r_end = r_begin + self.label_len + self.pred_len
            r_end = s_end + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        elif self.use_nearest_data:
            s_begin = index
            s_end = s_begin + (self.seq_len + self.pred_len + self.test_train_num - 1)
            
            # 这里先算r_end再算r_begin了
            r_end = s_end + self.adapt_start_pos
            r_begin = r_end - self.pred_len - self.label_len

            seq_x = self.data_x[s_begin:s_end]  # (201, 7) ->（seq_len + pred_len + ttn - 1, channel）
            seq_y = self.data_y[r_begin:r_end]  # (144, 7) -> (label_len + pred_len, channel)
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            import copy
            seq_x = copy.deepcopy(seq_x)
            seq_x_mark = copy.deepcopy(seq_x_mark)
            if self.adapt_start_pos < self.pred_len:  # 只有当start_pos小于pred_len时，才需要对后面补零
                seq_x[-(self.pred_len-self.adapt_start_pos):, :] = 0  # 将seq_x最后面的(self.pred_len-self.adapt_start_pos)部分的值置为0
                seq_x_mark[-(self.pred_len-self.adapt_start_pos):, :] = 0  # 同样也将seq_x_mark最后面的(self.pred_len-self.adapt_start_pos)部分的值置为0
        else:  # self.use_further_data
            s_begin = index
            s_mid = s_begin + (self.seq_len + self.pred_len + self.test_train_num - 1)
            s_end = s_mid + (self.adapt_start_pos - self.pred_len)
            
            r_begin = s_end - self.label_len
            r_end = s_end + self.pred_len

            seq_x = self.data_x[s_begin:s_end]  # (201, 7) ->（seq_len + pred_len + ttn - 1, channel）
            seq_y = self.data_y[r_begin:r_end]  # (144, 7) -> (label_len + pred_len, channel)
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        # 注意：这里的__len__也需要做修改！！！
        # 不然取数据的时候会超出范围
        # return len(self.data_x) - (self.seq_len + self.pred_len + self.test_train_num - 1) - self.pred_len + 1
        
        # 这里还进一步需要根据use_nearest_data来绝对长度大小
        if not self.use_nearest_data and not self.use_further_data:
            return (len(self.data_x) - (self.pred_len + self.test_train_num - 1)) - self.seq_len - self.pred_len + 1
        else:
            return (len(self.data_x) - (self.test_train_num + self.adapt_start_pos - 1)) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour_Test(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 use_nearest_data=False, use_further_data=False, adapt_start_pos = 1,  # 也别忘记use_nearest_data和adapt_start_pos参数
                 test_train_num=1):  # 别忘记加上test_train_num参数！！
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # 注意：别忘记要加这一句话！！！
        self.test_train_num = test_train_num
        # 以及这一句话，加上use_nearest_data参数！！！
        self.use_nearest_data = use_nearest_data
        self.use_further_data = use_further_data
        self.adapt_start_pos = adapt_start_pos

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # 这里最好也做一次修改！！
        # border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        if not self.use_nearest_data and not self.use_further_data:
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - (self.seq_len + self.pred_len + self.test_train_num - 1)]
        else:  # 无论use_nearest_data还是use_further_data都是这么取数据
            border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - (self.seq_len + self.test_train_num + self.adapt_start_pos - 1)]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
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

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        # # 改成把s_begin往左移
        # # s_begin = index
        # s_begin = index - (self.pred_len + self.test_train_num - 1)
        # s_end = index + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len

        if not self.use_nearest_data and not self.use_further_data:
            s_begin = index
            # s_end = s_begin + self.seq_len
            # 注意这里要做一些修改！！！
            s_end = s_begin + (self.seq_len + self.pred_len + self.test_train_num - 1)
            
            r_begin = s_end - self.label_len
            # r_end = r_begin + self.label_len + self.pred_len
            r_end = s_end + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        elif self.use_nearest_data:
            s_begin = index
            s_end = s_begin + (self.seq_len + self.pred_len + self.test_train_num - 1)
            
            # 这里先算r_end再算r_begin了
            r_end = s_end + self.adapt_start_pos
            r_begin = r_end - self.pred_len - self.label_len

            seq_x = self.data_x[s_begin:s_end]  # (201, 7) ->（seq_len + pred_len + ttn - 1, channel）
            seq_y = self.data_y[r_begin:r_end]  # (144, 7) -> (label_len + pred_len, channel)
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            import copy
            seq_x = copy.deepcopy(seq_x)
            seq_x_mark = copy.deepcopy(seq_x_mark)
            if self.adapt_start_pos < self.pred_len:  # 只有当start_pos小于pred_len时，才需要对后面补零
                seq_x[-(self.pred_len-self.adapt_start_pos):, :] = 0  # 将seq_x最后面的(self.pred_len-self.adapt_start_pos)部分的值置为0
                seq_x_mark[-(self.pred_len-self.adapt_start_pos):, :] = 0  # 同样也将seq_x_mark最后面的(self.pred_len-self.adapt_start_pos)部分的值置为0
        else:  # self.use_further_data
            s_begin = index
            s_mid = s_begin + (self.seq_len + self.pred_len + self.test_train_num - 1)
            s_end = s_mid + (self.adapt_start_pos - self.pred_len)
            
            r_begin = s_end - self.label_len
            r_end = s_end + self.pred_len

            seq_x = self.data_x[s_begin:s_end]  # (201, 7) ->（seq_len + pred_len + ttn - 1, channel）
            seq_y = self.data_y[r_begin:r_end]  # (144, 7) -> (label_len + pred_len, channel)
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    # 即：用当前数据能构造出几个样本
    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        # 注意：这里的__len__也需要做修改！！！
        # 不然取数据的时候会超出范围
        # return len(self.data_x) - (self.seq_len + self.pred_len + self.test_train_num - 1) - self.pred_len + 1
        
        # 这里还进一步需要根据use_nearest_data或use_further_data来修改长度大小
        if not self.use_nearest_data and not self.use_further_data:
            return (len(self.data_x) - (self.pred_len + self.test_train_num - 1)) - self.seq_len - self.pred_len + 1
        else:
            return (len(self.data_x) - (self.test_train_num + self.adapt_start_pos - 1)) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute_Test(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 use_nearest_data=False, use_further_data=False, adapt_start_pos = 1,  # 也别忘记use_nearest_data和adapt_start_pos参数
                 test_train_num=1):  # 别忘记加上test_train_num参数！！
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path

        # 注意：别忘记要加这一句话！！！
        self.test_train_num = test_train_num
        # 以及这一句话，加上use_nearest_data参数！！！
        self.use_nearest_data = use_nearest_data
        self.use_further_data = use_further_data
        self.adapt_start_pos = adapt_start_pos

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # 这里最好也做一次修改！！
        # border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        if not self.use_nearest_data and not self.use_further_data:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - (self.seq_len + self.pred_len + self.test_train_num - 1)]
        else:
            border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - (self.seq_len + self.test_train_num + self.adapt_start_pos - 1)]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if not self.use_nearest_data and not self.use_further_data:
            s_begin = index
            # s_end = s_begin + self.seq_len
            # 注意这里要做一些修改！！！
            s_end = s_begin + (self.seq_len + self.pred_len + self.test_train_num - 1)
            
            r_begin = s_end - self.label_len
            # r_end = r_begin + self.label_len + self.pred_len
            r_end = s_end + self.pred_len

            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        elif self.use_nearest_data:
            s_begin = index
            s_end = s_begin + (self.seq_len + self.pred_len + self.test_train_num - 1)
            
            # 这里先算r_end再算r_begin了
            r_end = s_end + self.adapt_start_pos
            r_begin = r_end - self.pred_len - self.label_len

            seq_x = self.data_x[s_begin:s_end]  # (201, 7) ->（seq_len + pred_len + ttn - 1, channel）
            seq_y = self.data_y[r_begin:r_end]  # (144, 7) -> (label_len + pred_len, channel)
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

            import copy
            seq_x = copy.deepcopy(seq_x)
            seq_x_mark = copy.deepcopy(seq_x_mark)
            if self.adapt_start_pos < self.pred_len:  # 只有当start_pos小于pred_len时，才需要对后面补零
                seq_x[-(self.pred_len-self.adapt_start_pos):, :] = 0  # 将seq_x最后面的(self.pred_len-self.adapt_start_pos)部分的值置为0
                seq_x_mark[-(self.pred_len-self.adapt_start_pos):, :] = 0  # 同样也将seq_x_mark最后面的(self.pred_len-self.adapt_start_pos)部分的值置为0
        else:  # self.use_further_data
            s_begin = index
            s_mid = s_begin + (self.seq_len + self.pred_len + self.test_train_num - 1)
            s_end = s_mid + (self.adapt_start_pos - self.pred_len)
            
            r_begin = s_end - self.label_len
            r_end = s_end + self.pred_len

            seq_x = self.data_x[s_begin:s_end]  # (201, 7) ->（seq_len + pred_len + ttn - 1, channel）
            seq_y = self.data_y[r_begin:r_end]  # (144, 7) -> (label_len + pred_len, channel)
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        # 注意：这里的__len__也需要做修改！！！
        # 不然取数据的时候会超出范围
        # return len(self.data_x) - (self.seq_len + self.pred_len + self.test_train_num - 1) - self.pred_len + 1
        
        # 这里还进一步需要根据use_nearest_data来绝对长度大小
        if not self.use_nearest_data and not self.use_further_data:
            return (len(self.data_x) - (self.pred_len + self.test_train_num - 1)) - self.seq_len - self.pred_len + 1
        else:
            return (len(self.data_x) - (self.test_train_num + self.adapt_start_pos - 1)) - self.seq_len - self.pred_len + 1


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


