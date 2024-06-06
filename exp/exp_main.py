from data_provider.data_factory import data_provider
from data_provider.data_factory import data_provider_at_test_time
from exp.exp_basic import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer
from models.etsformer import ETSformer
from models.crossformer import Crossformer
from models import Linear, DLinear, NLinear
from models import PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

from layers.RevIN import RevIN

# ETSformer
from utils.Adam import Adam

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim

import os
import time
import warnings


import copy
import math


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        self.test_train_num = self.args.test_train_num

        data_path = self.args.data_path
        # adapt_part_channel is an optional choice
        # if set, we only fine-tune part of channels in the dataset, instead of all channels
        if "ETTh1" in data_path: selected_channels = [1,3, 2,4,5,6]
        elif "ETTh2" in data_path: selected_channels = [7, 1,3]
        elif "ETTm1" in data_path: selected_channels = [1,3, 2,4,5]
        elif "ETTm2" in data_path: selected_channels = [1,7, 3]
        elif "illness" in data_path: selected_channels = [1,2, 3,4,5]
        elif "weather" in data_path: selected_channels = [17,18,19, 5,8,6,13,20]
        else: selected_channels = list(range(1, self.args.c_out))  # all channels
        # change channel to index
        for channel in range(len(selected_channels)):
            selected_channels[channel] -= 1 
        
        self.selected_channels = selected_channels
        
        
        if "ETTh1" in data_path: period = 24
        elif "ETTh2" in data_path: period = 24
        elif "ETTm1" in data_path: period = 96
        elif "ETTm2" in data_path: period = 96
        elif "electricity" in data_path: period = 24
        elif "traffic" in data_path: period = 24
        elif "illness" in data_path: period = 52.142857
        elif "weather" in data_path: period = 144
        elif "Exchange" in data_path: period = 1
        elif "WTH_informer" in data_path: period = 24
        else: period = 1
        self.period = period


    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'ETSformer': ETSformer,
            'Crossformer': Crossformer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        
        if self.args.model == 'Crossformer':
            model = Crossformer.Model(
                self.args.enc_in, 
                self.args.seq_len, 
                self.args.pred_len,
                self.args.seg_len,
                self.args.win_size,
                self.args.cross_factor,
                self.args.d_model, 
                self.args.d_ff,
                self.args.n_heads, 
                self.args.e_layers,
                self.args.dropout, 
                self.args.baseline,
                self.device
            ).float()
        else:
            model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    def _get_data_at_test_time(self, flag):
        data_set, data_loader = data_provider_at_test_time(self.args, flag)
        return data_set, data_loader


    def _select_optimizer(self):
        if self.args.model == 'ETSformer':
            if 'warmup' in self.args.lradj: lr = self.args.min_lr
            else: lr = self.args.learning_rate

            if self.args.smoothing_learning_rate > 0: smoothing_lr = self.args.smoothing_learning_rate
            else: smoothing_lr = 100 * self.args.learning_rate

            if self.args.damping_learning_rate > 0: damping_lr = self.args.damping_learning_rate
            else: damping_lr = 100 * self.args.learning_rate

            nn_params = []
            smoothing_params = []
            damping_params = []
            for k, v in self.model.named_parameters():
                if k[-len('_smoothing_weight'):] == '_smoothing_weight':
                    smoothing_params.append(v)
                elif k[-len('_damping_factor'):] == '_damping_factor':
                    damping_params.append(v)
                else:
                    nn_params.append(v)

            model_optim = Adam([
                {'params': nn_params, 'lr': lr, 'name': 'nn'},
                {'params': smoothing_params, 'lr': smoothing_lr, 'name': 'smoothing'},
                {'params': damping_params, 'lr': damping_lr, 'name': 'damping'},
            ])
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
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
                        if 'Linear' in self.args.model or self.args.model == 'Crossformer' or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or self.args.model == 'Crossformer' or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                # Is it necessary to get last [-self.args.pred_len:]?
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
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

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
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or self.args.model == 'Crossformer' or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or self.args.model == 'Crossformer' or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # whether get the last part with length 'pred_len'?
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
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
                    if self.args.model == 'ETSformer': torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        
        # map_location='cuda:0'?
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))

        return self.model

    def test(self, setting, test=0, flag='test'):
        # test_data, test_loader = self._get_data(flag='test')
        test_data, test_loader = self._get_data(flag=flag)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        preds = []
        trues = []

        criterion = nn.MSELoss()
        loss_list = []

        test_time_start = time.time()
        
        prediction_time = 0

        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # recording time
                prediction_time_start = time.time()
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or self.args.model == 'Crossformer' or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or self.args.model == 'Crossformer' or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # whether get the last part with length 'pred_len'?
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # selected_channels = self.selected_channels
                if self.args.adapt_part_channels:
                    outputs = outputs[:, :, self.selected_channels]
                    batch_y = batch_y[:, :, self.selected_channels]

                # MSE loss
                loss = criterion(outputs, batch_y)
                loss_list.append(loss.item())

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                
                # recording time
                prediction_time_end = time.time()
                prediction_time += prediction_time_end - prediction_time_start

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")
        
        print(f"Prediction cost time: {prediction_time}")

        # return
        return loss_list


    def get_data_error(self, setting, test=0):
        print('loading model from checkpoint !!!')
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
        
        assert self.args.batch_size == 1

        for flag in ["train_without_shuffle", "val_without_shuffle", "test"]:
            cur_data, cur_loader = self._get_data(flag=flag)

            test_time_start = time.time()
            
            results = []
            residuals = []

            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(cur_loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    pred, true = self._process_one_batch_with_model(self.model, cur_data,
                        batch_x, batch_y, 
                        batch_x_mark, batch_y_mark)

                    # reshape
                    pred = pred.detach().cpu().numpy()
                    pred = pred.reshape(pred.shape[1], pred.shape[2])
                    true = true.detach().cpu().numpy()
                    true = true.reshape(true.shape[1], true.shape[2])
                    
                    residual = pred - true
                    residuals.append(residual)
                    
                    mae, mse, rmse, mape, mspe = metric(pred, true)
                    # print('mse:{}, mae:{}'.format(mse, mae))
                    
                    error = pred - true
                    # print(error.shape)
                    err_mean = np.mean(error)
                    err_var = np.var(error)
                    err_abs_mean = np.mean(np.abs(error))
                    err_abs_var = np.var(np.abs(error))
                    pos_num, neg_num = 0, 0
                    for ei in range(error.shape[0]):
                        for ej in range(error.shape[1]):
                            if error[ei][ej] >= 0: pos_num += 1
                            else: neg_num += 1
                    assert pos_num + neg_num == error.shape[0] * error.shape[1]
                    
                    tmp_list = [mae, mse, rmse, mape, mspe, err_mean, err_var, err_abs_mean, err_abs_var, pos_num, neg_num]
                    results.append(tmp_list)
                    
                    if i % 100 == 0:
                        print(f"data {i} have been calculated, cost time: {time.time() - test_time_start}s")
                        print('mse:{}, mae:{}'.format(mse, mae))
            

            # result save
            dataset_name = self.args.data_path.split(".")[0]
            dir_path = f"./error_results/{dataset_name}_{self.args.seq_len}_{self.args.pred_len}_{self.args.model}/"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            if "train" in flag: file_flag = "train"
            elif "val" in flag: file_flag = "val"
            elif "test" in flag: file_flag = "test"
            
            file_name = dir_path + f"pl{self.args.pred_len}_{file_flag}.txt"
            residual_file_name = dir_path + f"residuals_pl{self.args.pred_len}_{file_flag}.npy"
            
            with open(file_name, "w") as f:
                for result in results:
                    for idx in range(len(result)-1):
                        item = result[idx]
                        f.write(f"{item}, ")
                    f.write(f"{result[-1]}")
                    f.write("\n")
            
            residuals = np.array(residuals)
            np.save(residual_file_name, residuals)
        
        return


    def select_with_distance(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1, weights_given=None, adapted_degree="small", weights_from="test"):
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        data_len = len(test_data)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []

        RECORD_LOSS = True
        
        a1, a2, a3, a4 = [], [], [], []
        # 4 arrays correspond to: loss_before_adapt, loss_selected_samples, loss_selected_sample_adapted, loss_after_adapt
        all_angels = []
        all_distances = []

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        criterion = nn.MSELoss()
        test_time_start = time.time()
        
        # # result save
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # load model params
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        if weights_given:
            print(f"The given weights is {weights_given}")
            print(f"The length of given weights is {len(weights_given)}")
        

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # manually drop last batch
            if data_len - i <= data_len % self.args.batch_size: break
            
            # copy from 'self.model' to 'cur_model'
            cur_model = copy.deepcopy(self.model)
            # cur_model.train()
            cur_model.eval()

            if is_training_part_params:
                params = []
                names = []
                params_norm = []
                names_norm = []
                cur_model.requires_grad_(False)
                
                # Need personalization!
                linear_map = {
                    "ETSformer": ["decoder.pred"],
                    "Crossformer": [f"decoder.decode_layers.{self.args.e_layers}.linear_pred"],
                    "Linear": ["Linear"],
                    "PatchTST": ["model.head.linear"],
                    "default": ["decoder.projection"],
                }
                
                if self.args.model in linear_map:
                    linear_name_list = linear_map[self.args.model]
                else:
                    linear_name_list = linear_map["default"]
                    
                
                # # TODO:这里改成map形式，用if-else-if太冗余了！！！
                # if self.args.model == 'ETSformer':
                #     linear_layer_name = "decoder.pred"
                # elif self.args.model == 'Crossformer':
                #     # There are e_layers+1 of layers in decoder in original Crossformer, so the last layer is self.args.e_layers
                #     linear_layer_name = f"decoder.decode_layers.{self.args.e_layers}.linear_pred"
                # elif "Linear" in self.args.model:
                #     linear_layer_name = "Linear"
                # elif "PatchTST" in self.args.model:
                #     linear_layer_name = "model.head.linear"
                # else:
                #     linear_layer_name = "decoder.projection"
                
                # change string to list
                if isinstance(linear_name_list, str):
                    linear_name_list = [linear_name_list]
                
                # print(linear_name_list)
                # print(cur_model)
                
                # traverse and get params:
                for n_m, m in cur_model.named_modules():
                    for linear_name in linear_name_list:
                        if linear_name in n_m:
                            m.requires_grad_(True)
                            for n_p, p in m.named_parameters():
                                if n_p in ['weight', 'bias']:  # weight means scale, bias means shift
                                    params.append(p)
                                    names.append(f"{n_m}.{n_p}")

                # set lr:
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # changing Adam to SGD optimizer:
                model_optim = optim.SGD(params, lr=lr)  # model_optim = optim.Adam(params, lr=lr)
            else:
                cur_model.requires_grad_(True)
                # set lr:
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # changing Adam to SGD optimizer:
                model_optim = optim.SGD(cur_model.parameters(), lr=lr)  # model_optim = optim.Adam(cur_model.parameters(), lr=lr)
            
            # 1. loss before adaptation:
            # cur_model.eval()
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if not self.args.use_nearest_data or self.args.use_further_data:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -seq_len:, :], batch_y, 
                    batch_x_mark[:, -seq_len:, :], batch_y_mark)
            else:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                    batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]
            loss_before_adapt = criterion(pred, true)
            a1.append(loss_before_adapt.item())
            # cur_model.train()
            
            # 2. get gradient with ground-truth
            if use_adapted_model:
                seq_len = self.args.seq_len
                pred_len = self.args.pred_len
                adapt_start_pos = self.args.adapt_start_pos
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred_answer, true_answer = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred_answer, true_answer = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            
            if self.args.adapt_part_channels:
                pred_answer = pred_answer[:, :, self.selected_channels]
                true_answer = true_answer[:, :, self.selected_channels]
            loss_ans_before = criterion(pred_answer, true_answer)
            loss_ans_before.backward()

            w_T = params[0].grad.T
            b = params[1].grad.unsqueeze(0)
            params_answer = torch.cat((w_T, b), 0)
            # get gradient with ground-truth, including W and b.
            params_answer = params_answer.ravel()

            model_optim.zero_grad()  # clear gradient


            # channels without significant periodicity
            unselected_channels = list(range(self.args.c_out))
            for item in self.selected_channels:
                unselected_channels.remove(item)

            import torch.nn.functional as F
            if self.args.adapt_part_channels:  
                test_x = batch_x[:, -seq_len:, self.selected_channels].reshape(-1)
            else:
                test_x = batch_x[:, -seq_len:, :].reshape(-1)
            
            distance_pairs = []
            
            if self.args.remove_nearest:
                for ii in range(self.test_train_num):
                    if self.args.adapt_part_channels:
                        lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                    else:
                        lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                    dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                    distance_pairs.append([ii, dist])
                # randomly select samples with the number of 'selected_data_num'
                import random
                selected_distance_pairs = random.sample(distance_pairs, self.args.selected_data_num)
            else:
                for ii in range(self.args.test_train_num):
                    # remove_cycle is used for ablation study
                    if not self.args.remove_cycle:
                        # phase difference between current sample and test sample
                        if 'illness' in self.args.data_path:
                            import math
                            cycle_remainer = math.fmod(self.args.test_train_num-1 + self.args.pred_len - ii, self.period)
                        else:
                            cycle_remainer = (self.args.test_train_num-1 + self.args.pred_len - ii) % self.period
                        # threshold for decision
                        threshold = self.period * self.args.lambda_period
                        if cycle_remainer > threshold or cycle_remainer < -threshold:
                            continue
                        
                    if self.args.adapt_part_channels:
                        lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                    else:
                        lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                    
                    # calculate sample similarity
                    dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                    distance_pairs.append([ii, dist])
                
                if not self.args.remove_distance:
                    # select samples with higher similarity
                    cmp = lambda item: item[1]
                    distance_pairs.sort(key=cmp)
                else:
                    # select samples with closer timestep difference
                    cmp = lambda item: item[0]
                    distance_pairs.sort(key=cmp, reverse=True)

                # select 'selected_data_num' samples
                selected_distance_pairs = distance_pairs[:self.args.selected_data_num]
                
            selected_indices = [item[0] for item in selected_distance_pairs]
            selected_distances = [item[1] for item in selected_distance_pairs]
            all_distances.append(selected_distances)

            cur_grad_list = []

            for epoch in range(test_train_epochs):

                gradients = []
                accpted_samples_num = set()

                # num_of_loss_per_update = 1
                mean_loss = 0

                for ii in selected_indices:

                    model_optim.zero_grad()

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                        batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])

                    if self.args.adapt_part_channels:
                        pred = pred[:, :, self.selected_channels]
                        true = true[:, :, self.selected_channels]
                    
                    # check whether use nearest data?
                    if not self.args.use_nearest_data or self.args.use_further_data:
                        loss = criterion(pred, true)
                    else:
                        data_used_num = (self.test_train_num - (ii+1)) + self.args.adapt_start_pos
                        if data_used_num < pred_len:
                            loss = criterion(pred[:, :data_used_num, :], true[:, :data_used_num, :])
                        else:
                            loss = criterion(pred, true)

                    # loss = criterion(pred, true)
                    mean_loss += loss

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        w_T = params[0].grad.T
                        b = params[1].grad.unsqueeze(0)
                        params_tmp = torch.cat((w_T, b), 0)
                        original_shape = params_tmp.shape
                        params_tmp = params_tmp.ravel()

                        # store the gradient in 'cur_grad_list'
                        cur_grad_list.append(params_tmp.detach().cpu().numpy())

                        model_optim.zero_grad()
            
            
            # function: multiply gradient by different weights
            def calc_weighted_params(params, weights):
                results = 0
                for i in range(len(params)):
                    results += params[i] * weights[i]
                return results
            
            # multiply gradient by different weights 
            if weights_given:
                weighted_params = calc_weighted_params(cur_grad_list, weights_given)
            else:
                weights_all_ones = [1 for i in range(self.test_train_num)]
                weighted_params = calc_weighted_params(cur_grad_list, weights_all_ones)
            
            # tranform 'weighted_params' from np.array to tensor
            weighted_params = torch.tensor(weighted_params)
            weighted_params = weighted_params.to(self.device)


            # calculate the angle between the gradients of ground-truth and the weighted sum of current gradients
            import math
            product = torch.dot(weighted_params, params_answer)
            product = product / (torch.norm(weighted_params) * torch.norm(params_answer))
            angel = math.degrees(math.acos(product))
            all_angels.append(angel)
            

            # change 'weighted_params' back to 'w_grad' and 'b_grad'
            weighted_params = weighted_params.reshape(original_shape)
            w_grad_T, b_grad = torch.split(weighted_params, [weighted_params.shape[0]-1, 1])
            w_grad = w_grad_T.T  # (7, 512)
            b_grad = b_grad.squeeze(0)  # (7)


            # update new param as: original param + gradient
            from torch.nn.parameter import Parameter
            cur_lr = self.args.learning_rate * self.args.adapted_lr_times

            # make gradients on unselected channels as 0
            if self.args.adapt_part_channels:
                w_grad[unselected_channels, :] = 0
                b_grad[unselected_channels] = 0

            # Note: params should minus gradient, instead of adding gradients!
            for linear_name in linear_name_list:
                cur_linear = f"cur_model.{linear_name}" if self.args.model != "Crossformer" else f"cur_model.decoder.decode_layers[{self.args.e_layers}].linear_pred"
                # manually GD on weight and bias of prediction layer
                exec(f"{cur_linear}.weight = Parameter({cur_linear}.weight - w_grad * cur_lr)")
                exec(f"{cur_linear}.bias = Parameter({cur_linear}.bias - b_grad * cur_lr)")
                            
            # if self.args.model == 'ETSformer':
            #     cur_model.decoder.pred.weight = Parameter(cur_model.decoder.pred.weight - w_grad * cur_lr)
            #     cur_model.decoder.pred.bias = Parameter(cur_model.decoder.pred.bias - b_grad * cur_lr)
            # elif self.args.model == 'Crossformer':
            #     # There are e_layers+1 of layers in decoder in original Crossformer, so the last layer is self.args.e_layers
            #     adapt_layer = cur_model.decoder.decode_layers[self.args.e_layers].linear_pred
            #     adapt_layer.weight = Parameter(adapt_layer.weight - w_grad * cur_lr)
            #     adapt_layer.bias = Parameter(adapt_layer.bias - b_grad * cur_lr)
            # elif "Linear" in self.args.model:
            #     cur_model.Linear.weight = Parameter(cur_model.Linear.weight - w_grad * cur_lr)
            #     cur_model.Linear.bias = Parameter(cur_model.Linear.bias - b_grad * cur_lr)
            # elif "PatchTST" in self.args.model:
            #     cur_model.model.head.linear.weight = Parameter(cur_model.model.head.linear.weight - w_grad * cur_lr)
            #     cur_model.model.head.linear.bias = Parameter(cur_model.model.head.linear.bias - b_grad * cur_lr)
            # else:
            #     cur_model.decoder.projection.weight = Parameter(cur_model.decoder.projection.weight - w_grad * cur_lr)
            #     cur_model.decoder.projection.bias = Parameter(cur_model.decoder.projection.bias - b_grad * cur_lr)

            # calculate mean loss
            mean_loss = mean_loss / self.args.selected_data_num
            a2.append(mean_loss.item())


            seq_len = self.args.seq_len
            label_len = self.args.label_len
            pred_len = self.args.pred_len
            tmp_loss = 0
            for ii in selected_indices:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                    batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])
                if self.args.adapt_part_channels:
                    pred = pred[:, :, self.selected_channels]
                    true = true[:, :, self.selected_channels]
                tmp_loss += criterion(pred, true)
            tmp_loss = tmp_loss / self.args.selected_data_num
            a3.append(tmp_loss.item())
            a3.append(0)


            # record loss after adaptation
            cur_model.eval()
            
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if use_adapted_model:
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)

            # if only adpat part channels, then do filtering
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]
                
            # store pred/true after adaptation
            pred_after_adapt = pred.detach().cpu().clone().numpy()
            true_after_adapt = true.detach().cpu().clone().numpy()

            # loss after adaptation
            loss_after_adapt = criterion(pred, true)
            a4.append(loss_after_adapt.item())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())


            if (i+1) % 100 == 0 or (data_len - i) <= 100 and (i+1) % 10 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                print(gradients)
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
                
                avg_1, avg_2, avg_3, avg_4 = 0, 0, 0, 0
                avg_angel = 0
                num = len(a1)
                for iii in range(num):
                    avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]; avg_4 += a4[iii]; avg_angel += all_angels[iii]
                avg_1 /= num; avg_2 /= num; avg_3 /= num; avg_4 /= num; avg_angel /= num
                print("1.before_adapt, 2.adapt_sample, 3.adapt_sample_after, 4.after_adapt, 5.angel_between_answer")
                print("average:", avg_1, avg_2, avg_3, avg_4, avg_angel)
                print("last one:", a1[-1], a2[-1], a3[-1], a4[-1], all_angels[-1])

                printed_selected_channels = [item+1 for item in self.selected_channels]
                print(f"adapt_part_channels: {self.args.adapt_part_channels}")
                print(f"remove_distance: {self.args.remove_distance}, remove_cycle: {self.args.remove_cycle}, remove_nearest: {self.args.remove_nearest}")
                print(f"first 25th selected_channels: {printed_selected_channels[:25]}")
                print(f"selected_distance_pairs are: {selected_distance_pairs}")

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()


        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")

        return mse, mae



    def select_with_distance_caching(self, setting, test=0, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1, weights_given=None, adapted_degree="small", weights_from="test"):
        # Note: we have to read all data once!!
        tmp_all_data, tmp_all_loader = self._get_data(flag='all')
        
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        # test_data, test_loader = self._get_data(flag='test')
        data_len = len(test_data)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
        
        self.model.eval()
        
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len
        
        
        # batch_x_lst_1 = []
        # mid_embeddings_1 = []
        # seq_len = self.args.seq_len
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tmp_test_loader):
        #     pred, true, mid_embedding = self._process_one_batch_with_model(self.model, test_data,
        #             batch_x[:, -seq_len:, :], batch_y, 
        #             batch_x_mark[:, -seq_len:, :], batch_y_mark, return_mid_embedding=True)
        #     batch_x_lst_1.append(batch_x)
        #     mid_embeddings_1.append(mid_embedding)
        # batch_x_lst_2 = []
        # mid_embeddings_2 = []
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        #     pred, true, mid_embedding = self._process_one_batch_with_model(self.model, test_data,
        #             batch_x[:, -seq_len:, :], batch_y, 
        #             batch_x_mark[:, -seq_len:, :], batch_y_mark, return_mid_embedding=True)
        #     batch_x_lst_2.append(batch_x)
        #     mid_embeddings_2.append(mid_embedding)
            
            
        # batch_x_lst_1 = []
        # mid_embeddings_1 = []
        # seq_len = self.args.seq_len
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tmp_test_loader):
        #     pred, true, mid_embedding = self._process_one_batch_with_model(self.model, test_data,
        #             batch_x[:, -seq_len:, :], batch_y, 
        #             batch_x_mark[:, -seq_len:, :], batch_y_mark, return_mid_embedding=True)
        #     batch_x_lst_1.append(batch_x)
        #     mid_embeddings_1.append(mid_embedding)
        # batch_x_lst_2 = []
        # mid_embeddings_2 = []
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        #     pred, true, mid_embedding = self._process_one_batch_with_model(self.model, test_data,
        #             batch_x[:, -seq_len:, :], batch_y, 
        #             batch_x_mark[:, -seq_len:, :], batch_y_mark, return_mid_embedding=True)
        #     batch_x_lst_2.append(batch_x)
        #     mid_embeddings_2.append(mid_embedding)
            
        preds = []
        trues = []

        a1, a2, a3, a4 = [], [], [], []
        all_angels = []
        all_distances = []

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        criterion = nn.MSELoss()
        test_time_start = time.time()
        
        backward_time = 0
        construction_time = 0
        adaptation_time = 0
        prediction_time = 0
        rest_time = 0

        # # result save
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # load model params
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
        self.model.eval()

        if weights_given:
            print(f"The given weights is {weights_given}")
            print(f"The length of given weights is {len(weights_given)}")
        
        
        # as one model will have different outputs under inputs with different batch_size
        # we need to ensure that batch_size equals to 1
        assert self.args.all_data_batch_size == 1
        
        # traverse train/val/test for inference, and pre-cache the mid_embedding
        # TODO: need personalize according to the used model
        all_mid_embeddings = []
        all_trend_parts = []
        all_revin_means, all_revin_stdevs = [], []
        all_batch_x_lst = []
        file_dir = "./all_mid_embeddings"
        PREFIX = "RevIN_" if self.args.add_revin else ""
        file_name = file_dir + f"/{PREFIX}{setting}.npy"
        trend_file_name = file_dir + f"/{PREFIX}{setting}_trend.npy"
        revin_mean_file_name = file_dir + f"/{PREFIX}{setting}_revin_mean.npy"
        revin_stdev_file_name = file_dir + f"/{PREFIX}{setting}_revin_stdev.npy"
        if os.path.exists(file_name):
            all_mid_embeddings = np.load(file_name)
            if self.args.model != "Informer":
                all_trend_parts = np.load(trend_file_name)
            if self.args.add_revin:
                # read mean and stdev in numpy.array format, and change to tensor
                all_revin_means = np.load(revin_mean_file_name)
                all_revin_stdevs = np.load(revin_stdev_file_name)
        else:
            # shape of each mid_embedding: [sample_num, seq_len, d_model]
            # Note: you should better use batch_size=1 for here.
            # Since we find that one model will have different outputs under inputs with different batch_size
            # A possible explanation is that batch_size will have impact on padding
            # Reference：https://www.zhihu.com/question/295911908 
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tmp_all_loader):
                pred, true, mid_embedding, *res = self._process_one_batch_with_model(self.model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark, return_mid_embedding=True)
                if self.args.model == "Informer":
                    if self.args.add_revin: revin_mean, revin_stdev = res
                    else: pass
                else:
                    if self.args.add_revin: trend_part, revin_mean, revin_stdev = res
                    else: trend_part = res

                # store into list
                all_mid_embeddings.append(mid_embedding.detach().cpu().clone().numpy())
                if self.args.model != "Informer":
                    all_trend_parts.append(trend_part.detach().cpu().clone().numpy())
                if self.args.add_revin:
                    all_revin_means.append(revin_mean.detach().cpu().clone().numpy())
                    all_revin_stdevs.append(revin_stdev.detach().cpu().clone().numpy())
                all_batch_x_lst.append(batch_x.detach().cpu().clone().numpy())
        
            # save into the file
            all_mid_embeddings = np.vstack(all_mid_embeddings)
            np.save(file_name, all_mid_embeddings)
            if self.args.model != "Informer":
                all_trend_parts = np.vstack(all_trend_parts)
                np.save(trend_file_name, all_trend_parts)
            if self.args.add_revin:
                all_revin_means = np.vstack(all_revin_means)
                np.save(revin_mean_file_name, all_revin_means)
                all_revin_stdevs = np.vstack(all_revin_stdevs)
                np.save(revin_stdev_file_name, all_revin_stdevs)
            all_batch_x_lst = np.vstack(all_batch_x_lst)
        
        # get data length of each segments
        # manually divide
        all_len = len(tmp_all_data)
        print("all_len:", all_len)
        original_len = all_len + seq_len + pred_len - 1
        if "ETT" in self.args.data:
            train_len = int(original_len * 0.6)
            test_len = int(original_len * 0.2)
            vali_len = original_len - train_len - test_len
        else:
            train_len = int(original_len * 0.7)
            test_len = int(original_len * 0.2)
            vali_len = original_len - train_len - test_len
        
        print("all_mid_embeddings.shape", all_mid_embeddings.shape)
        assert all_mid_embeddings.shape[0] == all_len
        
        # get params of prediction layer
        linear_params = []
        linear_names = []
        for n_m, m in self.model.named_modules():
            if self.args.model == 'ETSformer':
                linear_layer_name = "decoder.pred"
            elif self.args.model == 'Crossformer':
                # There are e_layers+1 of layers in decoder in original Crossformer, so the last layer is self.args.e_layers
                linear_layer_name = f"decoder.decode_layers.{self.args.e_layers}.linear_pred"
            elif "Linear" in self.args.model:
                linear_layer_name = "Linear"
            elif "PatchTST" in self.args.model:
                linear_layer_name = "model.head.linear"
            else:
                linear_layer_name = "decoder.projection"
            
            if linear_layer_name in n_m:
                for n_p, p in m.named_parameters():
                    if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                        linear_params.append(p)
                        linear_names.append(f"{n_m}.{n_p}")
        
        # define a linear projection model, and only fine-tune it
        linear_model = nn.Linear(self.args.d_model, self.args.c_out, bias=True)
        linear_model.requires_grad_(True)
        linear_model.weight = linear_params[0]
        linear_model.bias = linear_params[1]
        
        # self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            # manually drop last batch
            if data_len - i <= data_len % self.args.batch_size: break
            
            # get corresponding mid_embedding at first
            # Since the start point of prediction window for the first sample in test set starts from train_len + vali_len
            # So there will be 'train_len + vali_len - seq_len' samples before that first sample.
            # So START_INDEX should be train_len + vali_len - seq_len
            START_INDEX = train_len + vali_len - seq_len
            cur_mid_embedding = all_mid_embeddings[START_INDEX + i : START_INDEX + i + 1]
            cur_trend_part = all_trend_parts[START_INDEX + i : START_INDEX + i + 1]
            cur_revin_mean = all_revin_means[START_INDEX + i : START_INDEX + i + 1]
            cur_revin_stdev = all_revin_stdevs[START_INDEX + i : START_INDEX + i + 1]
            
            # copy from original linear_model
            cur_model = copy.deepcopy(linear_model)
            cur_model.eval()
            
            
            backward_time_start = time.time()

            # get params
            params = []
            names = []
            cur_model.requires_grad_(False)
            for n_m, m in cur_model.named_modules():
                m.requires_grad_(True)
                for n_p, p in m.named_parameters():
                    if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{n_m}.{n_p}")

            # lr and optimizer
            lr = self.args.learning_rate * self.args.adapted_lr_times
            model_optim = optim.SGD(params, lr=lr)
            
            
            # ? Please uncomment this part if need to check the correctness of the code
            # pred_0, true_0, mid_embedding_0, trend_part_0 = self._process_one_batch_with_model(self.model, test_data,
            #         batch_x[:, -seq_len:, :], batch_y, 
            #         batch_x_mark[:, -seq_len:, :], batch_y_mark, return_mid_embedding=True)
            
            # found = False
            # for idx in range(all_mid_embeddings.shape[0]):
            #     if (all_mid_embeddings[idx:idx+1] == mid_embedding_0.detach().cpu().clone().numpy()).all():
            #         found = True
            #         print("equal index:", idx)
            # if not found:
            #     print("does not find the same mid_embedding")    
            #     print("does not find the same mid_embedding")    

            # tmp loss
            # cur_model.eval()
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            # adapt_start_pos = self.args.adapt_start_pos
            pred, true = self._run_linear_model(cur_model, test_data,
                cur_mid_embedding, batch_y, cur_trend_part,
                cur_revin_mean, cur_revin_stdev)
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]
            # loss before adaptation
            loss_before_adapt = criterion(pred, true)
            a1.append(loss_before_adapt.item())
            # cur_model.train()
            

            # tmp loss
            # cur_model.eval()
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            # adapt_start_pos = self.args.adapt_start_pos
            pred, true = self._run_linear_model(cur_model, test_data,
                cur_mid_embedding, batch_y, cur_trend_part,
                cur_revin_mean, cur_revin_stdev)
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]
            loss_before_adapt = criterion(pred, true)
            a1.append(loss_before_adapt.item())
            # cur_model.train()
            
            # ? uncomment this part as well if need
            # print(pred_0.shape, pred.shape)
            # print(cur_model.bias)
            # print(self.model.decoder.projection.bias)
            # # print(pred_0, pred)
            # assert (mid_embedding_0.detach().cpu().clone().numpy() == cur_mid_embedding).all()
            # assert true_0.equal(true)
            # assert pred_0.equal(pred)
            # assert (trend_part_0.detach().cpu().clone().numpy() == cur_trend_part).all()
            
            
            # # store pred and true
            # pred_before_adapt = pred.detach().cpu().clone().numpy()
            # true_before_adapt = true.detach().cpu().clone().numpy()
            

            if use_adapted_model:
                seq_len = self.args.seq_len
                pred_len = self.args.pred_len
                # adapt_start_pos = self.args.adapt_start_pos
                pred_answer, true_answer = self._run_linear_model(cur_model, test_data,
                    cur_mid_embedding, batch_y, cur_trend_part, cur_revin_mean, cur_revin_stdev)
            
            if self.args.adapt_part_channels:
                pred_answer = pred_answer[:, :, self.selected_channels]
                true_answer = true_answer[:, :, self.selected_channels]
            loss_ans_before = criterion(pred_answer, true_answer)
            loss_ans_before.backward()

            w_T = params[0].grad.T
            b = params[1].grad.unsqueeze(0)
            params_answer = torch.cat((w_T, b), 0)
            params_answer = params_answer.ravel()

            model_optim.zero_grad()

            # record backward time
            backward_time += time.time() - backward_time_start

            # get unselected_channels
            unselected_channels = list(range(self.args.c_out))
            for item in self.selected_channels:
                unselected_channels.remove(item)
            
            # record time for construction contextualized dataset
            construction_time_start = time.time()
            
            
            import torch.nn.functional as F
            if self.args.adapt_part_channels:  
                test_x = batch_x[:, -seq_len:, self.selected_channels].reshape(-1)
            else:
                test_x = batch_x[:, -seq_len:, :].reshape(-1)
            
            distance_pairs = []
            
            if self.args.remove_nearest:
                for ii in range(self.test_train_num):
                    if self.args.adapt_part_channels:
                        lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                    else:
                        lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                    dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                    distance_pairs.append([ii, dist])
                # randomly selected 'selected_data_num' samples
                import random
                selected_distance_pairs = random.sample(distance_pairs, self.args.selected_data_num)
            else:
                for ii in range(self.args.test_train_num):
                    # remove_cycle is used for ablation study
                    if not self.args.remove_cycle:
                        # phase difference between current sample and test sample
                        if 'illness' in self.args.data_path:
                            import math
                            cycle_remainer = math.fmod(self.args.test_train_num-1 + self.args.pred_len - ii, self.period)
                        else:
                            cycle_remainer = (self.args.test_train_num-1 + self.args.pred_len - ii) % self.period
                        # threshold for decision
                        threshold = self.period * self.args.lambda_period
                        if cycle_remainer > threshold or cycle_remainer < -threshold:
                            continue
                        
                    if self.args.adapt_part_channels:
                        lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                    else:
                        lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                    
                    # calculate sample similarity
                    dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                    distance_pairs.append([ii, dist])

                if not self.args.remove_distance:
                    # select samples with higher similarity
                    cmp = lambda item: item[1]
                    distance_pairs.sort(key=cmp)
                else:
                    # select samples with closer timestep difference
                    cmp = lambda item: item[0]
                    distance_pairs.sort(key=cmp, reverse=True)

                # select 'selected_data_num' samples
                selected_distance_pairs = distance_pairs[:self.args.selected_data_num]
                
            selected_indices = [item[0] for item in selected_distance_pairs]
            selected_distances = [item[1] for item in selected_distance_pairs]
            all_distances.append(selected_distances)

            # record time
            construction_time_end = time.time()
            construction_time += construction_time_end - construction_time_start

            cur_grad_list = []
            
            # record time for adaptation
            adaptation_time_start = time.time()
            
            # start training
            for epoch in range(test_train_epochs):

                gradients = []
                mean_loss = 0

                for ii in selected_indices:
                    
                    # Note: Here we need to subtract one more (pred_len-1) to be correct!!!
                    # This is because the original sample taken was batch_x[:, -seq_len:, :], while the current sample selected is batch_x[:, ii: ii+seq_len, :].
                    # batch_x itself has a length of 'seq_len + pred_len + test_train_num-1'
                    # So the distance between them is exactly '(seq_len + pred_len + test_train_num-1) - (seq_len) - (ii)'
                    # This is equal to 'pred_len-1 + test_train_num-ii', but here we're taking it forward, so we subtract it the other way around.
                    selected_mid_embedding = all_mid_embeddings[(START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) : (START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) + 1]
                    selected_trend_part = all_trend_parts[(START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) : (START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) + 1]
                    # selected_batch_x = all_batch_x_lst[START_INDEX + i + (ii - self.test_train_num) : START_INDEX + i + (ii - self.test_train_num)+1]
                    selected_revin_mean = all_revin_means[(START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) : (START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) + 1]
                    selected_revin_stdev = all_revin_stdevs[(START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) : (START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) + 1]

                    model_optim.zero_grad()

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    pred, true = self._run_linear_model(cur_model, test_data,
                        selected_mid_embedding, batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                        selected_trend_part,
                        selected_revin_mean, selected_revin_stdev)
                    

                    if self.args.adapt_part_channels:
                        pred = pred[:, :, self.selected_channels]
                        true = true[:, :, self.selected_channels]
                    
                    
                    if not self.args.use_nearest_data or self.args.use_further_data:
                        loss = criterion(pred, true)
                    else:
                        data_used_num = (self.test_train_num - (ii+1)) + self.args.adapt_start_pos
                        if data_used_num < pred_len:
                            loss = criterion(pred[:, :data_used_num, :], true[:, :data_used_num, :])
                        else:
                            loss = criterion(pred, true)

                    # loss = criterion(pred, true)
                    mean_loss += loss

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        w_T = params[0].grad.T
                        b = params[1].grad.unsqueeze(0)
                        params_tmp = torch.cat((w_T, b), 0)
                        original_shape = params_tmp.shape
                        params_tmp = params_tmp.ravel()

                        # store the gradient in 'cur_grad_list'
                        cur_grad_list.append(params_tmp.detach().cpu().numpy())

                        model_optim.zero_grad()
                        
            
            # function: multiply gradient by different weights
            def calc_weighted_params(params, weights):
                results = 0
                for i in range(len(params)):
                    results += params[i] * weights[i]
                return results
            
            # multiply gradient by different weights 
            if weights_given:
                weighted_params = calc_weighted_params(cur_grad_list, weights_given)
            else:
                weights_all_ones = [1 for i in range(self.test_train_num)]
                weighted_params = calc_weighted_params(cur_grad_list, weights_all_ones)
            
            # tranform 'weighted_params' from np.array to tensor
            weighted_params = torch.tensor(weighted_params)
            weighted_params = weighted_params.to(self.device)


            # calculate the angle between the gradients of ground-truth and the weighted sum of current gradients
            import math
            product = torch.dot(weighted_params, params_answer)
            product = product / (torch.norm(weighted_params) * torch.norm(params_answer))
            angel = math.degrees(math.acos(product))
            all_angels.append(angel)
            

            # change 'weighted_params' back to 'w_grad' and 'b_grad'
            weighted_params = weighted_params.reshape(original_shape)
            w_grad_T, b_grad = torch.split(weighted_params, [weighted_params.shape[0]-1, 1])
            w_grad = w_grad_T.T  # (7, 512)
            b_grad = b_grad.squeeze(0)  # (7)


            # update new param as: original param + gradient
            from torch.nn.parameter import Parameter
            cur_lr = self.args.learning_rate * self.args.adapted_lr_times

            # make gradients on unselected channels as 0
            if self.args.adapt_part_channels:
                w_grad[unselected_channels, :] = 0
                b_grad[unselected_channels] = 0
            
            # Since this is a linear mapping layer, we only need to get weight and bias.
            cur_model.weight = Parameter(cur_model.weight - w_grad * cur_lr)
            cur_model.bias = Parameter(cur_model.bias - b_grad * cur_lr)

            # calculate mean loss
            mean_loss = mean_loss / self.args.selected_data_num
            a2.append(mean_loss.item())
            
            # record time
            adaptation_time_end = time.time()
            adaptation_time += adaptation_time_end - adaptation_time_start


            # # PART 3: check if over-fitting on selected samples           
            # seq_len = self.args.seq_len
            # label_len = self.args.label_len
            # pred_len = self.args.pred_len
            # tmp_loss = 0
            # for ii in selected_indices:
            #     selected_mid_embedding = all_mid_embeddings[(START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) : (START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) + 1]
            #     selected_trend_part = all_trend_parts[(START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) : (START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) + 1]
            #     selected_revin_mean = all_revin_means[(START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) : (START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) + 1]
            #     selected_revin_stdev = all_revin_stdevs[(START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) : (START_INDEX + i) + (ii - self.test_train_num) - (pred_len-1) + 1]
            #     pred, true = self._run_linear_model(cur_model, test_data,
            #             selected_mid_embedding, batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :],
            #             selected_trend_part,
            #             selected_revin_mean, selected_revin_stdev)
            #     if self.args.adapt_part_channels:
            #         pred = pred[:, :, self.selected_channels]
            #         true = true[:, :, self.selected_channels]
            #     tmp_loss += criterion(pred, true)
            # tmp_loss = tmp_loss / self.args.selected_data_num
            # a3.append(tmp_loss.item())
            a3.append(0)

            # record time fore prediction
            prediction_time_start = time.time()

            cur_model.eval()
            
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            if use_adapted_model:
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred, true = self._run_linear_model(cur_model, test_data,
                        cur_mid_embedding, batch_y, cur_trend_part,
                        cur_revin_mean, cur_revin_stdev)

            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]
                
            pred_after_adapt = pred.detach().cpu().clone().numpy()
            true_after_adapt = true.detach().cpu().clone().numpy()

            loss_after_adapt = criterion(pred, true)
            a4.append(loss_after_adapt.item())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())
            
            prediction_time_end = time.time()
            prediction_time += prediction_time_end - prediction_time_start
            
            
            rest_time_start = time.time()

            if (i+1) % 100 == 0 or (data_len - i) <= 100 and (i+1) % 10 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                print(gradients)
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
                
                avg_1, avg_2, avg_3, avg_4 = 0, 0, 0, 0
                avg_angel = 0
                num = len(a1)
                for iii in range(num):
                    avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]; avg_4 += a4[iii]; avg_angel += all_angels[iii]
                avg_1 /= num; avg_2 /= num; avg_3 /= num; avg_4 /= num; avg_angel /= num
                print("1.before_adapt, 2.adapt_sample, 3.adapt_sample_after, 4.after_adapt, 5.angel_between_answer")
                print("average:", avg_1, avg_2, avg_3, avg_4, avg_angel)
                print("last one:", a1[-1], a2[-1], a3[-1], a4[-1], all_angels[-1])

                printed_selected_channels = [item+1 for item in self.selected_channels]
                print(f"adapt_part_channels: {self.args.adapt_part_channels}")
                print(f"remove_distance: {self.args.remove_distance}, remove_cycle: {self.args.remove_cycle}, remove_nearest: {self.args.remove_nearest}")
                print(f"first 25th selected_channels: {printed_selected_channels[:25]}")
                print(f"selected_distance_pairs are: {selected_distance_pairs}")

            # # Draw figures for prediction results after adaptation:
            # # print("bacth_x.shape:", batch_x.detach().cpu().numpy().shape)
            # # print("pred_before_adapt.shape", pred_before_adapt.shape)
            # # print("pred_after_adapt.shape", pred_after_adapt.shape)
            # interval = 20 if 'illness' in self.args.data_path else 50
            # if i % interval == 0:
            #     # if need draw figures for prediction after adaptation
            #     if self.args.draw_adapt_figure:
            #         folder_path = './test_results/' + setting + '/'
            #         if not os.path.exists(folder_path):
            #             os.makedirs(folder_path)
            
            #         # Since actual length of batch_x is 'ttn+pred_len+seq_len', so we need to extract the last [:, -seq_len:, :] part!
            #         # And since illness is far too short, we will preserve more data for visualization
            #         if not 'illness' in self.args.data_path:
            #             input = batch_x[:, -self.args.seq_len:, :]
            #         else:
            #             input = batch_x[:, -3*self.args.seq_len:, :]
            #             # input = batch_x[:, :, :]
            #         input = input.detach().cpu().numpy()
            #         # print(batch_x.shape, input.shape, self.args.seq_len)
            #         assert (true_before_adapt == true_after_adapt).all()
                    
            #         gt = np.concatenate((input[0, :, -1], true_before_adapt[0, :, -1]), axis=0)
            #         pd_before_adapt = np.concatenate((input[0, :, -1], pred_before_adapt[0, :, -1]), axis=0)
            #         pd_after_adapt = np.concatenate((input[0, :, -1], pred_after_adapt[0, :, -1]), axis=0)
                    
            #         name = os.path.join(folder_path, str(i) + '.pdf')
            #         plt.figure()
            #         # give a bigger zorder to ground-truth, to make it at the top
            #         plt.plot(gt, label='GroundTruth', linewidth=2, zorder=3)
            #         if pd_before_adapt is not None:
            #             plt.plot(pd_before_adapt, label='Before adaptation', linewidth=2, zorder=1)
            #         if pd_after_adapt is not None:
            #             plt.plot(pd_after_adapt, label='After adaptation', linewidth=2, zorder=2)
            #         plt.legend()
            #         plt.savefig(name, bbox_inches='tight')

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()
            
            # record rest time
            rest_time += time.time() - rest_time_start

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")

        print(f"Backward cost time: {backward_time}")
        print(f"Construction dataset cost time: {construction_time}")
        print(f"Adaptation cost time: {adaptation_time}")
        print(f"Prediction cost time: {prediction_time}")
        print(f"Rest cost time: {rest_time}")

        return mse, mae



    
    def adapt_whole_model(self, setting, test=0, is_training_part_params=False, use_adapted_model=True, test_train_epochs=1, weights_given=None, adapted_degree="small", weights_from="test"):
        test_data, test_loader = self._get_data_at_test_time(flag='test')
        data_len = len(test_data)
        if test:
            print('loading model from checkpoint !!!')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        self.model.eval()

        preds = []
        trues = []

        a1, a2, a3, a4 = [], [], [], []
        all_distances = []

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        criterion = nn.MSELoss()
        test_time_start = time.time()

        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        if weights_given:
            print(f"The given weights is {weights_given}")
            print(f"The length of given weights is {len(weights_given)}")
        

        # self.model.eval()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if (data_len - i) <= data_len % self.args.batch_size: break
            
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
            cur_model = copy.deepcopy(self.model)
            # cur_model.train()
            cur_model.eval()

            if is_training_part_params:
                params = []
                names = []
                params_norm = []
                names_norm = []
                cur_model.requires_grad_(False)
                for n_m, m in cur_model.named_modules():
                    # TODO: 这里改成map形式，用if-else-if太冗余了！！！
                    if self.args.model == 'ETSformer':
                        linear_layer_name = "decoder.pred"
                    elif self.args.model == 'Crossformer':
                        # There are e_layers+1 of layers in decoder in original Crossformer, so the last layer is self.args.e_layers
                        linear_layer_name = f"decoder.decode_layers.{self.args.e_layers}.linear_pred"
                    elif "Linear" in self.args.model:
                        linear_layer_name = "Linear"
                    elif "PatchTST" in self.args.model:
                        linear_layer_name = "model.head.linear"
                    else:
                        linear_layer_name = "decoder.projection"
                    
                    if linear_layer_name in n_m:
                        m.requires_grad_(True)
                        for n_p, p in m.named_parameters():
                            if n_p in ['weight', 'bias']:  # weight is scale, bias is shift
                                params.append(p)
                                names.append(f"{n_m}.{n_p}")

                # set lr:
                lr = self.args.learning_rate * self.args.adapted_lr_times
                # changing Adam to SGD optimizer:
                model_optim = optim.SGD(params, lr=lr)  # model_optim = optim.Adam(params, lr=lr)
            else:
                cur_model.requires_grad_(True)
                # set lr:
                lr = self.args.learning_rate * self.args.adapted_lr_times
                model_optim = optim.SGD(cur_model.parameters(), lr=lr)
            
            
            # tmp loss
            # cur_model.eval()
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if not self.args.use_nearest_data or self.args.use_further_data:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -seq_len:, :], batch_y, 
                    batch_x_mark[:, -seq_len:, :], batch_y_mark)
            else:
                pred, true = self._process_one_batch_with_model(cur_model, test_data,
                    batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                    batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)
            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]
            # get loss before adaptation
            loss_before_adapt = criterion(pred, true)
            a1.append(loss_before_adapt.item())
            # cur_model.train()
        

            # get unselected channels
            unselected_channels = list(range(self.args.c_out))
            for item in self.selected_channels:
                unselected_channels.remove(item)

            # calculate sample similarity
            import torch.nn.functional as F
            if self.args.adapt_part_channels:  
                test_x = batch_x[:, -seq_len:, self.selected_channels].reshape(-1)
            else:
                test_x = batch_x[:, -seq_len:, :].reshape(-1)
            
            distance_pairs = []
            
            if self.args.remove_nearest:
                for ii in range(self.test_train_num):
                    if self.args.adapt_part_channels:
                        lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                    else:
                        lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                    dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                    distance_pairs.append([ii, dist])
                # randomly select samples with the number of 'selected_data_num'
                import random
                selected_distance_pairs = random.sample(distance_pairs, self.args.selected_data_num)
            else:
                for ii in range(self.args.test_train_num):
                    # remove_cycle is used for ablation study
                    if not self.argslambdaer.remove_cycle:
                        # phase difference between current sample and test sample
                        if 'illness' in self.args.data_path:
                            import math
                            cycle_remainer = math.fmod(self.args.test_train_num-1 + self.args.pred_len - ii, self.period)
                        else:
                            cycle_remainer = (self.args.test_train_num-1 + self.args.pred_len - ii) % self.period
                        # threshold for decision
                        threshold = self.period * self.args.lambda_period
                        if cycle_remainer > threshold or cycle_remainer < -threshold:
                            continue
                        
                    if self.args.adapt_part_channels:
                        lookback_x = batch_x[:, ii : ii+seq_len, self.selected_channels].reshape(-1)
                    else:
                        lookback_x = batch_x[:, ii : ii+seq_len, :].reshape(-1)
                        
                    dist = F.pairwise_distance(test_x, lookback_x, p=2).item()
                    distance_pairs.append([ii, dist])

                if not self.args.remove_distance:
                    # select samples with higher similarity
                    cmp = lambda item: item[1]
                    distance_pairs.sort(key=cmp)
                else:
                    # select samples with closer timestep difference
                    cmp = lambda item: item[0]
                    distance_pairs.sort(key=cmp, reverse=True)

                # select 'selected_data_num' samples
                selected_distance_pairs = distance_pairs[:self.args.selected_data_num]
                
            selected_indices = [item[0] for item in selected_distance_pairs]
            selected_distances = [item[1] for item in selected_distance_pairs]
            all_distances.append(selected_distances)

            cur_grad_list = []
            
            for epoch in range(test_train_epochs):
                
                accpted_samples_num = set()

                # num_of_loss_per_update = 1
                mean_loss = 0

                for ii in selected_indices:

                    model_optim.zero_grad()

                    seq_len = self.args.seq_len
                    label_len = self.args.label_len
                    pred_len = self.args.pred_len

                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
                        batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])


                    if self.args.adapt_part_channels:
                        pred = pred[:, :, self.selected_channels]
                        true = true[:, :, self.selected_channels]
                    
                    # check whether use nearest data?
                    if not self.args.use_nearest_data or self.args.use_further_data:
                        loss = criterion(pred, true)
                    else:
                        data_used_num = (self.test_train_num - (ii+1)) + self.args.adapt_start_pos
                        if data_used_num < pred_len:
                            loss = criterion(pred[:, :data_used_num, :], true[:, :data_used_num, :])
                        else:
                            loss = criterion(pred, true)

                    # loss = criterion(pred, true)
                    mean_loss += loss
                    
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        # scaler.step(model_optim_norm)
                        scaler.update()
                    else:
                        # directly backward and update params!!
                        loss.backward()
                        model_optim.step()
            

            # mean_loss = mean_loss / self.test_train_num
            mean_loss = mean_loss / self.args.selected_data_num
            a2.append(mean_loss.item())


            # seq_len = self.args.seq_len
            # label_len = self.args.label_len
            # pred_len = self.args.pred_len
            # tmp_loss = 0
            # for ii in selected_indices:
            #     pred, true = self._process_one_batch_with_model(cur_model, test_data,
            #         batch_x[:, ii : ii+seq_len, :], batch_x[:, ii+seq_len-label_len : ii+seq_len+pred_len, :], 
            #         batch_x_mark[:, ii : ii+seq_len, :], batch_x_mark[:, ii+seq_len-label_len : ii+seq_len+pred_len, :])
            #     if self.args.adapt_part_channels:
            #         pred = pred[:, :, self.selected_channels]
            #         true = true[:, :, self.selected_channels]
            #     tmp_loss += criterion(pred, true)
            # tmp_loss = tmp_loss / self.args.selected_data_num
            # a3.append(tmp_loss.item())
            a3.append(0)


            cur_model.eval()

            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            adapt_start_pos = self.args.adapt_start_pos
            if use_adapted_model:
                if not self.args.use_nearest_data or self.args.use_further_data:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -seq_len:, :], batch_y, 
                        batch_x_mark[:, -seq_len:, :], batch_y_mark)
                else:
                    pred, true = self._process_one_batch_with_model(cur_model, test_data,
                        batch_x[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y, 
                        batch_x_mark[:, -((pred_len - adapt_start_pos) + seq_len):-(pred_len - adapt_start_pos), :], batch_y_mark)

            if self.args.adapt_part_channels:
                pred = pred[:, :, self.selected_channels]
                true = true[:, :, self.selected_channels]

            # get loss after adaptation
            loss_after_adapt = criterion(pred, true)
            a4.append(loss_after_adapt.item())

            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

            if (i+1) % 100 == 0 or (data_len - i) <= 100 and (i+1) % 10 == 0:
                print("\titers: {0}, cost time: {1}s".format(i + 1, time.time() - test_time_start))
                tmp_p = np.array(preds); tmp_p = tmp_p.reshape(-1, tmp_p.shape[-2], tmp_p.shape[-1])
                tmp_t = np.array(trues); tmp_t = tmp_t.reshape(-1, tmp_t.shape[-2], tmp_t.shape[-1])
                tmp_mae, tmp_mse, _, _, _ = metric(tmp_p, tmp_t)
                print('mse:{}, mae:{}'.format(tmp_mse, tmp_mae))
                
                avg_1, avg_2, avg_3, avg_4 = 0, 0, 0, 0
                num = len(a1)
                for iii in range(num):
                    avg_1 += a1[iii]; avg_2 += a2[iii]; avg_3 += a3[iii]; avg_4 += a4[iii]
                avg_1 /= num; avg_2 /= num; avg_3 /= num; avg_4 /= num
                print("1.before_adapt, 2.adapt_sample, 3.adapt_sample_after, 4.after_adapt")
                print("average:", avg_1, avg_2, avg_3, avg_4)
                print("last one:", a1[-1], a2[-1], a3[-1], a4[-1])

                printed_selected_channels = [item+1 for item in self.selected_channels]
                print(f"adapt_part_channels: {self.args.adapt_part_channels}")
                print(f"remove_distance: {self.args.remove_distance}, remove_cycle: {self.args.remove_cycle}, remove_nearest: {self.args.remove_nearest}")
                print(f"first 25th selected_channels: {printed_selected_channels[:25]}")
                print(f"selected_distance_pairs are: {selected_distance_pairs}")


            # if i % 20 == 0:
            #     input = batch_x.detach().cpu().numpy()
            #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

            cur_model.eval()
            # cur_model.cpu()
            del cur_model
            torch.cuda.empty_cache()

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        # f = open("result.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        print(f"Test - cost time: {time.time() - test_time_start}s")

        return mse, mae
    
    
    def _run_linear_model(self, model, dataset_object, mid_embedding, batch_y, trend_part=None, revin_mean=None, revin_stdev=None):
        # change batch_x of type:numpy.array to tensor
        mid_embedding = torch.tensor(mid_embedding).float().to(self.device)
        batch_y = batch_y.float()
        
        # change trend_part and mean and variance of revin to tensor as well
        trend_part = torch.tensor(trend_part).float().to(self.device)
        revin_mean = torch.tensor(revin_mean).float().to(self.device)
        revin_stdev = torch.tensor(revin_stdev).float().to(self.device)

        # linear
        outputs = model(mid_embedding)
        # Only Informer do not include trend_part
        if trend_part is not None:
            if self.args.model != "Informer":
                outputs = outputs + trend_part
        # use mean and stdev of RevIN for denormalization
        if revin_mean is not None:
            outputs = outputs * revin_stdev
            outputs = outputs + revin_mean
        
        f_dim = -1 if self.args.features=='MS' else 0

        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # outputs is pred, and batch_y is true
        return outputs, batch_y
    

    def _process_one_batch_with_model(self, model, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, return_mid_embedding=False):
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
                if 'Linear' in self.args.model or self.args.model == 'Crossformer' or 'TST' in self.args.model:
                    outputs = model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if 'Linear' in self.args.model or self.args.model == 'Crossformer' or 'TST' in self.args.model:
                outputs = model(batch_x)
            else:
                if self.args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    if return_mid_embedding:
                        outputs, mid_embedding, *res = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_mid_embedding=return_mid_embedding)  # [B, L, D]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # [B, L, D]
        
        # if self.args.inverse:
        #     outputs = dataset_object.inverse_transform(outputs)
        
        f_dim = -1 if self.args.features=='MS' else 0

        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        # outputs is pred, and batch_y is true
        if return_mid_embedding:
            return outputs, batch_y, mid_embedding, *res
        else:
            return outputs, batch_y


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
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
                        if 'Linear' in self.args.model or self.args.model == 'Crossformer' or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or self.args.model == 'Crossformer' or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
