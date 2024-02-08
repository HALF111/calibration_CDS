import argparse
import os
import sys
import torch
from exp.exp_main import Exp_Main
from exp.exp_main_test import Exp_Main_Test
import random
import numpy as np


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='FEDformer',
                        help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--cross_activation', type=str, default='tanh'

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    # n_heads = 4 for Crossformer
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    # e_layers == 3 for Crossformer
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    # dropout == 0.2 for ETSformer & Crossformer
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    # activation == 'sigmoid' for ETSformer
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    
    # Reformer & Autoformer
    parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    # FEDformer
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')
    # ETSformer
    parser.add_argument('--K', type=int, default=1, help='Top-K Fourier bases')
    parser.add_argument('--min_lr', type=float, default=1e-30)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--std', type=float, default=0.2)
    parser.add_argument('--smoothing_learning_rate', type=float, default=0, help='optimizer learning rate')
    parser.add_argument('--damping_learning_rate', type=float, default=0, help='optimizer learning rate')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer')
    # Crossformer
    parser.add_argument('--seg_len', type=int, default=6, help='segment length (L_seg)')
    parser.add_argument('--win_size', type=int, default=2, help='window size for segment merge')
    parser.add_argument('--cross_factor', type=int, default=10, help='num of routers in Cross-Dimension Stage of TSA (c)')
    parser.add_argument('--baseline', action='store_true', help='whether to use mean of past series as baseline for prediction', default=False)
    
    # DLinear
    parser.add_argument('--individual_DLinear', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
    
    # PatchTST
    # 本文一共有3个dropout，分别为：dropout、fc_dropout、head_dropout
    # fc_dropout是指最后一层如果是预训练层（会用一维卷积的预测头），对应的dropout
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    # head_dropout是指最后一层如果是"展平+线性映射层"，对应的dropout
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    # 默认是需要做padding的！
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    # 默认是做RevIN的！！但RevIN的affine参数并不学习
    parser.add_argument('--revin_PatchTST', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    # RevIN默认减去均值（也可以像NLinear一样减掉最后一个值）
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    # 默认不做分解
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    # 默认也是不使用独立的头！
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    # lradj == 'exponential_with_warmup' for ETSformer
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # PatchTST
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')
    
    # 默认是做RevIN的！！但RevIN的affine参数并不学习
    parser.add_argument('--add_revin', action='store_true')  # whether to use RevIN
    # parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    # # RevIN默认减去均值（也可以像NLinear一样减掉最后一个值）
    # parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')

    # test_train_num
    parser.add_argument('--test_train_num', type=int, default=10, help='how many samples to be trained during test')
    parser.add_argument('--adapted_lr_times', type=float, default=1, help='the times of lr during adapted')  # adaptation时的lr是原来的lr的几倍？
    parser.add_argument('--adapted_batch_size', type=int, default=1, help='the batch_size for adaptation use')  # adaptation时的数据集取的batch_size设置为多大
    parser.add_argument('--test_train_epochs', type=int, default=1, help='the epochs for test-time adaptation')  # adaptation时做几轮adaptation？
    parser.add_argument('--run_train', action='store_true')
    parser.add_argument('--run_test', action='store_true')
    parser.add_argument('--run_test_batchsize1', action='store_true')
    parser.add_argument('--run_adapt', action='store_true')
    parser.add_argument('--run_calc', action='store_true')
    parser.add_argument('--run_get_grads', action='store_true')
    parser.add_argument('--run_get_lookback_data', action='store_true')
    parser.add_argument('--run_select_with_distance', action='store_true')
    parser.add_argument('--run_select_caching', action='store_true')
    # selected_data_num表示从过去test_train_num个样本中按照距离挑选出最小的多少个出来
    # 因此这里要求必须有lookback_data_num <= test_train_num成立
    parser.add_argument('--selected_data_num', type=int, default=10)
    
    # 获取all数据时候必须要用batch_size为1来遍历
    parser.add_argument('--all_data_batch_size', type=int, default=1, help='the batch_size for getting all data')
    
    parser.add_argument('--lambda_period', type=float, default=0.1)

    parser.add_argument('--get_grads_from', type=str, default="test", help="options:[test, val]")
    parser.add_argument('--adapted_degree', type=str, default="small", help="options:[small, large]")

    # 解析解用的参数alpha和lambda
    parser.add_argument('--lambda_reg', type=int, default=1)
    parser.add_argument('--alpha', type=int, default=1)

    # 改用更近（填0）或更远（周期性）的数据做adaptation
    parser.add_argument('--use_nearest_data', action='store_true')
    parser.add_argument('--use_further_data', action='store_true')
    # 理论上当adapt_start_pos == pred_len时，本方法与原来方法相同;
    # 但是但是由于实现的原因，要求必须保证：
    # 1.当use_nearest_data时，要求保证adapt_start_pos严格小于pred_len
    # 2.当use_further_data时，要求保证adapt_start_pos严格大于等于pred_len
    parser.add_argument('--adapt_start_pos', type=int, default=1)

    parser.add_argument('--run_calc_acf', action='store_true')
    parser.add_argument('--acf_lag', type=int, default=1)
    parser.add_argument('--run_calc_kldiv', action='store_true')
    parser.add_argument('--get_data_error', action='store_true')

    parser.add_argument('--adapt_part_channels', action='store_true')
    # # 仅对周期性数据做fine-tuning
    # parser.add_argument('--adapt_cycle', action='store_true')
    
    # 对几个指标的ablation study？
    parser.add_argument('--remove_distance', action='store_true')
    parser.add_argument('--remove_cycle', action='store_true')
    parser.add_argument('--remove_nearest', action='store_true')
    
    # 微调整个模型
    parser.add_argument('--adapt_whole_model', action='store_true')
    
    # 画出微调前后的结果对比图
    parser.add_argument('--draw_adapt_figure', action='store_true')

    # KNN
    parser.add_argument('--run_knn', action='store_true')
    parser.add_argument('--feature_dim', type=int, default=50)
    parser.add_argument('--k_value', type=int, default=10)

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # Exp = Exp_Main
    Exp = Exp_Main_Test

    if args.is_training:
        for ii in range(args.itr):
            print(f"-------Start iteration {ii+1}--------------------------")

            # setting record of experiments
            # 别忘记加上test_train_num一项！！！
            setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_revin{}_{}_{}'.format(
                args.task_id,
                args.model,
                args.mode_select,
                args.modes,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.add_revin,
                # args.test_train_num,
                args.des,
                ii)

            exp = Exp(args)  # set experiments
            if args.run_train:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

            if args.run_test:
                print('>>>>>>>normal testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # exp.test(setting, flag="test")
                exp.test(setting, test=1, flag="test")

            if args.run_test_batchsize1:
                print('>>>>>>>normal testing but batch_size is 1 : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # exp.test(setting, flag="test_with_batchsize_1")
                exp.test(setting, test=1, flag="test_with_batchsize_1")

            if args.run_adapt:
                # # 对整个模型进行fine-tuning
                # print('>>>>>>>my testing with all parameters trained : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # exp.my_test(setting, test=1, is_training_part_params=False, use_adapted_model=True, test_train_epochs=args.test_train_epochs)


                # 只对最后的全连接层projection层进行fine-tuning
                print('>>>>>>>my testing with test-time training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # exp.my_test(setting, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1)
                exp.my_test(setting, test=1, is_training_part_params=True, use_adapted_model=True, test_train_epochs=args.test_train_epochs)

                # exp.my_test_mp(setting, is_training_part_params=True, use_adapted_model=True, test_train_epochs=1)

            if args.run_calc:
                print('>>>>>>>run_calc test with test-time training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

                #获取梯度
                weight_path = "./grads_npy/" + setting
                if args.get_grads_from == "test":
                    weight_file = f"{weight_path}/weights_{args.get_grads_from}_{args.adapted_degree}_ttn{args.test_train_num}.txt"
                elif args.get_grads_from == "val":
                    weight_file = f"{weight_path}/weights_{args.get_grads_from}_{args.adapted_degree}_ttn{args.test_train_num}.txt"

                if os.path.exists(weight_file):
                    with open(weight_file) as f:
                        weights_str = f.readline()
                        weights_str_list = weights_str.split(',')
                        weights = [float(weight) for weight in weights_str_list]
                    print(weights)
                else:
                    weights = None

                mse, mae = exp.calc_test(setting, test=1, is_training_part_params=True, use_adapted_model=True, test_train_epochs=args.test_train_epochs, weights_given=weights)

                # 存储mse和mae结果，便于读取
                result_dir = "./mse_and_mae_results"
                dataset_name = args.data_path.replace(".csv", "")
                file_name = f"{dataset_name}_pl{args.pred_len}_alpha{int(args.alpha)}_ttn{args.test_train_num}_lambda{int(args.lambda_reg)}.txt"

                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                file_path = os.path.join(result_dir, file_name)
                with open(file_path, "w") as f:
                    f.write(f"{mse}, {mae}")
            
            if args.run_select_with_distance:
                # 只对最后的全连接层projection层进行fine-tuning
                print('>>>>>>>my testing with test-time training : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                mse, mae = exp.select_with_distance(setting, test=1, is_training_part_params=True, use_adapted_model=True, test_train_epochs=args.test_train_epochs)
                
                # 存储mse和mae结果，便于读取
                result_dir = "./mse_and_mae_results"
                dataset_name = args.data_path.replace(".csv", "")
                if args.add_revin:
                    file_name = f"RevIN_{args.model}_{dataset_name}_pl{args.pred_len}_ttn{args.test_train_num}_select{args.selected_data_num}_lr{args.adapted_lr_times:.2f}.txt"
                else:
                    file_name = f"{args.model}_{dataset_name}_pl{args.pred_len}_ttn{args.test_train_num}_select{args.selected_data_num}_lr{args.adapted_lr_times:.2f}.txt"

                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                file_path = os.path.join(result_dir, file_name)
                with open(file_path, "w") as f:
                    f.write(f"{mse}, {mae}")

            if args.run_select_caching:
                # 只对最后的全连接层projection层进行fine-tuning
                print('>>>>>>>my testing with test-time training with caching : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                mse, mae = exp.select_with_distance_caching(setting, test=1, is_training_part_params=True, use_adapted_model=True, test_train_epochs=args.test_train_epochs)
                
                # 存储mse和mae结果，便于读取
                result_dir = "./mse_and_mae_results"
                dataset_name = args.data_path.replace(".csv", "")
                if args.add_revin:
                    file_name = f"RevIN_{args.model}_{dataset_name}_pl{args.pred_len}_ttn{args.test_train_num}_select{args.selected_data_num}_lr{args.adapted_lr_times:.2f}.txt"
                else:
                    file_name = f"{args.model}_{dataset_name}_pl{args.pred_len}_ttn{args.test_train_num}_select{args.selected_data_num}_lr{args.adapted_lr_times:.2f}.txt"

                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                file_path = os.path.join(result_dir, file_name)
                with open(file_path, "w") as f:
                    f.write(f"{mse}, {mae}")

            if args.adapt_whole_model:
                # 微调整个模型
                mse, mae = exp.adapt_whole_model(setting, test=1, is_training_part_params=False, use_adapted_model=True, test_train_epochs=args.test_train_epochs)
            
            if args.run_get_grads:
                print('>>>>>>>get grads : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                if args.get_grads_from == "test":  # 在test数据集上做
                    exp.get_grads(setting, test=1, is_training_part_params=True, use_adapted_model=True, test_train_epochs=args.test_train_epochs, flag="test", adapted_degree=args.adapted_degree)
                elif args.get_grads_from == "val":  # 在val数据集上做
                    exp.get_grads(setting, test=1, is_training_part_params=True, use_adapted_model=True, test_train_epochs=args.test_train_epochs, flag="val", adapted_degree=args.adapted_degree)

            if args.run_get_lookback_data:
                print('>>>>>>>get look-back data : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.get_lookback_data(setting)

            if args.run_calc_acf:
                # 记得一定一定一定要加上"--batch_size 1"！！！
                print('>>>>>>>calc ACF with lag={} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.acf_lag, setting))
                exp.calc_acf(setting, lag=args.acf_lag)
            
            if args.run_calc_kldiv:
                # 记得一定一定一定要加上"--batch_size 1"！！！
                print('>>>>>>>calc KLdiv between train/val/test{} : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.acf_lag, setting))
                exp.calc_KLdiv(setting)
            
            if args.get_data_error:
                # 记得一定一定一定要加上"--batch_size 1"！！！
                print('>>>>>>>get_data_error of train/val/test: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.get_data_error(setting=setting)
            
            if args.run_knn:
                # 记得一定一定一定要加上"--batch_size 1"！！！
                print('>>>>>>>run_knn for test{}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.run_KNN(setting=setting)

            # print('>>>>>>>my testing but with original model : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.my_test(setting, is_training_part_params=True, use_adapted_model=False)


            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
