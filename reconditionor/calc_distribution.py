import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# name = "ETTh1"
seq_len = 336
pred_len = 96

scale = False
scaler = StandardScaler()

type_map = {'train': 0, 'val': 1, 'test': 2}

# model = "FEDformer"
model = "PatchTST"

final_results = {}

for name in ["ETTh1", "ETTh2", "ETTm2", "Electricity", "traffic", "weather", "national_illness"]:
    
    pred_len = 24 if "ill" in name else 96
    seq_len = 104 if "ill" in name else 336

    dir_path = f"./error_results/{name}_{seq_len}_{pred_len}_{model}/"
    print(name)

    # for flag in ["train", "val", "test"]:
    for flag in ["train"]:
        print(flag)
        
        # # PART I：获取error计算结果
        # # result = [mae, mse, rmse, mape, mspe, err_mean, err_var, err_abs_mean, err_abs_var, pos_num, neg_num]
        # results = []
        # with open(dir_path + f"pl{pred_len}_{flag}.txt") as f:
        #     lines = f.readlines()
        #     for result in lines:
        #         result = result.split(",")
        #         result = [float(item) for item in result]
        #         results.append(result)
        # print(results[:5])
        
        
        # PART2：获取残差数据
        residuals_file_name = dir_path + f"residuals_pl{pred_len}_{flag}.npy"
        residuals = np.load(residuals_file_name)
        print(residuals.shape)
        
        if "ETTh" in name or "traffic" in name or "ECL" in name or "Exchange" in name:
            period = 24
        elif "ETTm" in name:
            period = 96
        elif "weather" in name:
            period = 144
        elif "ili" in name:
            period = 52
        
        data_len = residuals.shape[0] // period * period
        residuals_copy = residuals[:data_len]
        residuals_copy_reshape = residuals_copy.reshape(data_len//period, period, residuals.shape[1], residuals.shape[2])
        print(residuals_copy_reshape.shape)
        # assert (residuals_copy_reshape[1, 0] == residuals[1*period+0]).all()
        # assert (residuals_copy_reshape[2, 4] == residuals[2*period+4]).all()
        
        residuals_period_list = np.split(residuals_copy_reshape, period, axis=1)
        print(len(residuals_period_list))
        for i in range(len(residuals_period_list)):
            residuals_period_list[i] = residuals_period_list[i].squeeze(axis=1)
        print(residuals_period_list[0].shape)
        # assert (residuals_period_list[0][1] == residuals[1*period+0]).all()
        # assert (residuals_period_list[4][2] == residuals[2*period+4]).all()
        
        
        residuals = residuals.flatten()
        
        print(max(residuals), min(residuals))
        print(residuals.mean(), residuals.var())
        print(residuals.shape)
        
        for i in range(len(residuals_period_list)):
            residuals_period_list[i] = residuals_period_list[i].flatten()
            print(residuals_period_list[i].mean(), residuals_period_list[i].var())
        print(residuals_period_list[0].shape)
        
        
        
        # 先按箱子划分一下吧：
        def split_bins(array, max_val, min_val, bin_num):
            interval = (max_val - min_val) / bin_num
            bins = np.arange(min_val-interval, max_val+interval, interval)
            splitted_array = pd.cut(array, bins=bins)
            # print(splitted_array.value_counts())
            freq = splitted_array.value_counts().values
            # print(freq)
            return freq
        
        # 计算kl散度？？？
        BIN_NUM = 50
        max_val, min_val = max(residuals), min(residuals)
        freq_all = split_bins(residuals, max_val, min_val, BIN_NUM)
        
        import scipy.stats
        KL_list = []
        for i in range(len(residuals_period_list)):
            freq_cur = split_bins(residuals_period_list[i], max_val, min_val, BIN_NUM)
            # freq_cur = [item*period for item in freq_cur]
            # print(freq_cur, freq_all)
            kl_cur = scipy.stats.entropy(freq_cur, freq_all)
            print(f"kl for {i}-th cycle:", kl_cur)
            # print(scipy.stats.entropy(freq_all, freq_cur))
            # print("---")
            KL_list.append(kl_cur)
        
        # 平均后即为最终的指标
        reconditionor = sum(KL_list)/len(KL_list)
        print(f"Final indicator: {reconditionor}")
        
        final_results[name] = reconditionor
    
    print("final_results:", final_results)
        
        # # 画图1
        # # max_value, min_value = 3, -3
        # max_value, min_value = 1, -1
        # interval = 0.01
        # ranges = list(np.arange(min_value, max_value, interval))
        # print(ranges)

        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.figure(figsize=(10, 15), dpi=100)
        # n, bins, patches = plt.hist(residuals, bins=ranges, edgecolor="r", histtype="bar", alpha=0.5)
        
        # if name == "ili":
        #     plt.hist(residuals_period_list[-1], bins=ranges, edgecolor="b", histtype="bar", alpha=0.5)
        #     plt.hist(residuals_period_list[18], bins=ranges, edgecolor="g", histtype="bar", alpha=0.5)

        # # # 2.0 在每个条柱上标注数量
        # # for i in range(len(n)):
        # #     plt.text(bins[i]+(bins[1]-bins[0])/2, n[i]*1.01, int(n[i]), ha='center', va='bottom', fontsize=25)
        
        # # 2.1 添加刻度
        # min_1 = min_value
        # max_1 = max_value
        # division_number = len(ranges)+1
        # t1 = np.linspace(min_1, max_1, num=division_number)
        # plt.xticks(t1)
        
        # # 2.2 添加网格和标题
        # plt.grid(linestyle='--')

        # plt.xlabel("difference", fontdict={'size': 16})
        # plt.ylabel("numbers", fontdict={'size': 16})
        # plt.title(f"the histogram of residual between pred and true of {name}_{flag}", fontdict={'size': 20})
        # # plt.legend(loc='best')
        # plt.savefig(f"./residuals_{name}_{flag}.png")
        # plt.show()
        

        # # 画各个分布的子图
        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.figure(figsize=(10, 15), dpi=100)
        # if name == "ili":
        #     plt.hist(residuals_period_list[-1], bins=ranges, edgecolor="b", histtype="bar", alpha=0.8)
        #     plt.hist(residuals_period_list[18], bins=ranges, edgecolor="g", histtype="bar", alpha=0.8)

        # # # 2.0 在每个条柱上标注数量
        # # for i in range(len(n)):
        # #     plt.text(bins[i]+(bins[1]-bins[0])/2, n[i]*1.01, int(n[i]), ha='center', va='bottom', fontsize=25)
        
        # # 2.1 添加刻度
        # min_1 = min_value
        # max_1 = max_value
        # division_number = len(ranges)+1
        # t1 = np.linspace(min_1, max_1, num=division_number)
        # plt.xticks(t1)
        
        # # 2.2 添加网格和标题
        # plt.grid(linestyle='--')

        # plt.xlabel("difference", fontdict={'size': 16})
        # plt.ylabel("numbers", fontdict={'size': 16})
        # plt.title(f"the histogram of residual between pred and true of {name}_{flag}", fontdict={'size': 20})
        # # plt.legend(loc='best')
        # plt.savefig(f"./residuals_{name}_{flag}_period.png")
        # plt.show()
        
        

            
        
        
        
        
        
        