# Calibration-CDS

This is the code for the KDD'24 paper - "Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift".


## How to use?

We use PatchTST and ETTh1 datasets as an example.

1. `sh scripts/PatchTST/train/ETTh1.sh` This is to first train the forecasting models. Here all scripts in scripts/PatchTST/train are simply copied from the original PatchTST repository, but adding an extra `--run_train --run_test`.
2. `sh scripts/PatchTST/detection/ETTh1.sh` This is to obtain the prediction residuals for calculating the **Reconditionor** indicators. Here `--get_data_error --batch_size 1` is used.
3. `python reconditionor/calc_distribution.py` This is to calculate the **Reconditionor** indicators.
4. `sh scripts/PatchTST/adaptation/ETTh1.sh` This is to use **SOLID** to make sample-level adaptations on the forecasting models, thus making better performance. `--test_train_num 1000 --run_select_with_distance --selected_data_num 10 --adapted_lr_times 10` is used to make adaptation.


## Citation
```bibtex
@inproceedings{2024_calibration,
  title={Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift},
  author={Mouxiang Chen and Lefei Shen and Han Fu and Zhuo Li and Jianling Sun and Chenghao Liu}
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HALF111/calibration_CDS&type=Date)](https://star-history.com/#HALF111/calibration_CDS&Date)
