# 2. ETTh2
seq_len=336
root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

model_name=PatchTST
gpu=0

for pred_len in 96
do
    python -u run_longExp.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 --learning_rate 0.0001 \
      --get_data_error --batch_size 1
done
