# 8.Illness
seq_len=104
root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

model_name=PatchTST
gpu=0

for pred_len in 24
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
      --patch_len 24\
      --stride 2\
      --des 'Exp' \
      --train_epochs 10 \
      --lradj 'constant'\
      --itr 1 \
      --learning_rate 0.0025 \
      --get_data_error --batch_size 1
done