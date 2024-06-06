
seq_len=104
model_name=PatchTST

root_path_name=./dataset/
data_path_name=national_illness.csv
model_id_name=national_illness
data_name=custom

random_seed=2021
# for pred_len in 24 36 48 60
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
      --batch_size 16 \
      --learning_rate 0.0025 \
      --run_test \
      --test_train_num 200 --run_select_with_distance --selected_data_num 3 --adapted_lr_times 2
    #   >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done