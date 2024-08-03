lamda=0.1
seq_lens='3'
pred_lens='3'
model_name="NLinear"
features="M"
scale=0
breakfast=1
lunch=1
supper=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/NLinear" ]; then
    mkdir ./logs/NLinear
fi


if [ ! -d "./logs/NLinear/img" ]; then
    mkdir ./logs/NLinear/img
fi

# Loop through lamda values
for seq_len in $seq_lens
do
    echo "Running with seq_len=$seq_len"
    
    for pred_len in $pred_lens
    do
        echo "Running with seq_len=$seq_len pred_len=$pred_len"
        python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path data.csv \
        --model_id weight \
        --model $model_name \
        --data weight \
        --features  $features\
        --variation 0\
        --image 1\
        --text 0\
        --fusion "NO" \
        --breakfast $breakfast\
        --lunch $lunch\
        --supper $supper\
        --scale $scale\
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 2 \
        --des 'Exp' \
        --lamda $lamda \
        --checkpoints "/share/ckpt/yxgui/LTSF-CKPT/NLinear/img" \
        --itr 1 --batch_size 32 --learning_rate 0.005 >logs/NLinear/img/$model_name'_'weight_$seq_len'_'$pred_len'_'$features"_"$breakfast$lunch$supper"_l_"$lamda.log
        echo "Job submitted for pred_len=$pred_len"
    done

    echo "Job submitted for seq_len=$seq_len"
    echo "  "
done

# #!/bin/bash
# #SBATCH -N 1
# #SBATCH --partition fvl
# #SBATCH --qos high
# #SBATCH -J bash

# #SBATCH --ntasks-per-node=1
# #SBATCH --mem=200G
# #SBATCH --gres=gpu:1
# #SBATCH --time=2-00:00:00

# # Define the lamda values to loop through
# gamma_values="0.1 0.6667"

# lamda=0.1
# seq_len=5
# pred_len=5
# model_name="NLinear"
# features="M"
# profile=0
# scale=0



# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi

# # Loop through lamda values
# for gamma in $gamma_values
# do
#     echo "Running with gamma=$gamma"
    
#     python -u run_longExp.py \
#       --is_training 1 \
#       --root_path ./dataset/ \
#       --data_path new_8_all_feat1.csv \
#       --model_id weight \
#       --model $model_name \
#       --data weight \
#       --features M \
#       --image 1\
#       --user_profile 0\
#       --scale 0\
#       --seq_len $seq_len \
#       --pred_len $pred_len \
#       --enc_in 2 \
#       --des 'Exp' \
#       --lamda $lamda \
#       --gamma $gamma \
#       --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'weight_$seq_len'_'$pred_len'_'$features'_L_'$lamda'_G_'$gamma'_scal_'$scale'_prof_'$profile.log
    
#     echo "Job submitted for gamma=$gamma"
# done






# #!/bin/bash
# #SBATCH -N 1
# #SBATCH --partition fvl
# #SBATCH --qos medium
# #SBATCH -J trains


# #SBATCH --ntasks-per-node=1
# #SBATCH --mem=200G
# #SBATCH --gres=gpu:1
# #SBATCH --time=2-00:00:00

# seq_len=5
# model_name="NLinear"
# features="M"


# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi

# python -u run_longExp.py \
#   --is_training 1 \
#   --root_path ./dataset/ \
#   --data_path new_8_all_feat1.csv \
#   --model_id weight$seq_len'_'5 \
#   --model $model_name \
#   --data weight \
#   --features M \
#   --image 1 \
#   --seq_len $seq_len \
#   --pred_len 5 \
#   --enc_in 2 \
#   --des 'Exp' \
#   --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'weight_$seq_len'_'5_$features'_all_his_food'.log


