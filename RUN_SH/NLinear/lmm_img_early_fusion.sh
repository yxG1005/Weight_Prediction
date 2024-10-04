Lambda=0.1

seq_lens='3'
pred_lens='5'
model_name="NLinear"
features="M"
breakfast=1
lunch=1
supper=1



if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/NLinear" ]; then
    mkdir ./logs/NLinear
fi

if [ ! -d "./logs/NLinear/lmm_img_early_fusion" ]; then
    mkdir ./logs/NLinear/lmm_img_early_fusion
fi

# Loop through Lambda values
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
        --features  $features\
        --image 1\
        --text 1\
        --text_from_img \
        --fusion "early" \
        --breakfast $breakfast\
        --lunch $lunch\
        --supper $supper\
        --seq_len $seq_len \
        --pred_len $pred_len \
        --Lambda $Lambda \
        --checkpoints "NLinear/lmm_img_early_fusion" \
        --itr 1 --batch_size 32 --learning_rate 0.005 >./logs/NLinear/lmm_img_early_fusion/$model_name'_'weight_$seq_len'_'$pred_len'_'$features"_"$breakfast$lunch$supper"_l_"$Lambda.log
        echo "Job submitted for pred_len=$pred_len"
    done

    echo "Job submitted for seq_len=$seq_len"
    echo "  "
done
