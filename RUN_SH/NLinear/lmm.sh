lamda=0.1

gamma=0
seq_lens='3'
pred_lens='3'
model_name="NLinear"
features="M"
scale=0
breakfast=1
lunch=1
supper=1



if [ ! -d "./logs/NLinear" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/NLinear" ]; then
    mkdir ./logs/NLinear
fi


if [ ! -d "./logs/NLinear/lmm" ]; then
    mkdir ./logs/NLinear/lmm
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
        --image 0\
        --text 1\
        --text_from_img \
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
        --checkpoints "/share/ckpt/yxgui/LTSF-CKPT/NLinear/lmm" \
        --itr 1 --batch_size 32 --learning_rate 0.005 >logs/NLinear/lmm/$model_name'_'weight_$seq_len'_'$pred_len'_'$features"_"$breakfast$lunch$supper"_l_"$lamda'_'.log
        echo "Job submitted for pred_len=$pred_len"
    done

    echo "Job submitted for seq_len=$seq_len"
    echo "  "
done
