lamda=0.1
seq_lens="3"
pred_lens="5"
model_name="iTransformer"
features="M"
profile=0
scale=0

e_layers=1
d_model=16
heads=2
d_ff=32

breakfast=1
lunch=1
supper=1




if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/iTransformer" ]; then
    mkdir ./logs/iTransformer
fi

if [ ! -d "./logs/iTransformer/txt" ]; then
    mkdir ./logs/iTransformer/txt
fi

# Loop through lamda values
for seq_len in $seq_lens
do
    echo "Running with breakfast=$breakfast  lunch=$lunch  supper=$supper"
    for pred_len in $pred_lens
    do
        python -u run_longExp.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path data.csv \
            --model_id weight \
            --model $model_name \
            --data weight \
            --features $features \
            --variation 0\
            --image 0\
            --text 1\
            --fusion "NO" \
            --breakfast $breakfast\
            --lunch $lunch\
            --supper $supper\
            --scale $scale\
            --seq_len $seq_len \
            --pred_len $pred_len \
            --e_layers $e_layers \
            --n_heads $heads \
            --enc_in 7 \
            --dec_in 862 \
            --c_out 7 \
            --des 'Exp' \
            --d_model $d_model\
            --d_ff $d_ff \
            --lamda $lamda \
            --checkpoints "/share/ckpt/yxgui/LTSF-CKPT/iTransformer/txt" \
            --itr 1 --batch_size 32 --learning_rate 0.005 >logs/iTransformer/txt/$model_name'_'weight_$seq_len'_'$pred_len'_'$features'_ins_scal_'$scale'_depth_'$e_layers'_d_model'$d_model'_nhead'$heads"_"'d_ff_'$d_ff'_'$breakfast$lunch$supper'_l_'$lamda.log
        
    done
done