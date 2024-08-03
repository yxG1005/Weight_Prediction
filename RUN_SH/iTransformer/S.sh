scale=0
seq_lens="3"
pred_lens="3"
model_name="iTransformer"
features="S"

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

if [ ! -d "./logs/iTransformer/S" ]; then
    mkdir ./logs/iTransformer/S
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
            --image 1\
            --text 0\
            --fusion "NO" \
            --breakfast $breakfast\
            --lunch $lunch\
            --supper $supper\
            --seq_len $seq_len \
            --pred_len $pred_len \
            --scale $scale \
            --e_layers $e_layers \
            --n_heads $heads \
            --enc_in 7 \
            --dec_in 862 \
            --c_out 7 \
            --des 'Exp' \
            --d_model $d_model\
            --d_ff $d_ff \
            --checkpoints "/share/ckpt/yxgui/LTSF-CKPT/iTransformer/S" \
            --itr 1 --batch_size 32 --learning_rate 0.005 >logs/iTransformer/S/$model_name'_'weight_$seq_len'_'$pred_len'_'$features'_depth_'$e_layers'_d_model'$d_model'_nhead'$h'd_ff'$d_ff.log
        
 
    done
done