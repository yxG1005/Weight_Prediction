Lambda=0.1
seq_lens="3"
pred_lens="5"
model_name="PatchTST"
features="M"

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

if [ ! -d "./logs/PatchTST" ]; then
    mkdir ./logs/PatchTST
fi

if [ ! -d "./logs/PatchTST/lmm" ]; then
    mkdir ./logs/PatchTST/lmm
fi

# Loop through Lambda values
for seq_len in $seq_lens
do
    for pred_len in $pred_lens
    do
        echo "Running with seq_len=$seq_len pred_len=$pred_len"
        python -u run_longExp.py \
            --is_training 1 \
            --Lambda $Lambda \
            --root_path ./dataset/ \
            --data_path data.csv \
            --model_id weight \
            --model $model_name \
            --features $features \
            --image 0\
            --text 1\
            --text_from_img \
            --fusion "NO" \
            --breakfast $breakfast\
            --lunch $lunch\
            --supper $supper\
            --seq_len $seq_len \
            --pred_len $pred_len \
            --e_layers $e_layers \
            --n_heads $heads \
            --enc_in 4 \
            --d_model $d_model\
            --d_ff $d_ff \
            --dropout 0.2\
            --fc_dropout 0.2\
            --head_dropout 0\
            --patch_len 3\
            --stride 1\
            --checkpoints "PatchTST/lmm" \
            --itr 1 --batch_size 32 --learning_rate 0.005 >logs/PatchTST/lmm/$model_name'_'weight_$seq_len'_'$pred_len'_'$features'_depth_'$e_layers'_d_model'$d_model'_nhead'$heads'd_ff'$d_ff"_l_"$Lambda.log
        
        echo "Job submitted for Running with seq_len=$seq_len pred_len=$pred_len"
    done
done