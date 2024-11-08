
seq_lens="3 5"
pred_lens="3 5"
model_name="PatchTST"
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

if [ ! -d "./logs/PatchTST" ]; then
    mkdir ./logs/PatchTST
fi

if [ ! -d "./logs/PatchTST/S" ]; then
    mkdir ./logs/PatchTST/S
fi


for seq_len in $seq_lens
do
    for pred_len in $pred_lens
    do
        echo "Running with seq_len=$seq_len pred_len=$pred_len"
        python -u run_longExp.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path data.csv \
            --model_id weight \
            --model $model_name \
            --features $features \
            --image 1\
            --text 0\
            --fusion "NO" \
            --breakfast $breakfast\
            --lunch $lunch\
            --supper $supper\
            --seq_len $seq_len \
            --pred_len $pred_len \
            --e_layers $e_layers \
            --n_heads $heads \
            --enc_in 1 \
            --d_model $d_model\
            --d_ff $d_ff \
            --dropout 0.2\
            --fc_dropout 0.2\
            --head_dropout 0\
            --patch_len 3\
            --stride 1\
            --checkpoints "PatchTST/S" \
            --itr 1 --batch_size 32 --learning_rate 0.005 >logs/PatchTST/S/$model_name'_'weight_$seq_len'_'$pred_len'_'$features'_depth_'$e_layers'_d_model'$d_model'_nhead'$heads'd_ff'$d_ff.log
        
        echo "Job submitted for Running with seq_len=$seq_len pred_len=$pred_len"
    done
done