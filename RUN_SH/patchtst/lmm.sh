
seq_len=3
pred_lens="3"
model_name="PatchTST"
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


lamdas="0.1"

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/PatchTST" ]; then
    mkdir ./logs/PatchTST
fi

if [ ! -d "./logs/PatchTST/lmm" ]; then
    mkdir ./logs/PatchTST/lmm
fi

# Loop through lamda values
for lamda in $lamdas
do
    echo "Running with lambda=$lamda"
    for pred_len in $pred_lens
    do
        python -u run_longExp.py \
            --is_training 1 \
            --lamda $lamda \
            --root_path ./dataset/ \
            --data_path data.csv \
            --model_id weight \
            --model $model_name \
            --data weight \
            --features $features \
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
            --e_layers $e_layers \
            --n_heads $heads \
            --enc_in 4 \
            --dec_in 1 \
            --c_out 7 \
            --des 'Exp' \
            --d_model $d_model\
            --d_ff $d_ff \
            --dropout 0.2\
            --fc_dropout 0.2\
            --head_dropout 0\
            --patch_len 3\
            --stride 1\
            --checkpoints "/share/ckpt/yxgui/LTSF-CKPT/PatchTST/lmm" \
            --itr 1 --batch_size 32 --learning_rate 0.005 >logs/PatchTST/lmm/$model_name'_'weight_$seq_len'_'$pred_len'_'$features'_ins_scal_'$scale'_depth_'$e_layers'_d_model'$d_model'_nhead'$h'd_ff'$d_ff"_l_"$lamda.log
        
        echo "Job submitted for pred_len=$pred_len"
    done
done