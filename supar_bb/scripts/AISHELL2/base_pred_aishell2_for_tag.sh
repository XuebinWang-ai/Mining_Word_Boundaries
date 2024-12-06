#! /bin/bash

mode="predict"

data_root=../data
data_name="wo_punc"

data_name="AISHELL2"
data_dir="${data_root}/${data_name}"

device=0

for seed in 0 1 3; do
base_model_path=../exp-bitag/ctb5_wo_punc/bi.wo_punc.ctb5.crf-cws.bert.epc=5.seed${seed}/model

# constrained decode
for threshold in  0.0 0.1 0.5 0.9; do
    data=${data_dir}/filtered/seed${seed}/${data_name}.filter_${threshold}.txt
    pred_mid=${data_dir}/filtered/temp

    pred=${data_dir}/train2/seed${seed}/${data_name}.filter_${threshold}.txt
    mkdir "${data_dir}/train2/seed${seed}/"
    echo ${data}
    CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
        --device $device \
        --path ${base_model_path} \
        --data ${data} \
        --pred ${pred_mid} \
        --constrained \

    echo 'Transfer to single tag...'
    python scripts/analysis/bi2one_tag.py \
        ${pred_mid} \
        ${pred}
        
    echo ${pred}
    echo 'Finish.'
done

done

