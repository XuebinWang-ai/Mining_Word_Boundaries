#! /bin/bash

data_root=../data
data_name="wo_punc"

data_name="zx"
data_dir="${data_root}/${data_name}"
encoder="bert"
epochs=10
device=0

for seed in 0 1 3; do
    base_model_path=../exp-bitag/ctb5_wo_punc/bi.wo_punc.ctb5.crf-cws.bert.epc=5.seed${seed}/model

    data=${data_dir}/${data_name}.example.pauese
    pred_mid=${data_dir}/filtered/temp

    train2=${data_dir}/train2/seed${seed}/for-self-training.txt
    mkdir "${data_dir}/train2/seed${seed}/"
    echo ${data}

    # crf decode for self-training
    mode="predict"
    CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
        --device $device \
        --path ${base_model_path} \
        --data ${data} \
        --pred ${pred_mid} \

    echo 'Transfer to single tag...'
    python scripts/analysis/bi2one_tag.py \
        ${pred_mid} \
        ${train2}

    echo ${train2}
    echo 'Finish.'

    # self-training
    mode="train"
    model_path="../exp-bitag/${data2_name}/seed${seed}/${method}/model.self-training.pt"
    log_file=../log/${data2_name}/${method}.seed${seed}.self-training/$mode.log
    echo ${log_file}

    CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_cws $mode \
        --conf ../config/ctb.cws.${encoder}.ini  \
        --build \
        --device $device \
        --seed ${seed} \
        --path "${model_path}" \
        --encoder ${encoder} \
        --bert ${bert} \
        --epochs ${epochs} \
        --batch-size 1000 \
        --train ${data_dir}/train.seg \
        --train2 $train2 \
        --use_train2 \
        --dev ${dev} \
        --test ${test} \
        > ${log_file} 2>&1 &
done


