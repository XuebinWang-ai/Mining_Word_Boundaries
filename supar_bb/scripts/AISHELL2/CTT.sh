#! /bin/bash
# 2024.7.2

corpus_name="ctb5"
data_root=../data
data_type="wo_punc"

data2_name="AISHELL2"

data2_dir="${data_root}/${data2_name}/train2/"
data_dir="${data_root}/${corpus_name}/${data_type}"

encoder="bert"
bert="../bert-base-chinese"
epochs=10
mode="train"
con='.con'

dev=${data_root}/${data2_name}/Eavl/dev.306.txt
test=${data_root}/${data2_name}/Eavl/test.643.txt

for seed in 0 1 3; do
device=$seed
method="CTT.${data_type}.${corpus_name}.${data2_name}.crf-cws.${encoder}.epc=${epochs}/"

mkdir -p "../log/${data2_name}/seed${seed}/${method}"
mkdir -p "../exp-bitag/${data2_name}/seed${seed}/${method}"

for threshold in  0.0 0.1 0.5 0.9; do

model_path="../exp-bitag/${data2_name}/seed${seed}/${method}/model.threshold=${threshold}.pt"
train2=${data2_dir}/train2/seed${seed}/${data_name}.filter_${threshold}.txt

log_file=../log/${data2_name}/${method}.seed${seed}.threshold=${threshold}/$mode.log
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

done

