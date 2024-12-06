#! /bin/bash

# w/ punc ctb5, train baseline
for seed in 0 1 3; do
device=$seed
corpus_name="ctb5"
mkdir -p "../log/${corpus_name}"
data_name="segment"
data_dir="../data/${corpus_name}/${data_name}"
encoder="bert"
bert="../bert-base-chinese"
epochs=5
mode="train"
method="bi.${data_name}.crf-cws.${encoder}.epc=${epochs}.seed${seed}"
CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_cws train \
        --conf ../config/ctb.cws.${encoder}.ini  \
        --build \
        --device $device \
        --seed ${seed} \
        --path ../exp-bitag/${corpus_name}/${method}/model \
        --encoder ${encoder} \
        --bert ${bert} \
        --epochs ${epochs} \
        --batch-size 1000 \
        --train ${data_dir}/train.seg \
        --dev ${data_dir}/dev.seg \
        --test ${data_dir}/test.seg \
        > ../log/${corpus_name}/${method}.${mode}.log 2>&1 &
done
# tail -f ../log/${corpus_name}/${method}.${mode}.log

# w/o punc ctb5, train baseline
for seed in 0 1 3; do
# seed=3
device=$seed
corpus_name="ctb5"
mkdir -p "../log/${corpus_name}_wo_punc"
data_name="wo_punc"
data_dir="../data/${corpus_name}/${data_name}"
encoder="bert"
bert="../bert-base-chinese"
epochs=5
mode="train"
method="bi.${data_name}.${corpus_name}.crf-cws.${encoder}.epc=${epochs}.seed${seed}"
CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_cws train \
        --conf ../config/ctb.cws.${encoder}.ini  \
        --build \
        --device $device \
        --seed ${seed} \
        --path ../exp-bitag/${corpus_name}_wo_punc/${method}/model \
        --encoder ${encoder} \
        --bert ${bert} \
        --epochs ${epochs} \
        --batch-size 1000 \
        --train ${data_dir}/train.seg \
        --dev ${data_dir}/dev.seg \
        --test ${data_dir}/test.seg \
        > ../log/${corpus_name}_wo_punc/${method}.${mode}.log 2>&1 &

echo ../log/${corpus_name}_wo_punc/${method}.${mode}.log

done
