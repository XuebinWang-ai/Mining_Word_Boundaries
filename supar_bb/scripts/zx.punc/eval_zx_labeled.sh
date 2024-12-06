#! /bin/bash

### with-punc evaluate
# region
device=0
data_root=../data
encoder="bert"
mode="evaluate"

log_file=../exp-bitag-wxb/zx.punc/log/${mode}.punc.zx.log

echo > $log_file

for seed in 0 1 2;
do
method="bi.segment.crf-cws.bert.epc=5.seed${seed}"
# zx

echo seed=$seed >> $log_file

CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
        --device 0 \
        --path ../exp-bitag-wxb/ctb5/${method}/model.wxb \
        --data ${data_root}/zx-labeled/dev.word.conll \
	>> $log_file 2>&1 

CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
        --device 0 \
        --path ../exp-bitag-wxb/ctb5/${method}/model.wxb \
        --data ${data_root}/zx-labeled/test.word.conll \
	>> $log_file 2>&1 

done

# endregion