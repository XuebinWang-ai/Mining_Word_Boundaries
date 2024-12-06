## Experiment

### STEPS
1. 去除标点 
`remove_punc.py` 位于 `data/ctb5/remove_punc.py`

For example:
```bash
python remove_punc.py ../zx/zx.unlabeled.pinyin.conll wo_punc/zx_unlabeled_wo_punc.pinyin.seg
```

2. Baseline 概率预测
脚本位于 `supar_bb/scripts/analysis/pred_zx_for_prob.sh`

For example:
```bash
CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
    --device $device \
    --path ../exp-bitag-wxb/ctb5_no_punc/bi.wo_punc.ctb5.crf-cws.bert.epc=5.seed0/model.wxb \
    --data ${data_dir}/zx_unlabeled_wo_punc.pinyin.fine_grain.seg \
    --compute_marg_probs \
    --pred ../exp-bitag-wxb/ctb5_no_punc_CTT/zx-unlabeled-fine-grain/zx_unlabeled.pinyin.fine_grain.${mode}.prob.conll \
    > ../exp-bitag-wxb/ctb5_no_punc_CTT/log/zx_unlabeled.pinyin.fine_grain.${mode}.log 2>&1 &
```

3. 概率筛选

脚本位于 `supar_bb/scripts/analysis/marg_prob_filter.sh`

For example:
```bash
echo > /data4/xbwang/Speech_CWS/CWS_wxb/exp-bitag-wxb/ctb5_no_punc_CTT/log/zx_filtered_threshold.log

for threshold in 1.10 1.00 0.95 0.94 0.93 0.92 0.91 0.90 0.85 0.80 0.50 0.40 0.30 0.20 0.10 0.05 0.01;
  do
    python scripts/analysis/filter_according_marg_prob.py \
            /data4/xbwang/Speech_CWS/CWS_wxb/data/ctb5/wo_punc/zx_unlabeled_wo_punc.pinyin.fine_grain.seg \
            /data4/xbwang/Speech_CWS/CWS_wxb/exp-bitag-wxb/ctb5_no_punc_CTT/zx-unlabeled-fine-grain/zx_unlabeled.pinyin.fine_grain.predict.prob.conll \
            ${threshold} \
            /data4/xbwang/Speech_CWS/CWS_wxb/exp-bitag-wxb/ctb5_no_punc_CTT/zx-unlabeled-fine-grain/zx_filtered_${threshold}.conll \
            >> /data4/xbwang/Speech_CWS/CWS_wxb/exp-bitag-wxb/ctb5_no_punc_CTT/log/zx_unlabeled_fine-grain_filtered_threshold.log
  done
```

4. 停顿限制解码，bi-tag to onetag

脚本位于 `supar_bb/scripts/exp-bitag-wxb.ctb5.pinyin/pred_zx_unlabeled.sh`

```bash
CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
        --device $device \
        --path ../exp-bitag-wxb/ctb5_no_punc/${method}/model.wxb \
        --data ${data_dir}/zx.fine_grain_filtered_0.90.conll \
        --pred ../exp-bitag-wxb/ctb5_no_punc/zx.fine_grain.out.pinyin.filtered.conll \
        --constrained \
        > ../exp-bitag-wxb/ctb5_no_punc/zx.fine_grain.${data_name}.${mode}.pinyin.filtered.log 2>&1


echo 'Transfer to single tag...'
python ../exp-bitag-wxb/ctb5_no_punc_CTT/temp.py \
    ../exp-bitag-wxb/ctb5_no_punc/zx.fine_grain.out.pinyin.filtered.conll \
    ../exp-bitag-wxb/ctb5_no_punc/zx.fine_grain.out.one_tag.pinyin.filtered.conll

echo 'Finish.'
```

5. Complete-then-train (CTT) 训练

脚本位于：`supar_bb/scripts/exp-bitag-wxb.ctb5.pinyin/complete.2.sh`
```bash
CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_cws $mode \
        --conf ../config/ctb.cws.${encoder}.ini  \
        --build \
        --device 0 \
        --seed ${seed} \
        --path ../exp-bitag-wxb/ctb5_no_punc/${method}/model.3.wxb \
        --encoder ${encoder} \
        --bert ${bert} \
        --epochs ${epochs} \
        --batch-size 1000 \
        --train ${data_dir}/${corpus_name}.train.seg \
        --train2 $train2 \
        --use_train2 \
        --dev ${data_dir}/${corpus_name}.dev.seg \
        --test ${data_dir}/${corpus_name}.test.seg \
        > ../log/ctb5_no_punc/${method}/${mode}.3.log 2>&1 &
```

6. 评估CTT训练结果
脚本位于：`supar_bb/scripts/exp-bitag-wxb.ctb5.pinyin/eval_zx_labeled_2.sh`
```bash
CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
        --device 0 \
        --path ../exp-bitag-wxb/ctb5_no_punc/${method}/model.3.wxb \
        --data ${data_dir}/zx.test.word.seg \
	>> ${log_file} 2>&1
```
