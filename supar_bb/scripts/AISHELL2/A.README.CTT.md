## Experiment

### STEPS
1. 去除标点 
```bash
cd data/
python remove_punc.py AISHELL2/AISHELL2.unlabeled.char.conll AISHELL2/AISHELL2.unlabeled_wo_punc.char.seg
```

2. Baseline 概率预测
脚本位于 `supar_bb/scripts/exp-bitag-wxb.ctb5-AISHELL2/baseline_pred.sh`

For example:
```bash
CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
        --device $device \
        --path ../exp-bitag-wxb/ctb5_no_punc/bi.wo_punc.ctb5.crf-cws.bert.epc=5.seed0/model.wxb \
        --data ${data_dir}/AISHELL2.unlabeled_wo_punc.char.seg \
        --compute_marg_probs \
        --pred ../exp-bitag-wxb/ctb5_no_punc_CTT/${fg}-unlabeled-char-level/${fg}_unlabeled.char.${mode}.prob.conll \
        > ../exp-bitag-wxb/ctb5_no_punc_CTT/log/${fg}_unlabeled.char.${mode}.log 2>&1 &

```

3. 概率筛选

脚本位于 `supar_bb/scripts/exp-bitag-wxb.ctb5-AISHELL2/marg_prob_filter.sh`，


For example:
```bash
fg="AISHELL2"

echo > ../exp-bitag-wxb/ctb5_no_punc_CTT/log/${fg}_filtered_threshold.char.log

for threshold in 1.10 1.00 0.95 0.90 0.85 0.80 0.50 0.40 0.30 0.20 0.10 0.05 0.01;
  do
    python scripts/analysis/filter_according_marg_prob.py \
            ../data/${fg}/${fg}.unlabeled_wo_punc.char.seg \
            ../exp-bitag-wxb/ctb5_no_punc_CTT/${fg}-unlabeled-char-level/${fg}_unlabeled.char.predict.prob.conll \
            ${threshold} \
            ../exp-bitag-wxb/ctb5_no_punc_CTT/${fg}-unlabeled-char-level/${fg}_filtered_${threshold}.conll \
            >> ../exp-bitag-wxb/ctb5_no_punc_CTT/log/${fg}_filtered_threshold.char.log
  done
```

4. 停顿限制解码，bi-tag to onetag

脚本位于 `supar_bb/scripts/exp-bitag-wxb.ctb5-AISHELL2/pred_zx_unlabeled.sh`

```bash
CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
        --device $device \
        --path ../exp-bitag-wxb/ctb5_no_punc/${method}/model.wxb \
        --data ../exp-bitag-wxb/ctb5_no_punc_CTT/${type_dir}/${fg}_filtered_${threshold}.conll \
        --pred ../exp-bitag-wxb/ctb5_no_punc_CTT/${type_dir}/${fg}.out-bitag.char.filtered_${threshold}.conll \
        --constrained \
        > ../exp-bitag-wxb/ctb5_no_punc_CTT/non-sence-log/${fg}.char.${data_name}.${mode}.filtered.log 2>&1


echo 'Transfer to single tag...'
python ../exp-bitag-wxb/ctb5_no_punc_CTT/bi2one_tag.py \
    ../exp-bitag-wxb/ctb5_no_punc_CTT/${type_dir}/${fg}.out-bitag.char.filtered_${threshold}.conll \
    ${data_dir}/${fg}.onetag.char.conll

echo 'Finish.'
```

5. Complete-then-train (CTT) 训练

**模型保存在 `../exp-bitag-wxb/ctb5_no_punc`**
# 问题：可不可以以 zx.dev.seg为评测目标而不是 ctb.dev.seg (可以，但是效果不如后者)
脚本位于：`supar_bb/scripts/exp-bitag-wxb.ctb5-AISHELL2/CTT.wo_punc.char.sh`
```bash
CUDA_VISIBLE_DEVICES=$device nohup python -u -m supar.cmds.crf_cws $mode \
        --conf ../config/ctb.cws.${encoder}.ini  \
        --build \
        --device 0 \
        --seed ${seed} \
        --path ../exp-bitag-wxb/ctb5_no_punc/${method}/model.wxb \
        --encoder ${encoder} \
        --bert ${bert} \
        --epochs ${epochs} \
        --batch-size 1000 \
        --train ${data_dir}/${corpus_name}.train.seg \
        --train2 $train2 \
        --use_train2 \
        --dev ${data_dir}/${corpus_name}.dev.seg \
        --test ${data_dir}/${corpus_name}.test.seg \
        > ../log/ctb5_no_punc/${method}/${mode}.log 2>&1 &
```

6. 评估CTT训练结果
脚本位于：`supar_bb/scripts/exp-bitag-wxb.ctb5.pinyin/eval_zx_labeled_2.sh`
```bash
CUDA_VISIBLE_DEVICES=$device python -u -m supar.cmds.crf_cws $mode \
        --device 0 \
        --path ${model_path} \
        --data ${data_dir}/zx.dev.word.seg \
	>> ${log_file} 2>&1
```