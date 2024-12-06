## Environment

```bash
conda activate speech_cws
```

## Train (baseline, BMES)

```bash
bash scripts/train_crf_cws_bert_wxb.sh
```
Model and log files are saved in the `exp/ctb5/` and `log/ctb5` folders, respectively.

## Train (bi-label. BM, MM, ME...)

```bash
cd supar_bb/
bash scripts/train_crf_cws_bert_wxb_bi.sh
```
Model and log files are saved in 
the `exp-bitag-wxb/ctb5/` and `log/ctb5` folders, respectively.

## Train(bi-label. Ctb5+zx_unlabeled data)
```bash
cd supar_bb/
bash scripts/train_crf_cws_bert_train2_wxb.sh
```
Model and log files are saved in 
the `exp-bitag-wxb/ctb5_zx_unlabeled_train2` and `ctb5_zx_unlabeled_train2`, respectively.

## Eval (baseline, BMES)

```bash
bash scripts/eval_crf_cws_bert.sh
```

## Predict (bi-label. BM, MM, ME...)
用 bitag 的 baseline 跑 `data/zx/result.txt.conll` or `data/ctb5/wo_punc/zx.result.wo_punc.seg`
这个conll文件是之前包含停顿的zx文件。
```bash
cd supar_bb/
bash scripts/pred_zx.sh
```

### Filter after Prediction
根据 baseline 的概率筛选
`all_sent_with_boundary.txt` 是包含所有文本，但是加入了语音停顿；
`result.txt` 是只包含有语音停顿的句子对应的文本（且有分词gold）。
```bash
bash scripts/marg_prob_filter.sh
```

## Complete-then-train
### train bi-tag baseline using ctb5 data w/wo punc
The data is included punc data and no_punc data
```bash
cd supar_bb/
bash scripts/train_crf_cws_bert_w_wo_punc_wxb.sh
```

After training, use the baseline to complete the labels of zx_unlabeled data
```bash
cd supar_bb/
bash scripts/pred_zx_w_wo_wxb.sh
```
The predicted result is `CWS_wxb/exp-bitag-wxb/ctb_no_punc_CTT/zx-result.out.conll`,
which need to be transfer to `.../zx-result.out.one_tag.conll`

Finally, we use ctb5+zx-result.out.one_tag.conll data to train a new model from scratch.
```bash
cd supar_bb/
bash scripts/train_complete_w_wo_wxb.sh
```
