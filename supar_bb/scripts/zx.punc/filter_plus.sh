#! /bin/bash

fg='zx'
data=../data/${fg}/zx.unlabeled.char.nums.conll
length=50  # 50ms

echo > ../exp-bitag-wxb/${fg}.punc/log/${fg}_filtered_threshold.le${length}.log
for seed in 0 1 2; do

# for threshold in  0.30 0.50;
for threshold in  0.30;
  do
    echo $threshold
    
    prob=../exp-bitag-wxb/${fg}.punc/seed${seed}/zx_unlabeled.prob.seed${seed}.conll
    out=../exp-bitag-wxb/${fg}.punc/seed${seed}/${fg}.le${length}_pause.plus.filtered_${threshold}.conll

    python scripts/analysis/filter_prob_length.py \
            ${threshold} \
            ${length} \
            ${prob} \
            ${data} \
            ${out} \
            >> ../exp-bitag-wxb/${fg}.punc/log/${fg}_filtered_threshold.le${length}.log
  done

done
