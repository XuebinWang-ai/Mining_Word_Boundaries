## Mining Word Boundaries from Speech-Text Parallel Data for Cross-domain Chinese Word Segmentation

This is the repo for this paper, a novel approach for mining pauses for Cross-domain CWS. 
This paper has been accepted in COLING 2025.

### Abstract

Inspired by early research on exploring naturally annotated data for Chinese Word Segmentation (CWS), and also by recent research on integration of speech and text processing, this work for the first time proposes to explicitly mine word boundaries from speech-text parallel data. 
We employ the Montreal Forced Aligner (MFA) toolkit to perform character-level alignment on speech-text data, giving pauses as candidate word boundaries. 
Based on detailed analysis of collected pauses, we propose an effective probability-based strategy for filtering unreliable word boundaries. 
To more effectively utilize word boundaries as extra training data, we also propose a robust complete-then-train (CTT) strategy. 
We conduct cross-domain CWS experiments on two target domains, i.e., ZX and AISHELL2. 
We have annotated about 1,000 sentences as the evaluation data of AISHELL2. 
Experiments demonstrate the effectiveness of our proposed approach. 

### Installation

```
pip install -r requirements.txt
```

### Preparation

*Data*: Chinese Penn Treebank 5 (CTB5) and [AISHELL2](https://www.aishelltech.com/aishell_2).

*Mining puases*: Utilize [MFA](https://mfa-models.readthedocs.io/en/latest/index.html) to mine pauses 
from AISHELL2 and ZhuXian. 
Place the obtained pause data in the `data/AISHELL2/` and `data/ZX/` folders, 
the file format is like `data/AISHELL2/AISHELL2.example.pauses` and `data/ZX/ZX.example.pauses`.

*Bert-Base-Chinese*: download from [hugging face](https://huggingface.co/google-bert/bert-base-chinese) 
and place it in the `bert-base-chinese/` folder.

*Evaluation data of AISHELL2*: We manually annotated the AISHELL2 dev/test datasets and 
placed it in the `data/AISHELL2/Eval` folder. 

### Training

```
cd supar_bb/
```

### Baseline
Train baseline (with punctuation / without punctuation)
```
sh scripts/train_crf_cws_bert_baseline.sh
```

### Complete-Then-Train Strategy
Take AISHELL2 as an example.

0. Self-training method.
    ```
    sh scripts/AISHELL2/self-training.sh
    ```

1. Utilize base model to predict probabilities of pauses.
    ```
    sh scripts/AISHELL2/base_pred_aishell2_for_prob.sh
    ```

2. Filter word boundaries according to probability.
    ```
    sh scripts/AISHELL2/marg_prob_filter.sh
    ```

3. Restrict decoding to filtered word boundaries.
    ```
    sh scripts/AISHELL2/base_pred_aishell2_for_tag.sh
    ```

4. Train on AISHELL2 and CTB5 dataset.
    ```
    sh scripts/AISHELL2/CTT.sh
    ```
