import re
import sys
import os
from tqdm import tqdm


punctuation = frozenset("、。।，@<>”(),.:;¿?¡!\&%#*~【】，…‥「」『』〝〟″⟨⟩♪・‹›«»～′$+=-…“￥（）——|：《》？！∶┅；`︱－〖〗‘’{}[]──")
punc_str = "\\" + "\\".join(punctuation)
pattern = re.compile(rf"[{punc_str}]")


if __name__ == "__main__":
    file_in = sys.argv[1]
    file_out = sys.argv[2]

    sentences = []
    sen = []
    with open(file_in, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                if sen:
                    sentences.append(sen)
                    sen = []
                continue
            sen.append(line.split())

    new_sentences = []
    special_tags = 0
    for sen in sentences:
        length = len(sen)
        new_sen = []
        next_tag = None
        for index, (char, tag) in enumerate(sen):
            if re.match(pattern, char):
                if tag in {"m", "s", "_", "V"}:
                    continue
                else:
                    special_tags += 1
                    if tag == "e" and new_sen:
                        new_sen[-1][-1] = tag if new_sen[-1][-1] not in {"s", 'e'} else new_sen[-1][-1]
                        next_tag = None
                    elif tag == "b":
                        next_tag = tag
            else:
                if not next_tag:
                    new_sen.append([char, tag])
                else:
                    if sen[index + 1][-1] in {"b", "s"}:
                        new_sen.append([char, "s"])
                    else:
                        new_sen.append([char, "b"])
                    next_tag = None
        new_sentences.append(new_sen)
    min_len = 1000
    with open(file_out, 'w', encoding='utf-8') as f:
        for line in new_sentences:
            # 限定最短长度必须为 2
            if len(line) > 1:
                for char, tag in line:
                    f.write(f"{char}\t{tag}\n")
                
                f.write("\n")
                min_len = min(min_len, len(line))
    print("限定最短长度为", min_len)
    # print(special_tags)

"""
python remove_punc.py ../zx/zx.unlabeled.pinyin.conll wo_punc/zx_unlabeled_wo_punc.pinyin.seg
"""
