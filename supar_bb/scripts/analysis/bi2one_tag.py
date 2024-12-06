# -*- encoding: utf-8 -*-
# @Time     :2024/04/29 09:44:07
# @author   :Wxb

## 将两个tag变成一个tag
import os
import sys

if __name__ == "__main__":
    file_in = sys.argv[1]
    file_out = sys.argv[2]
    
    fo = open(file_out, 'w', encoding='utf-8')
    fin = open(file_in, 'r', encoding='utf-8')
    for line in fin:
        if not line.strip():
            print(file=fo)
        else:
            char, tag = line.strip().split()
            assert len(tag) == 2
            print(f"{char}\t{tag[0]}", file=fo)
    
    fo.close()
    fin.close()
    
