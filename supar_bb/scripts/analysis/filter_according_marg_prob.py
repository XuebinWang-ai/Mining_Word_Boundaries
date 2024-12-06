

"""
天      _
地      V
不      _
仁      V

天      be      0.000_0.000_1.000_0.000_0.000_0.000_0.000_0.000_0.000_0.000_0.000_0.000_0.000_0.000_0.000_0.000
地      eb      0.000_0.000_0.000_0.000_0.000_0.000_0.000_0.000_0.789_0.000_0.000_0.211_0.000_0.000_0.000_0.000
"""

import os
import sys
import re

EPS = 1e-5

bilabels = ['bb', 'bm', 'be', 'bs',
            'mb', 'mm', 'me', 'ms',
            'eb', 'em', 'ee', 'es',
            'sb', 'sm', 'se', 'ss']

bilabels_to_id = dict([(bilabels[i], i) for i in range(len(bilabels))])

#print(bilabels)

bilabels_with_pause = ['eb', 'es', 'sb', 'ss']
bilabels_id_with_pause = [bilabels_to_id[label] for label in bilabels_with_pause]
#print(bilabels_id_with_pause)



cnt_pause = 0
cnt_pause_wrong = 0
cnt_correctly_filter = 0
cnt_wrongly_filter = 0

if __name__ == '__main__':
    assert 5 == len(sys.argv)
    fname_with_pause = sys.argv[1]
    fname_with_probs = sys.argv[2]
    threshold_filter_if_le = float(sys.argv[3])
    fname_out_conll = sys.argv[4]

    result_line = []
    with open(fname_with_pause, 'r') as fp1:
        with open(fname_with_probs, 'r') as fp2:
            for (line1, line2) in zip(fp1, fp2):
                line1, line2 = line1.strip(), line2.strip()
                result_line.append(line1)
                if '' == line1:
                    assert '' == line2
                    continue
                tokens1 = re.split('\t', line1)
                tokens2 = re.split('\t', line2)
                assert tokens1[0] == tokens2[0]
                assert len(tokens1) == 2 and len(tokens2) == 3
                assert tokens1[-1] in {'_', 'V', 'X'}
                if tokens1[-1] != '_':
                    probs = re.split('_', tokens2[-1])
                    probs = [float(p) for p in probs]
                    pause_prob = sum([probs[i] for i in bilabels_id_with_pause])
                    cnt_pause += 1
                    if 'X' == tokens1[-1]:
                        cnt_pause_wrong += 1
                    # filter
                    # if pause_prob <= threshold_filter_if_le + EPS:
                    if pause_prob <= threshold_filter_if_le - EPS:
                        if 'V' == tokens1[-1]:
                            cnt_wrongly_filter += 1
                        else:
                            cnt_correctly_filter += 1
                        result_line[-1] = tokens1[0] + '\t_'
    with open(fname_out_conll, 'w') as fp:
        print('\n'.join(result_line), file=fp)

    print('[threshold: %.2f]: %d %d (correct: %d) (wrong: %d)' % (threshold_filter_if_le, \
          cnt_pause, cnt_pause_wrong, cnt_correctly_filter, cnt_wrongly_filter))




