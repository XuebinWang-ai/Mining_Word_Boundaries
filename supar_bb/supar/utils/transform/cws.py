from supar.utils.logging import progress_bar
from supar.utils.transform.transform import Sentence, Transform
from supar.utils.transform.conll import CoNLL

import pdb
import os


class CWSCoNLLSentence(Sentence):

    def __init__(self, transform, lines):
        super().__init__(transform)

        self.values = []
        # record annotations for post-recovery
        self.annotations = dict()
        # print(lines)
        for i, line in enumerate(lines):
            value = line.split('\t')
            self.annotations[len(self.values)] = line
            self.values.append(value)
        # print(self.values)
        self.values = list(zip(*self.values))

    def __repr__(self):
        chars = self.values[0]
        tags = getattr(self, 'tags', None)
        marg_probs = getattr(self, 'probs', None)
        assert tags is not None and len(chars) == len(tags)+1, f"{len(chars)}, {len(tags)+1}"
        if marg_probs is not None:
            assert len(chars) == len(marg_probs)
        
        tags = tags + [tags[-1][-1] + 's']
        lines = []
        #pdb.set_trace()
        
        for i in range(len(chars)):
            lines.append(chars[i] + '\t' + tags[i])
            if marg_probs is not None:
                probs = ['%.3f' % p for p in marg_probs[i]]
                lines[-1] += '\t' + '_'.join(probs)
        return '\n'.join(lines) + '\n'
        if getattr(self, 'segs', None) is not None:
            words = []
            for b, e in self.segs:
                words.append(''.join(chars[b:e]))
            return CoNLL.toconll(words)
        else:
            lines = ['\t'.join(value) for value in zip(*self.values)]
        return '\n'.join(lines) + '\n'


class CWSCoNLL(Transform):

    fields = ['FORM', 'TAG']

    def __init__(self,
                 FORM=None, TAG=None):
        super().__init__()

        self.FORM = FORM
        self.TAG = TAG

    @property
    def src(self):
        return self.FORM,

    @property
    def tgt(self):
        return self.TAG,

    @classmethod
    def recover_words(cls, tags, is_pred):
        """ Transform a `bmes` tag sequence to a span-based sequence.
        NOTE: the tag sequence should be legal, i.e., forbid transitions like `b` -> `s`

        Args:
            tags (list[str]): [b, m, e, s, s, ...]
            [be es ss ... ]
        Returns:
            [(0, 1), (2, 3), ...]
        """
        if is_pred:
            tags = tags + [tags[-1][-1] + 's']
        spans = []
        for i, t in enumerate(tags):
            assert len(t) == 2
            if t[0] == 'b' or t[0] == 's':
                spans.append([i, i+1])
            elif t[0] == 'm' or t[0] == 'e':
                spans[-1][1] += 1
            else:
                assert False
        return [tuple(span) for span in spans]

    def load(self, data, lang=None, max_len=None, **kwargs):
        if isinstance(data, str) and os.path.exists(data):
            with open(data, 'r') as f:
                lines = [line.strip() for line in f]
        
        print(lines)
        i, start, sentences = 0, 0, []
        kk = []
        for line in progress_bar(lines):
            if not line:
                sentences.append(CWSCoNLLSentence(self, lines[start:i]))
                kk.append(lines[start: i])
                start = i + 1
            i += 1
        
        print(kk)
        print(sentences)
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences
