# -*- coding: utf-8 -*-

import os
from datetime import datetime
import copy
import itertools

import pdb
import random
import dill
import supar
import torch
import torch.nn as nn
from supar.models import WordSegmentationModel, TagWordSegmentationModel, CRFWordSegmentationModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset
from supar.utils.common import bos, pad, unk, nul, BOS, PAD, UNK, NUL, MIN
from supar.utils.embedding import Embedding
from supar.utils.field import Field, NGramField, BiLabelField
from supar.utils.fn import download
from supar.utils.logging import get_logger, progress_bar, init_logger
from supar.utils.parallel import is_master
from supar.utils.metric import SegF1Metric
from supar.utils.transform.cws import CWSCoNLL

logger = get_logger(__name__)


class WordSegmenter(Parser):

    NAME = 'word-segmenter'
    MODEL = WordSegmentationModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.CHAR, self.BICHAR, self.TRICHAR, self.BERT = self.transform.FORM

    def train(self, train, train2, dev, test, buckets=32, batch_size=5000, update_steps=1,
              verbose=True, **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating training configs.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000,
                 verbose=True, **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating evaluation configs.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, lang=None, buckets=8, batch_size=5000,
                verbose=True, **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., ``en``) or language name (e.g., ``English``) for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``None``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding unconsumed arguments for updating prediction configs.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        # return super().predict(**Config().update(locals()))
        args = self.args.update(locals())
        init_logger(logger, verbose=args.verbose)
        if self.args.constrained:
            self.transform.train()
        else:
            self.transform.eval()

        if self.args.output_toconll:
            self.transform.append(Field('segs'))
        if self.args.compute_marg_probs:
            self.transform.append(Field('probs'))

        logger.info("Loading the data")
        dataset = Dataset(self.transform, data, lang=lang)
        dataset.build(args.batch_size, args.buckets)
        logger.info(f"\n{dataset}")

        logger.info("Making predictions on the dataset")
        start = datetime.now()
        preds = self._predict(dataset.loader)
        elapsed = datetime.now() - start

        for name, value in preds.items():
            setattr(dataset, name, value)
        if pred is not None and is_master():
            logger.info(f"Saving predicted results to {pred}")
            os.makedirs(os.path.dirname(pred) or './', exist_ok=True)
            self.transform.save(pred, dataset.sentences)
        logger.info(f"{elapsed}s elapsed, {len(dataset) / elapsed.total_seconds():.2f} Sents/s")

        return dataset

    @classmethod
    def load(cls, path, reload=False, **kwargs):
        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state = torch.load(path if os.path.exists(path) else download(supar.MODEL.get(path, path), reload=reload))
        sd = state['state_dict']
        cls = supar.PARSER[state['name']] if cls.NAME is None else cls
        args = state['args'].update(args)
        model = cls.MODEL(**args)
        model.load_pretrained(state['pretrained_embed_dict'])
        model.load_state_dict(state['state_dict'], False)
        model.to(args.device)
        transform = state['transform']
        return cls(args, model, transform)

    def save(self, path):
        model = self.model
        if hasattr(model, 'module'):
            model = self.model.module
        args = model.args
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained_embed_dict = {name: state_dict.pop(f'{name}.weight', None)
                                 for name in ('pretrained_char_embed', 'pretrained_bichar_embed', 'pretrained_trichar_embed')}
        state = {'name': self.NAME,
                 'args': args,
                 'state_dict': state_dict,
                 'pretrained_embed_dict': pretrained_embed_dict,
                 'transform': self.transform}
        torch.save(state, path, pickle_module=dill)

    def _train(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _evaluate(self, loader):
        raise NotImplementedError

    @torch.no_grad()
    def _predict(self, loader):
        raise NotImplementedError

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        raise NotImplementedError


class TagWordSegmenter(WordSegmenter):

    NAME = 'tag-word-segmenter'
    MODEL = TagWordSegmentationModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.CHAR, self.BICHAR, self.TRICHAR, self.BERT = self.transform.FORM
        self.TAG = self.transform.TAG

    @classmethod
    def load(cls, path, reload=False, **kwargs):
        return super().load(path, reload, **kwargs)

    # region
    # def _train(self, loader):
    #     self.model.train()

    #     bar = progress_bar(loader)

    #     for i, batch in enumerate(bar, 1):
    #         chars, *feats, tags = batch
    #         char_mask = chars.ne(self.args.pad_index)
    #         mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
    #         s_tag = self.model(chars, feats)
    #         loss = self.model.loss(s_tag, tags, mask)
    #         loss = loss / self.args.update_steps
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
    #         if i % self.args.update_steps == 0:
    #             self.optimizer.step()
    #             self.scheduler.step()
    #             self.optimizer.zero_grad()

    #         bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
    #     logger.info(f"{bar.postfix}")
    # endregion
    # one iteration/epoch
    def _train(self, loader, loader2=None):
        self.model.train()

        use_loader2 = False
        if loader2 is not None:
            use_loader2 = True
            if 1 == self.args.update_steps:
                self.args.update_steps = 2

        # n_batch
        n_batch1 = len(loader)  # bucket number? shuffle every iteration, or not?
        n_batch = n_batch1
        n_loader2 = len(loader2) if use_loader2 else 0
        loader = iter(loader)
        loader2 = iter(loader2) if use_loader2 else None
        if use_loader2:
            # region   扩充小的长度
            if n_batch1 < n_loader2:
                times = n_loader2 // n_batch1 + 1
                n_batch1 *= times
                loader = iter(list(loader) * times)
            else:
                times = n_batch1 // n_loader2 + 1
                n_loader2 *= times
                loader2 = iter(list(loader2) * times)
            # endregion

            n_batch = 2 * min(n_batch1, n_loader2)
            logger.info(f"\n{'train-len:':6} {n_batch1}, {'train2-len:':6} {n_loader2}")

        bar = progress_bar(range(n_batch))

        # update_steps = 2, 2次backward后，更新参数
        # use two batches, one from train, another from train2
        # for i in range(n_batch):
        for i in bar:
            # randomly select a batch
            # if use_loader2 and i % 2 == 0:
            if use_loader2 and i % 2 == 1:
                batch = next(loader2)
            else:
                batch = next(loader)
            chars, *feats, tags = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            s_tag = self.model(chars, feats)
            loss = self.model.loss(s_tag, tags, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        logger.info(f"{bar.postfix}")

        # region
        # second_bar = progress_bar(range(n_batch, max(n_batch1, n_loader2)))
        # for i in second_bar:
        #     if n_batch1 < n_loader2:
        #         batch = next(loader2)
        #     else:
        #         batch = next(loader)
        #     chars, *feats, tags = batch
        #     char_mask = chars.ne(self.args.pad_index)
        #     mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
        #     s_tag = self.model(chars, feats)
        #     loss = self.model.loss(s_tag, tags, mask)
        #     loss = loss / self.args.update_steps
        #     loss.backward()
        #     nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        #     self.optimizer.step()
        #     self.scheduler.step()
        #     self.optimizer.zero_grad()
        #     second_bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        # logger.info(f"{second_bar.postfix}")
        # endregion
        

    def _train_rate(self, loader, loader2=None):
        self.model.train()

        use_loader2 = False
        if loader2 is not None:
            use_loader2 = True

        # n_batch
        n_batch, n_loader2 = len(loader), len(loader2) if use_loader2 else 0
        loader, loader2 = iter(loader), iter(loader2) if use_loader2 else None
        if use_loader2:
            self.args.update_steps = 0

        bar = progress_bar(range(n_batch))

        # update_steps = 2, 2次backward后，更新参数
        # use two batches, one from train, another from train2
        # for i in range(n_batch):
        for i in bar:
            # randomly select a batch
            # if use_loader2 and i % 2 == 0:
            if use_loader2 and i % 2 == 1:
                batch = next(loader2)
                # time = 10
                time = 1
            else:
                batch = next(loader)
                time = 1
            chars, *feats, tags = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            s_tag = self.model(chars, feats)
            loss = self.model.loss(s_tag, tags, mask) * time
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f}")
        logger.info(f"{bar.postfix}")


    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        #pdb.set_trace()
        total_loss, metric = 0, SegF1Metric()

        for batch in loader:
            chars, *feats, tags = batch
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)  # decide padding position
            s_tag = self.model(chars, feats)
            loss = self.model.loss(s_tag, tags, mask)
            preds = self.model.decode(s_tag, mask, None)

            # for t, m, pred in zip(tags, mask, preds):
            #     tag_flag_matrix = t[m].tolist()
            #     tag_ids = [x.index(True) for x in tag_flag_matrix]
            #     print('gold', t := self.TAG.vocab[tag_ids], len(t))
            #     print('pred:', pred, len(pred))
            #     print('tag', self.TAG.vocab[pred])
            
            preds = [CWSCoNLL.recover_words(self.TAG.vocab[pred], True) for pred in preds]
            golds = []
            # now: [[F ... T ...], ...] 
            for t, m in zip(tags, mask):
                tag_flag_matrix = t[m].tolist() 
                tag_ids = [x.index(True) for x in tag_flag_matrix] 
                golds.append(CWSCoNLL.recover_words(self.TAG.vocab[tag_ids], False))
            # orig: [tagid, tagidx, ...]; padding positions are discarded
            #golds = [CWSCoNLL.recover_words(self.TAG.vocab[t[m].tolist()], False) for t, m in zip(tags, mask)]
            total_loss += loss.item()
            metric(preds, golds)
        total_loss /= len(loader)

        return total_loss, metric


    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()
        bar = progress_bar(loader)
        results = {'segs': [], 'tags': []}
        if self.args.compute_marg_probs:
            results['probs'] = []
        for i, batch in enumerate(bar):
            if self.args.constrained:
                chars, *feats, gold_tags = batch
            else:
                chars, *feats = batch
                gold_tags = None
            char_mask = chars.ne(self.args.pad_index)
            mask = char_mask if len(chars.shape) < 3 else char_mask.any(-1)
            # print("tags", [(i, each) for i, each in enumerate(self.TAG.vocab)])
            # print("chars shape:", chars.shape)
            s_tag = self.model(chars, feats)
            preds = self.model.decode(s_tag, mask, gold_tags)
            # print("preds shape:", preds.shape)

            if self.args.compute_marg_probs:
                marg_probs = self.model.get_marg_probs(s_tag, mask)
                results['probs'].extend(marg_probs)
            preds = [self.TAG.vocab[pred] for pred in preds]
            results['tags'].extend(preds)
            results['segs'].extend([CWSCoNLL.recover_words(pred, True) for pred in preds])
            bar.set_postfix_str(f"batch-id: {i}")
        logger.info(f"{bar.postfix}")
        return results

    @classmethod
    def compose_bigram_label(cls, label_sequence):
        #results = copy.deepcopy(label_sequence)
        results = [label_sequence[i]+label_sequence[i+1] for i in range(len(label_sequence)-1)]
        results.append(label_sequence[-1]+'s')
        return results
    
    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.args.embed_dict).to(args.device)
            return parser

        logger.info("Building the fields")
        BICHAR, TRICHAR, BERT = None, None, None
        if args.encoder != 'lstm':
            from transformers import AutoTokenizer
            t = AutoTokenizer.from_pretrained(args.bert)
            CHAR = NGramField('chars',
                              pad=t.pad_token,
                              unk=t.unk_token)
            CHAR.vocab = t.get_vocab()
        else:
            CHAR = NGramField('chars', pad=PAD, unk=UNK, ngram=1)
            if 'bichar' in args.feat:
                BICHAR = NGramField('bichars', pad=PAD, unk=UNK, ngram=2)
            if 'trichar' in args.feat:
                TRICHAR = NGramField('trichars', pad=PAD, unk=UNK, ngram=3)
            if 'bert' in args.feat:
                from transformers import AutoTokenizer
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = NGramField('bert',
                                  pad=t.pad_token,
                                  unk=t.unk_token)
                BERT.vocab = t.get_vocab()

        #TAG = Field('tags', fn=cls.compose_bigram_label)
        TAG = BiLabelField('tags')
        TAG.specials = ['bb', 'bm', 'be', 'bs', 
                        'mb', 'mm', 'me', 'ms', 
                        'eb', 'em', 'ee', 'es', 
                        'sb', 'sm', 'se', 'ss' ]
        TAG.specials_w_boundary = ['eb', 'es', 'sb', 'ss']
        transform = CWSCoNLL(FORM=(CHAR, BICHAR, TRICHAR, BERT), TAG=TAG)
        train = Dataset(transform, args.train)
        if args.encoder == 'lstm':
            dev = Dataset(transform, args.dev)
            test = Dataset(transform, args.test)
            datasets = [train, dev, test]
            char_embed = Embedding.load(args.char_embed, args.unk) if args.char_embed else None
            CHAR.build(train, args.min_freq, embed=char_embed, norm=lambda x: x / torch.std(x))
            if BICHAR is not None:
                bichar_embed = Embedding.load(args.bichar_embed, args.unk) if args.bichar_embed else None
                BICHAR.build(train, min_freq=3, embed=bichar_embed, norm=lambda x: x / torch.std(x))
            if TRICHAR is not None:
                trichar_embed = Embedding.load(args.trichar_embed, args.unk) if args.trichar_embed else None
                TRICHAR.build(train, min_freq=5, embed=trichar_embed, norm=lambda x: x / torch.std(x))
        TAG.build(train)
        assert len(TAG.vocab.itos) == 16
        print(TAG.vocab.itos)

        args.update({
            'n_chars': len(CHAR.vocab) if args.encoder != 'lstm' else CHAR.vocab.n_init,
            'n_tags': len(TAG.vocab),
            'n_bichars': len(BICHAR.vocab) if BICHAR is not None else None,
            'n_trichars': len(TRICHAR.vocab) if TRICHAR is not None else None,
            'bert_pad_index': BERT.pad_index if BERT is not None else None,
            'pad_index': CHAR.pad_index,
            'unk_index': CHAR.unk_index,
        })
        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained({'pretrained_char_embed': CHAR.embed if hasattr(CHAR, 'embed') else None,
                                                   'pretrained_bichar_embed': BICHAR.embed if hasattr(BICHAR, 'embed') else None,
                                                   'pretrained_trichar_embed': TRICHAR.embed if hasattr(TRICHAR, 'embed') else None}).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class CRFWordSegmenter(TagWordSegmenter):

    NAME = 'crf-word-segmenter'
    MODEL = CRFWordSegmentationModel
