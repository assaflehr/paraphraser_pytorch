import random

import csv
import time
import numpy as np
from collections import namedtuple, defaultdict


import torch
from torch.utils.data import Dataset, DataLoader
import torchtext.data as data
from seq2seq.dataset import SourceField,TargetField
from util import revers_vocab


# on sample of the dataset
OneSample = namedtuple('OneSample', ['sent_0', 'sent_1', 'sent_x', 'is_x_0', 'sent_0_target']) #'src' is out



class TimeStyleDataset(Dataset):
    def __init__(self, max_id, time_mod_3_result, label_smoothing=False):
        """
        max_id how many samples are in this dataset. Size of one epoc!
        time_mod_3_result - to make sure time is train/test/valid is different, pass 0,1,2
        label_smoothing - trick from "soumith/ganhacks", instead of 0/1 labels, pass 0-0.3, 0.7-1.2 labels
        """
        self.max_id = int(max_id)
        # TODO : add more!!# see here: https://docs.python.org/2/library/datetime.html month can be: %b,%B,%m , year: %y,%Y
        self.formats = ["%b %d %Y %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%d-%b-%Y %H.%M.%S.00",
                        "%B %d %y %H:%M:%S", "%B %d %H:%M:%S %Y", "%d/%B/%Y %H:%M:%S"]
        self.time_mod_3_result = time_mod_3_result
        self.label_smoothing = label_smoothing

    def __len__(self):
        return self.max_id

    def _tvt(self, idx):
        # for 10, if train, return 9, valid 10, test 11
        return (idx // 3) * 3 + self.time_mod_3_result  # 10//3 * 3 + x = 9+x

    def __getitem__(self, idx):
        if idx > self.max_id:
            raise IndexError(f'TimeStyleDataset {idx} is out of range {self.max_id}')

        """ x -  x[0] semantic0   , style0
                 x[1] semantic0   , style1   
                 x[2] semantic0orX, style0 (1/2 the time 0 , half X)
        """
        max_time = 2 * int(2e9)  # means 1970-2001

        random_ids = np.random.randint(low=0, high=max_time, size=2)
        idx = random_ids[0]  # other_idx is 50% same, 50% other
        other_idx = random_ids[1] if np.random.randint(0, 2) == 0 else random_ids[0]

        idx = self._tvt(idx)
        other_idx = self._tvt(other_idx)

        # two random formats
        random_fs = np.random.choice(len(self.formats), size=2, replace=False)

        sent_0 = time.strftime(self.formats[random_fs[0]], time.gmtime(idx))
        sent_1 = time.strftime(self.formats[random_fs[1]], time.gmtime(idx))
        # TODO: question: isn't it toally obvios? just do 1 less than the other
        sent_x = time.strftime(self.formats[random_fs[0]], time.gmtime(other_idx))

        y = torch.FloatTensor(np.array([idx != other_idx], np.float32))
        if self.label_smoothing:
            y[y == 1.0] = (0.7 + 0.5 * np.random.rand())
            y[y == 0.0] = (0.3 * np.random.rand())

        return OneSample(sent_0, sent_1, sent_x, y, sent_0)







def build_time_ds():
    """" Wrapper class of torchtext.data.Field that forces batch_first to be True
    and prepend <sos> and append <eos> to sequences in preprocessing step. """
    tokenize = lambda s: ['%', '&'] + ['BASEBALL']

    TEXT_TARGET = TargetField(batch_first=True, sequential=True, use_vocab=True, lower=False,
                              init_token=TargetField.SYM_SOS,
                              eos_token=TargetField.SYM_EOS, tokenize=tokenize,
                              preprocessing=lambda x: x)  # fix_length=10
    TEXT = SourceField(batch_first=True, sequential=True, use_vocab=True, lower=False,
                       tokenize=tokenize, preprocessing=lambda x: x)  # fix_length=10
    LABEL = data.Field(batch_first=True, sequential=False, use_vocab=False, tensor_type=torch.FloatTensor)

    fields = [('sent_0', TEXT), ('sent_1', TEXT), ('sent_x', TEXT), ('is_x_0', LABEL), ('sent_0_target', TEXT_TARGET)]
    ds_train = data.Dataset(TimeStyleDataset(1e3, 1, label_smoothing=True), fields)
    ds_eval = data.Dataset(TimeStyleDataset(1e3, 2), fields)

    print('printing dataset directly, before tokenizing:')
    print('sent_0', ds_train[2].sent_0)  # not processed
    print('is_x_0', ds_train[2].is_x_0)  # not processed

    print('\nbuilding vocab:')
    # TEXT.build_vocab(ds, max_size=80000)
    TEXT_TARGET.build_vocab(ds_train, max_size=80000)
    TEXT.vocab = TEXT_TARGET.vocab  # same except from the added <sos>,<eos>

    print('vocab TEXT: len', len(TEXT.vocab), 'common', TEXT.vocab.freqs.most_common()[:50])
    print('vocab TEXT_TARGET:', len(TEXT_TARGET.vocab), 'uncommon', TEXT_TARGET.vocab.freqs.most_common()[-10::])
    print('vocab ', TEXT_TARGET.SYM_SOS, TEXT_TARGET.sos_id, TEXT_TARGET.vocab.stoi[TEXT_TARGET.SYM_SOS])
    print('vocab ', TEXT_TARGET.SYM_EOS, TEXT_TARGET.eos_id, TEXT_TARGET.vocab.stoi[TEXT_TARGET.SYM_EOS])
    print('vocab ', 'out-of-vocab', TEXT_TARGET.eos_id, TEXT_TARGET.vocab.stoi['out-of-vocab'])

    device = None if torch.cuda.is_available() else -1
    # READ:  https://github.com/mjc92/TorchTextTutorial/blob/master/01.%20Getting%20started.ipynb
    sort_within_batch = True
    train_iter = iter(
        data.BucketIterator(dataset=ds_train, device=device, batch_size=32, sort_within_batch=sort_within_batch,
                            sort_key=lambda x: len(x.sent_0)))
    eval_iter = iter(
        data.BucketIterator(dataset=ds_eval, device=device, batch_size=32, sort_within_batch=sort_within_batch,
                            sort_key=lambda x: len(x.sent_0)))
    # performance note: the first next, takes 3.5s, the next are fast (10000 is 1s)


    for i in range(1):
        b = next(train_iter)
        # usage
        print('\nb.is_x_0', b.is_x_0[0], b.is_x_0.type())
        # print ('b.src is values+len tuple',b.src[0].shape,b.src[1].shape )
        print('b.sent_0_target', b.sent_0_target.shape, b.sent_0_target[0],
              revers_vocab(TEXT_TARGET.vocab, b.sent_0_target[0], ''))
        print('b_sent0', b.sent_0[0].shape, b.sent_0[1].shape, b.sent_0[0][0],
              revers_vocab(TEXT.vocab, b.sent_0[0][0], ''))
        print('b_sent1', b.sent_1[0].shape, b.sent_1[1].shape, b.sent_1[0][0],
              revers_vocab(TEXT.vocab, b.sent_1[0][0], ''))
        print('b_sentx', b.sent_x[0].shape, b.sent_x[1].shape, b.sent_x[0][0],
              revers_vocab(TEXT.vocab, b.sent_x[0][0], ''))
        print('b_y', b.is_x_0.shape, b.is_x_0)
        print(b.sent_0[1])

    return ds_train, ds_eval, train_iter, eval_iter


def test():
    dataset = TimeStyleDataset(1e3, 1)
    for i in range(3):
        sample = dataset[i];
        print('test sample', sample)
        # print ('len',len(dataset))


# test()



class SemStyleDS(Dataset):
    def __init__(self, ds, TEXT, TEXT_TARGET, LABEL, max_id=None, label_smoothing=False):
        """
        for quora dataset
        ds - source dataset from which samples are drawn using ds[i]
        label_smoothing - trick from "soumith/ganhacks", instead of 0/1 labels, pass 0
        """
        self.max_id = max_id
        # self.ds = ds
        self.label_smoothing = label_smoothing
        self.TEXT = TEXT
        self.TEXT_TARGET = TEXT_TARGET
        self.LABEL = LABEL

        self.dups = defaultdict(list)
        self.not_dups = defaultdict(list)
        self.id2q = {}
        for i, sample in enumerate(ds):
            q1, q2 = int(sample.qid1), int(sample.qid2)
            self.id2q[q1] = sample.question1
            self.id2q[q2] = sample.question2
            d = self.dups if int(sample.is_duplicate) == int(1) else self.not_dups
            d[q1].append(q2)
            d[q2].append(q1)

        # 36.9% are duplicates
        print(f'dups {len(self.dups)}, not_dups {len(self.not_dups)}')
        self.dups_key_list = list(self.dups)[:self.max_id]

    def __len__(self):
        return len(self.dups_key_list)  # len(self.ds)

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError(f'TimeStyleDataset {idx} is out of range {self.max_id}')

        """ x -  x[0] semantic0   , style0
                 x[1] semantic0   , style1   
                 x[2] semantic0orX, style0 (1/2 the time 0 , half X)
        """
        # q = random.choice(self.dups_key_list)
        q = self.dups_key_list[idx]
        sent_0 = self.id2q[q]

        sent_1 = random.choice(self.dups[q])  # choose one of the dups
        sent_1 = self.id2q[sent_1]

        y = np.random.randint(0, 2)
        if y == 0:
            sent_x = sent_0  # same style, same semantics
        else:
            if q in self.not_dups:
                sent_x = self.id2q[random.choice(self.not_dups[q])]
            else:
                sent_x = self.id2q[random.choice(self.dups_key_list)]  # just total stranger (can fail once in a lot)
        y = y + 0.0  # torch.FloatTensor([y],np.float32)
        if self.label_smoothing:
            y[y == 1.0] = (0.7 + 0.5 * np.random.rand())
            y[y == 0.0] = (0.3 * np.random.rand())

        return OneSample(self.TEXT.preprocess(sent_0), self.TEXT.preprocess(sent_1), self.TEXT.preprocess(sent_x),
                         self.LABEL.preprocess(y), self.TEXT_TARGET.preprocess(sent_0))


def build_quora_dataset():
    # Create a dataset which is only used as internal tsv reader
    SOURCE_INT = data.Field(batch_first=True, sequential=False, use_vocab=False)  # tensor_type =torch.IntTensor)
    ds = data.TabularDataset('train.csv', format='csv', skip_header=True,
                             fields=[('id', SOURCE_INT), ('qid1', SOURCE_INT), ('qid2', SOURCE_INT),
                                     ('question1', SOURCE_INT), ('question2', SOURCE_INT),
                                     ('is_duplicate', SOURCE_INT)])

    tokenize = 'revtok'  # lambda x: x.split(' ') # 'revtok' #
    TEXT_TARGET = TargetField(batch_first=True, sequential=True, use_vocab=True, lower=True,
                              init_token=TargetField.SYM_SOS,
                              eos_token=TargetField.SYM_EOS, tokenize=tokenize)  # fix_length=30)
    TEXT = SourceField(batch_first=True, sequential=True, use_vocab=True, lower=True,
                       tokenize=tokenize)  # , fix_length=20)
    LABEL = data.Field(batch_first=True, sequential=False, use_vocab=False, tensor_type=torch.FloatTensor)

    sem_style_ds = SemStyleDS(ds, TEXT, TEXT_TARGET, LABEL, max_id=1000 * 1000)
    for i in range(5):
        print(type(sem_style_ds[i].sent_0), type(sem_style_ds[i].is_x_0), sem_style_ds[i])

    ds_train = data.Dataset(sem_style_ds,
                            fields=[('sent_0', TEXT), ('sent_1', TEXT), ('sent_x', TEXT), ('is_x_0', LABEL),
                                    ('sent_0_target', TEXT_TARGET)])
    # import pdb; pdb.set_trace()
    print('printing dataset directly, before tokenizing:')
    print('sent_0', ds_train[2].sent_0)  # not processed
    print('is_x_0', ds_train[2].is_x_0)  # not processed

    print('\nbuilding vocab:')

    TEXT_TARGET.build_vocab(ds_train,
                            vectors='fasttext.simple.300d')  # , max_size=80000)#,vectors='fasttext.simple.300d')  #vectors=,'fasttext.simple.300d' not-simple 'fasttext.en.300d' ,'glove.twitter.27B.50d': '
    TEXT.vocab = TEXT_TARGET.vocab  # same except from the added <sos>,<eos>

    print('vocab TEXT: len', len(TEXT.vocab), 'common', TEXT.vocab.freqs.most_common()[:30])
    print('vocab TEXT_TARGET:', len(TEXT_TARGET.vocab), TEXT_TARGET.vocab.freqs.most_common()[:30])
    print('vocab ', TEXT_TARGET.SYM_SOS, TEXT_TARGET.sos_id, TEXT_TARGET.vocab.stoi[TEXT_TARGET.SYM_SOS])
    print('vocab ', TEXT_TARGET.SYM_EOS, TEXT_TARGET.eos_id, TEXT_TARGET.vocab.stoi[TEXT_TARGET.SYM_EOS])
    print('vocab ', 'out-of-vocab', TEXT_TARGET.eos_id, TEXT_TARGET.vocab.stoi['out-of-vocab'])

    device = None if torch.cuda.is_available() else -1
    # READ:  https://github.com/mjc92/TorchTextTutorial/blob/master/01.%20Getting%20started.ipynb
    bucket_iter_train = data.BucketIterator(dataset=ds_train, device=device, batch_size=32, sort_within_batch=False,
                                            sort_key=lambda x: len(x.sent_0))
    print('$' * 40, 'change batch_size to 32')
    training_batch_generator = iter(bucket_iter_train)

    # performance note: the first next, takes 3.5s, the next are fast (10000 is 1s)


    for i in range(5):
        b = next(training_batch_generator)
        # usage
        print('\nb.is_x_0', b.is_x_0[0], b.is_x_0.type())
        # print ('b.src is values+len tuple',b.src[0].shape,b.src[1].shape )
        print('b.sent_0_target', b.sent_0_target.shape, b.sent_0_target[0])
        print('b.sent_0_target', b.sent_0_target.shape, b.sent_0_target[0],
              revers_vocab(TEXT_TARGET.vocab, b.sent_0_target[0], ' '))
        print('b_sent0', b.sent_0[0].shape, b.sent_0[1].shape, b.sent_0[0][0],
              revers_vocab(TEXT.vocab, b.sent_0[0][0], ' '))
        print('b_sent1', b.sent_1[0].shape, b.sent_1[1].shape, b.sent_1[0][0],
              revers_vocab(TEXT.vocab, b.sent_1[0][0], ' '))
        print('b_sentx', b.sent_x[0].shape, b.sent_x[1].shape, b.sent_x[0][0],
              revers_vocab(TEXT.vocab, b.sent_x[0][0], ' '))
        print('b_y', b.is_x_0.shape, b.is_x_0[0])

    # addons

    bucket_iter_train = data.BucketIterator(dataset=ds_train, shuffle=True, device=device, batch_size=32,
                                            sort_within_batch=False, sort_key=lambda x: len(x.sent_0))
    bucket_iter_valid = bucket_iter_train  # data.BucketIterator(dataset=ds_val, shuffle=False, device=device, batch_size=32,
    #    sort_within_batch=False, #sort_key=lambda x: len(x.sent_0)
    # )
    return bucket_iter_train, bucket_iter_valid


class BibleStyleDS(Dataset):
    def __init__(self, bibles, TEXT, TEXT_TARGET, LABEL, label_smoothing=False):
        """
        ds - source dataset from which samples are drawn using ds[i]
        label_smoothing - trick from "soumith/ganhacks", instead of 0/1 labels, pass 0
        """
        self.bibles = bibles
        self.label_smoothing = label_smoothing
        self.TEXT = TEXT
        self.TEXT_TARGET = TEXT_TARGET
        self.LABEL = LABEL

    def __len__(self):
        return len(self.bibles)

    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError(f'BibleStyleDataset {idx} is out of range {len(self.bibles)}')

        """ x -  x[0] semantic0   , style0
                 x[1] semantic0   , style1   
                 x[2] semantic0orX, style0 (1/2 the time 0 , half X)
        """

        # style is chosen randomly between first and second
        style_0 = np.random.randint(0, 2)
        style_1 = (1 + style_0) % 2

        sent_0 = self.bibles[idx][style_0]
        sent_1 = self.bibles[idx][style_1]
        y = np.random.randint(0, 2)
        if y == 0:
            sent_x = sent_0  # same style, same semantics
        else:
            sent_x = random.choice(self.bibles)[style_0]

        y = y + 0.0  # torch.FloatTensor([y],np.float32)
        if self.label_smoothing:
            if y == 1.0:
                y = (0.7 + 0.5 * np.random.rand())
            else:
                y = (0.3 * np.random.rand())

        return OneSample(self.TEXT.preprocess(sent_0), self.TEXT.preprocess(sent_1), self.TEXT.preprocess(sent_x),
                         self.LABEL.preprocess(y), self.TEXT_TARGET.preprocess(sent_0))




def build_bible_datasets(verbose=False):
    """
    :return: bucket_iter_train, bucket_iter_valid
     To get an epoch-iterator , do iter= iter(bucket_iter_train). and then loop on next(iter)
     It easy to get dataset/fields from it , using bucket_iter_train.dataset.fields
    """

    def as_id_to_sentence(filename):
        d = {}
        with open(filename, 'r') as f:
            f.readline()
            for l in csv.reader(f.readlines(), quotechar='"', delimiter=',',
                                quoting=csv.QUOTE_ALL, skipinitialspace=True):
                # id,b,c,v,t
                # 1001001,1,1,1,At the first God made the heaven and the earth.
                d[l[0]] = l[4]
        return d

    bbe = as_id_to_sentence('t_bbe.csv')
    wbt = as_id_to_sentence('t_wbt.csv')
    print('num of sentences',len(bbe), len(wbt))

    # merge into a list with (s1,s2) tuple
    bibles = []
    for sent_id, sent_wbt in wbt.items():
        if sent_id in bbe:
            sent_bbe = bbe[sent_id]
            bibles.append((sent_wbt, sent_bbe))
    if verbose:
        print(len(bibles), bibles[0])



    tokenize = 'revtok'  # lambda x: x.split(' ') # 'revtok' #
    TEXT_TARGET = TargetField(batch_first=True, sequential=True, use_vocab=True, lower=True,
                              init_token=TargetField.SYM_SOS,
                              eos_token=TargetField.SYM_EOS, tokenize=tokenize)  # fix_length=30)
    TEXT = SourceField(batch_first=True, sequential=True, use_vocab=True, lower=True,
                       tokenize=tokenize)  # , fix_length=20)
    LABEL = data.Field(batch_first=True, sequential=False, use_vocab=False, tensor_type=torch.FloatTensor)

    bible_style_ds_trn = BibleStyleDS([x for (i,x) in enumerate(bibles) if i%10 != 9], TEXT, TEXT_TARGET, LABEL, label_smoothing=False)
    bible_style_ds_val = BibleStyleDS([x for (i,x) in enumerate(bibles) if i%10 == 9], TEXT, TEXT_TARGET, LABEL, label_smoothing=False)
    if verbose:
        for i in range(1):
            print("RAW SENTENCES", bible_style_ds_val[i])
        # print (type(bible_style_ds[i].sent_0),type(bible_style_ds[i].is_x_0),bible_style_ds[i])

    fields = [('sent_0', TEXT), ('sent_1', TEXT), ('sent_x', TEXT), ('is_x_0', LABEL), ('sent_0_target', TEXT_TARGET)]
    ds_train = data.Dataset(bible_style_ds_trn, fields)
    ds_val = data.Dataset(bible_style_ds_val, fields)

    # import pdb; pdb.set_trace()
    if verbose:
        print('printing dataset directly, before tokenizing:')
        print('sent_0', ds_train[2].sent_0)  # not processed
        print('is_x_0', ds_train[2].is_x_0)  # not processed

    print('\nbuilding vocab:')

    TEXT_TARGET.build_vocab(ds_train, vectors='fasttext.simple.300d',
                            min_freq=20)  # , max_size=80000)#,vectors='fasttext.simple.300d')  #vectors=,'fasttext.simple.300d' not-simple 'fasttext.en.300d' ,'glove.twitter.27B.50d': '
    TEXT.vocab = TEXT_TARGET.vocab  # same except from the added <sos>,<eos>
    print ('total',len(TEXT.vocab),'after ignoring non-frequent')
    if verbose:
        print('vocab TEXT: len', len(TEXT.vocab), 'common', TEXT.vocab.freqs.most_common()[:5])
        print('vocab TEXT: len', len(TEXT.vocab), 'uncommon', TEXT.vocab.freqs.most_common()[-5:])
        print('vocab TEXT_TARGET:', len(TEXT_TARGET.vocab), TEXT_TARGET.vocab.freqs.most_common()[:5])
        print('vocab ', TEXT_TARGET.SYM_SOS, TEXT_TARGET.sos_id, TEXT_TARGET.vocab.stoi[TEXT_TARGET.SYM_SOS])
        print('vocab ', TEXT_TARGET.SYM_EOS, TEXT_TARGET.eos_id, TEXT_TARGET.vocab.stoi[TEXT_TARGET.SYM_EOS])
        print('vocab ', 'out-of-vocab',  TEXT_TARGET.vocab.stoi['out-of-vocab'])
        print('vocab ', 'i0', [ (i,TEXT_TARGET.vocab.itos[i]) for i in range(6)] )
    device = torch.device('cuda') if torch.cuda.is_available() else -1
    # READ:  https://github.com/mjc92/TorchTextTutorial/blob/master/01.%20Getting%20started.ipynb
    print('device is cuda or -1 for cpu:', device)

    bucket_iter_train = data.BucketIterator(dataset=ds_train, shuffle=True, device=device, batch_size=32,
                                            sort_within_batch=False, sort_key=lambda x: len(x.sent_0))
    bucket_iter_valid = data.BucketIterator(dataset=ds_val, shuffle=False, device=device, batch_size=32,
                                            sort_within_batch=False, #sort_key=lambda x: len(x.sent_0)
                                        )



    if verbose: #show few samples
        for i in range(1):
            # performance note: the first next, takes 3.5s, the next are fast (10000 is 1s)
            b = next(iter(bucket_iter_train))
            # usage
            print('\nb.is_x_0', b.is_x_0[0], b.is_x_0.type())
            # print ('b.src is values+len tuple',b.src[0].shape,b.src[1].shape )
            print('b.sent_0_target', b.sent_0_target.shape)  # ,b.sent_0_target[0])
            print('b.sent_0_target', revers_vocab(TEXT_TARGET.vocab, b.sent_0_target[0], ' '))
            print('b_sent0', b.sent_0[0].shape, b.sent_0[1].shape, revers_vocab(TEXT.vocab, b.sent_0[0][0], ' '))
            print('b_sent1', b.sent_1[0].shape, b.sent_1[1].shape, revers_vocab(TEXT.vocab, b.sent_1[0][0], ' '))
            print('b_sentx', b.sent_x[0].shape, b.sent_x[1].shape, revers_vocab(TEXT.vocab, b.sent_x[0][0], ' '), b.sent_x[0][0])
            print('b_y', b.is_x_0.shape, b.is_x_0[0])


    return bucket_iter_train, bucket_iter_valid


def test() :
    #ds_train, ds_eval, train_iter, eval_iter = build_time_ds()
    bucket_iter_train, bucket_iter_valid = build_bible_datasets(verbose=True)
    print(next(iter(bucket_iter_valid)).sent_0[0])
    print(next(iter(bucket_iter_valid)).sent_0[0])


#test()

