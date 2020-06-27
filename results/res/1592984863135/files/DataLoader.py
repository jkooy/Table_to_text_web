#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:43
# @Author  : Tianyu Liu

import tensorflow as tf
import time
import numpy as np
import re, time, os


class DataLoader(object):
    def __init__(self, data_dir, limits):
        self.train_data_path = [data_dir + '/train/train.summary.id', data_dir + '/train/train.box.val.id',
                                data_dir + '/train/train.box.lab.id', data_dir + '/train/train.box.pos',
                                data_dir + '/train/train.box.rpos']
        self.test_data_path = [data_dir + '/test/test.summary.id', data_dir + '/test/test.box.val.id',
                               data_dir + '/test/test.box.lab.id', data_dir + '/test/test.box.pos',
                               data_dir + '/test/test.box.rpos']
        self.dev_data_path = [data_dir + '/valid/valid.summary.id', data_dir + '/valid/valid.box.val.id',
                              data_dir + '/valid/valid.box.lab.id', data_dir + '/valid/valid.box.pos',
                              data_dir + '/valid/valid.box.rpos']
        self.limits = limits
        self.man_text_len = 100
        start_time = time.time()

        print('Reading datasets ...')
#         self.train_set = self.load_data(self.train_data_path)
        self.test_set = self.load_data(self.test_data_path)
        # self.small_test_set = self.load_data(self.small_test_data_path)
#         self.dev_set = self.load_data(self.dev_data_path)
        print ('Reading datasets comsumes %.3f seconds' % (time.time() - start_time))

    def load_data(self, path):
        summary_path, text_path, field_path, pos_path, rpos_path = path
        summaries = open(summary_path, 'r').read().strip().split('\n')
        texts = open(text_path, 'r').read().strip().split('\n')
        fields = open(field_path, 'r').read().strip().split('\n')
        poses = open(pos_path, 'r').read().strip().split('\n')
        rposes = open(rpos_path, 'r').read().strip().split('\n')
        if self.limits > 0:
            summaries = summaries[:self.limits]
            texts = texts[:self.limits]
            fields = fields[:self.limits]
            poses = poses[:self.limits]
            rposes = rposes[:self.limits]
        print summaries[0].strip().split(' ')
        summaries = [list(map(int, summary.strip().split(' '))) for summary in summaries]
        
        print('................texts.................', texts[0])
        texts = [list(map(int, text.strip().split(' '))) for text in texts]
        fields = [list(map(int, field.strip().split(' '))) for field in fields]
        poses = [list(map(int, pos.strip().split(' '))) for pos in poses]
        rposes = [list(map(int, rpos.strip().split(' '))) for rpos in rposes]
        return summaries, texts, fields, poses, rposes
    
    def single_test(self):
        class Vocab(object):
            def __init__(self):
                vocab = dict()
                vocab['PAD'] = 0
                vocab['START_TOKEN'] = 1
                vocab['END_TOKEN'] = 2
                vocab['UNK_TOKEN'] = 3
                cnt = 4
                with open("original_data/word_vocab.txt", "r") as v:
                    for line in v:
                        word = line.strip().split()[0]
                        vocab[word] = cnt
                        cnt += 1
                self._word2id = vocab
                self._id2word = {value: key for key, value in vocab.items()}

                key_map = dict()
                key_map['PAD'] = 0
                key_map['START_TOKEN'] = 1
                key_map['END_TOKEN'] = 2
                key_map['UNK_TOKEN'] = 3
                cnt = 4
                with open("original_data/field_vocab.txt", "r") as v:
                    for line in v:
                        key = line.strip().split()[0]
                        key_map[key] = cnt
                        cnt += 1
                self._key2id = key_map
                self._id2key = {value: key for key, value in key_map.items()}

            def word2id(self, word):
                ans = self._word2id[word] if word in self._word2id else 3
                return ans

            def id2word(self, id):
                ans = self._id2word[int(id)]
                return ans

            def key2id(self, key):
                ans = self._key2id[key] if key in self._key2id else 3
                return ans

            def id2key(self, id):
                ans = self._id2key[int(id)]
                return ans


        fboxes = "original_data/test.box"

        mixb_word, mixb_label, mixb_pos = [], [], []

        box = open(fboxes, "r").read().strip().split('\n')
        box_word, box_label, box_pos = [], [], []

        print(box[0])
        ib = box[0]

        item = ib.split('\t')
        box_single_word, box_single_label, box_single_pos = [], [], []
        for it in item:
            if len(it.split(':')) > 2:
                continue
            # print it
            prefix, word = it.split(':')
            if '<none>' in word or word.strip()=='' or prefix.strip()=='':
                continue
            new_label = re.sub("_[1-9]\d*$", "", prefix)
            if new_label.strip() == "":
                continue
            box_single_word.append(word)
            box_single_label.append(new_label)
            if re.search("_[1-9]\d*$", prefix):
                field_id = int(prefix.split('_')[-1])
                box_single_pos.append(field_id if field_id<=30 else 30)
            else:
                box_single_pos.append(1)
        box_word.append(box_single_word)
        box_label.append(box_single_label)
        box_pos.append(box_single_pos)


        ######################## reverse box #############################
        box = box_pos
        tmp_pos = []
        single_pos = []
        reverse_pos = []
        for pos in box:
            tmp_pos = []
            single_pos = []
            for p in pos:
                if int(p) == 1 and len(tmp_pos) != 0:
                    single_pos.extend(tmp_pos[::-1])
                    tmp_pos = []
                tmp_pos.append(p)
            single_pos.extend(tmp_pos[::-1])
            reverse_pos = single_pos

    
    
        vocab = Vocab()

        texts = (" ".join([str(vocab.word2id(word)) for word in box_word[0]]) + '\n')
        text = list(map(int,texts.strip().split(' ')))
        print(text)

        fields = (" ".join([str(vocab.key2id(word)) for word in box_label[0]]) + '\n')
        field = list(map(int,fields.strip().split(' ')))
        print(field)

        pos = box_pos[0]
        print(pos)   

        rpos = reverse_pos
        print(rpos)
               
        text_len = len(text)
        pos_len = len(pos)
        rpos_len = len(rpos)
                
        batch_data = {'enc_in':[], 'enc_fd':[], 'enc_pos':[], 'enc_rpos':[], 'enc_len':[],
                          'dec_in':[], 'dec_len':[], 'dec_out':[]}
        
        batch_data['enc_in'].append(text)
        batch_data['enc_len'].append(text_len)
        batch_data['enc_fd'].append(field)
        batch_data['enc_pos'].append(pos)
        batch_data['enc_rpos'].append(rpos)
        
        yield batch_data
        
    def batch_iter(self, data, batch_size, shuffle):
        summaries, texts, fields, poses, rposes = data
        
        data_size = len(summaries)
        num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
            else int(data_size / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            summaries = np.array(summaries)[shuffle_indices]
            texts = np.array(texts)[shuffle_indices]
            fields = np.array(fields)[shuffle_indices]
            poses = np.array(poses)[shuffle_indices]
            rposes = np.array(rposes)[shuffle_indices]

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            max_summary_len = max([len(sample) for sample in summaries[start_index:end_index]])
            max_text_len = max([len(sample) for sample in texts[start_index:end_index]])
            batch_data = {'enc_in':[], 'enc_fd':[], 'enc_pos':[], 'enc_rpos':[], 'enc_len':[],
                          'dec_in':[], 'dec_len':[], 'dec_out':[]}

            for summary, text, field, pos, rpos in zip(summaries[start_index:end_index], texts[start_index:end_index],
                                            fields[start_index:end_index], poses[start_index:end_index],
                                            rposes[start_index:end_index]):
                summary_len = len(summary)
                text_len = len(text)
                pos_len = len(pos)
                rpos_len = len(rpos)
                assert text_len == len(field)
                assert pos_len == len(field)
                assert rpos_len == pos_len
                gold = summary + [2] + [0] * (max_summary_len - summary_len)
                summary = summary + [0] * (max_summary_len - summary_len)
                text = text + [0] * (max_text_len - text_len)
                field = field + [0] * (max_text_len - text_len)
                pos = pos + [0] * (max_text_len - text_len)
                rpos = rpos + [0] * (max_text_len - text_len)
                
                if max_text_len > self.man_text_len:
                    text = text[:self.man_text_len]
                    field = field[:self.man_text_len]
                    pos = pos[:self.man_text_len]
                    rpos = rpos[:self.man_text_len]
                    text_len = min(text_len, self.man_text_len)
                
                batch_data['enc_in'].append(text)
                batch_data['enc_len'].append(text_len)
                batch_data['enc_fd'].append(field)
                batch_data['enc_pos'].append(pos)
                batch_data['enc_rpos'].append(rpos)
                batch_data['dec_in'].append(summary)
                batch_data['dec_len'].append(summary_len)
                batch_data['dec_out'].append(gold)
            yield batch_data