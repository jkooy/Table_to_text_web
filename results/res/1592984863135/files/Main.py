#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:44
# @Author  : Tianyu Liu

import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import time
from SeqUnit import *
from DataLoader import DataLoader
import numpy as np
from PythonROUGE import PythonROUGE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from preprocess import *
from util import * 
import logging


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
format=log_format, datefmt='%m/%d %I:%M:%S %p')


tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 400, "Size of embedding.")
tf.app.flags.DEFINE_integer("field_size", 50, "Size of embedding.")
tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 50, "Number of training epoch.")
tf.app.flags.DEFINE_integer("source_vocab", 20003,'vocabulary size')
tf.app.flags.DEFINE_integer("field_vocab", 1480,'vocabulary size')
tf.app.flags.DEFINE_integer("position_vocab", 31,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 20003,'vocabulary size')
tf.app.flags.DEFINE_integer("report", 1000,'report valid results after some steps')
tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')

tf.app.flags.DEFINE_string("mode",'test','train')
# tf.app.flags.DEFINE_string('train')
# tf.app.flags.DEFINE_string("load",'0','load directory') # BBBBBESTOFAll
tf.app.flags.DEFINE_string("load",'1592984863135','load directory') # BBBBBESTOFAll
tf.app.flags.DEFINE_string("dir",'processed_data','data set directory')
tf.app.flags.DEFINE_integer("limits", 0,'max data set size')


tf.app.flags.DEFINE_boolean("dual_attention", True,'dual attention layer or normal attention')
tf.app.flags.DEFINE_boolean("fgate_encoder", True,'add field gate in encoder lstm')

tf.app.flags.DEFINE_boolean("field", False,'concat field information to word embedding')
tf.app.flags.DEFINE_boolean("position", False,'concat position information to word embedding')
tf.app.flags.DEFINE_boolean("encoder_pos", True,'position information in field-gated encoder')
tf.app.flags.DEFINE_boolean("decoder_pos", True,'position information in dual attention decoder')


FLAGS = tf.app.flags.FLAGS
last_best = 0.0

gold_path_test = 'processed_data/test/test_split_for_rouge/gold_summary_'
gold_path_valid = 'processed_data/valid/valid_split_for_rouge/gold_summary_'

# test phase
if FLAGS.load != "0":
    save_dir = 'results/res/' + FLAGS.load + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = 'results/evaluation/' + FLAGS.load + '/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'
# train phase
else:
    prefix = str(int(time.time() * 1000))
    save_dir = 'results/res/' + prefix + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = 'results/evaluation/' + prefix + '/'
    os.mkdir(save_dir)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'

log_file = save_dir + 'log.txt'


def train(sess, dataloader, model):
    write_log("#######################################################")
    for flag in FLAGS.__flags:
        write_log(flag + " = " + str(FLAGS.__flags[flag]))
    write_log("#######################################################")
    trainset = dataloader.train_set
    k = 0
    loss, start_time = 0.0, time.time()
    for _ in range(FLAGS.epoch):
        for x in dataloader.batch_iter(trainset, FLAGS.batch_size, True):
            loss += model(x, sess)
            k += 1
            progress_bar(k%FLAGS.report, FLAGS.report)
            if (k % FLAGS.report == 0):
                cost_time = time.time() - start_time
                write_log("%d : loss = %.3f, time = %.3f " % (k // FLAGS.report, loss, cost_time))
                loss, start_time = 0.0, time.time()
                if k // FLAGS.report >= 1: 
                    ksave_dir = save_model(model, save_dir, k // FLAGS.report)
                    write_log(evaluate(sess, dataloader, model, ksave_dir, 'valid'))
                    


def test(sess, dataloader, model):
    logging.info("test = %s", model)
#     evaluate(sess, dataloader, model, save_dir, 'test')
    output(sess, dataloader, model, save_dir, 'test')

def save_model(model, save_dir, cnt):
    new_dir = save_dir + 'loads' + '/' 
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    nnew_dir = new_dir + str(cnt) + '/'
    if not os.path.exists(nnew_dir):
        os.mkdir(nnew_dir)
    model.save(nnew_dir)
    return nnew_dir


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
    
def output(sess, dataloader, model, ksave_dir, mode='valid'):
    if mode == 'valid':
        # texts_path = "original_data/valid.summary"
        texts_path = "processed_data/valid/valid.box.val"
        gold_path = gold_path_valid
        evalset = dataloader.dev_set
    else:
        # texts_path = "original_data/test.summary"
        texts_path = "processed_data/test/test.box.val"
        gold_path = gold_path_test
        evalset = dataloader.test_set
    
    # for copy words from the infoboxes
    v = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []
    
    print('..............Begin test iteration...........')
    print('Total bacth number is:', len(list(dataloader.batch_iter(evalset, FLAGS.batch_size, False))))
    k = 3
    
    fboxes = "original_data/test.box"
    box = open(fboxes, "r").read().strip().split('\n')

    mixb_word, mixb_label, mixb_pos = [], [], []

    box_word, box_label, box_pos = [], [], []
    ib = box[k]

    
#     ib = 'name_1:xuehau he\tname_2:ssss\tposition_1:second\tposition_2:hhhh'
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

    textss = (" ".join([str(vocab.word2id(word)) for word in box_word[0]]) + '\n')
    text = list(map(int,textss.strip().split(' ')))

    fields = (" ".join([str(vocab.key2id(word)) for word in box_label[0]]) + '\n')
    field = list(map(int,fields.strip().split(' ')))

    pos = box_pos[0]

    rpos = reverse_pos

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
        
    print('.......................input data........................', batch_data)
    predictions, atts = model.generate(batch_data, sess)
    print('.......................predict........................',predictions)
    atts = np.squeeze(atts)
    idx = 0
#     print('path is:', pred_path + str(k))
#         print('x is:',x)

    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]
    print('input is', ib)
    print(texts[k])
    
    for summary in np.array(predictions):
        with open(pred_path + str(k), 'w') as sw:
            summary = list(summary)
            if 2 in summary:
                summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
            real_sum, unk_sum, mask_sum = [], [], []
            for tk, tid in enumerate(summary):
                if tid == 3:
                    sub = item[np.argmax(atts[tk,: len(item)])]
                    real_sum.append(sub)
#                     mask_sum.append("**" + str(sub) + "**")
                else:
                    real_sum.append(v.id2word(tid))
#                     mask_sum.append(v.id2word(tid))
#                 unk_sum.append(v.id2word(tid))
    print('pred set is:', ' '.join(real_sum))
    return pred_list  
    
    '''
    for x in dataloader.single_test():
        print('.......................single test........................',x)
#         def generate(self, x, sess):
#         predictions, atts = sess.run([self.g_tokens, self.atts],
#                                {self.encoder_input: x['enc_in'], self.encoder_field: x['enc_fd'], 
#                                 self.encoder_len: x['enc_len'], self.encoder_pos: x['enc_pos'],
#                                 self.encoder_rpos: x['enc_rpos']})
#         return predictions, atts
        predictions, atts = model.generate(x, sess)
        atts = np.squeeze(atts)
        idx = 0
        print('path is:', pred_path + str(k))
#         print('x is:',x)
        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum, unk_sum, mask_sum = [], [], []
                for tk, tid in enumerate(summary):
                    if tid == 3:
                        print('npmax',np.argmax(atts[tk,: len(texts[k])]))
                        print('mm',np.argmax(atts[tk,: len(texts[k])]))
                        print('texts',np.shape(texts))
                        print('texts_full',texts[k])
#                         sub = texts[k]
#                         real_sum=texts[k]
#                         mask_sum.append("**" + str(sub) + "**")
                    else:
                        real_sum.append(v.id2word(tid))
#                         mask_sum.append(v.id2word(tid))
#                     unk_sum.append(v.id2word(tid))
#                 sw.write(" ".join([str(x) for x in real_sum]) + '\n')
#                 pred_list.append([str(x) for x in real_sum])
#                 k += 1
#                 idx += 1
        break
#     write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
#     write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")


#     for tk in range(k):
#         with open(gold_path + str(tk), 'r') as g:
#             gold_list.append([g.read().strip()])

#     gold_set = [[gold_path + str(i)] for i in range(k)]
#     pred_set = [pred_path + str(i) for i in range(k)]
    
#     print('gold set is:', ','.join(str(gold_list[0])))
    print(real_sum)
    print('pred set is:', ' '.join(real_sum))
#     print('pred set is:', [i for p in real_sum for i in p])
    
    return pred_list
    '''
    

def evaluate(sess, dataloader, model, ksave_dir, mode='valid'):
    if mode == 'valid':
        # texts_path = "original_data/valid.summary"
        texts_path = "processed_data/valid/valid.box.val"
        gold_path = gold_path_valid
        evalset = dataloader.dev_set
    else:
        # texts_path = "original_data/test.summary"
        texts_path = "processed_data/test/test.box.val"
        gold_path = gold_path_test
        evalset = dataloader.test_set
    
    # for copy words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')
    print('test', texts[0])
    texts = [list(t.strip().split()) for t in texts]
#     print('test split', texts)
    v = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []
    
    print('..............Begin test iteration...........')
    print('Total bacth number is:', len(list(dataloader.batch_iter(evalset, FLAGS.batch_size, False))))
    k = 0
    
    
#     x = dataloader.batch_iter(evalset, FLAGS.batch_size, False)[0]
#     predictions, atts = model.generate(x, sess)
#     atts = np.squeeze(atts)
#     idx = 0
#     print('path is:', pred_path + str(k))
#         print('x is:',x)
#     for summary in np.array(predictions):
#         with open(pred_path + str(k), 'w') as sw:
#             summary = list(summary)
#             if 2 in summary:
#                 summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
#             real_sum, unk_sum, mask_sum = [], [], []
#             for tk, tid in enumerate(summary):
#                 if tid == 3:
#                     sub = texts[k][np.argmax(atts[tk,: len(texts[k]),idx])]
#                     real_sum.append(sub)
#                     mask_sum.append("**" + str(sub) + "**")
#                 else:
#                     real_sum.append(v.id2word(tid))
#                     mask_sum.append(v.id2word(tid))
#                 unk_sum.append(v.id2word(tid))
#             sw.write(" ".join([str(x) for x in real_sum]) + '\n')
#             pred_list.append([str(x) for x in real_sum])
#             pred_unk.append([str(x) for x in unk_sum])
#             pred_mask.append([str(x) for x in mask_sum])
#             k += 1
#             idx += 1

    for x in dataloader.batch_iter(evalset, FLAGS.batch_size, False):
        predictions, atts = model.generate(x, sess)
        atts = np.squeeze(atts)
        idx = 0
        print('path is:', pred_path + str(k))
#         print('x is:',x)
        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum, unk_sum, mask_sum = [], [], []
                for tk, tid in enumerate(summary):
                    if tid == 3:
                        sub = texts[k][np.argmax(atts[tk,: len(texts[k])])]
                        real_sum.append(sub)
                        mask_sum.append("**" + str(sub) + "**")
                    else:
                        real_sum.append(v.id2word(tid))
                        mask_sum.append(v.id2word(tid))
                    unk_sum.append(v.id2word(tid))
                sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                pred_list.append([str(x) for x in real_sum])
                pred_unk.append([str(x) for x in unk_sum])
                pred_mask.append([str(x) for x in mask_sum])
                k += 1
                idx += 1
        break
    write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
    write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")


    for tk in range(k):
        with open(gold_path + str(tk), 'r') as g:
            gold_list.append([g.read().strip().split()])

    gold_set = [[gold_path + str(i)] for i in range(k)]
    pred_set = [pred_path + str(i) for i in range(k)]
    
    print('pred set is:', ' '.join(real_sum))
#     print('gold set is:', ','.join(str(gold_list[0])))
#     print('pred set is:', ','.join(str(pred_list[0])))
#     print('pred set is:', pred_unk[0],'\n \n \n')
#     print('pred set is:', pred_mask[0])
    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_list)
    copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    (str(F_measure), str(recall), str(precision), str(bleu))
    # print copy_result

    for tk in range(k):
        with open(pred_path + str(tk), 'w') as sw:
            sw.write(" ".join(pred_unk[tk]) + '\n')

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_unk)
    nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
    (str(F_measure), str(recall), str(precision), str(bleu))
    # print nocopy_result
    result = copy_result + nocopy_result 
    # print result
    if mode == 'valid':
        print result

    return result



def write_log(s):
    print s
    with open(log_file, 'a') as f:
        f.write(s+'\n')


def main():
    config = tf.ConfigProto(allow_soft_placement=True, device_count={'cpu':0})
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        copy_file(save_file_dir)
        dataloader = DataLoader(FLAGS.dir, FLAGS.limits)
#         print('dataloader',dataloader.test_set[0])
        model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size, emb_size=FLAGS.emb_size,
                        field_size=FLAGS.field_size, pos_size=FLAGS.pos_size, field_vocab=FLAGS.field_vocab,
                        source_vocab=FLAGS.source_vocab, position_vocab=FLAGS.position_vocab,
                        target_vocab=FLAGS.target_vocab, scope_name="seq2seq", name="seq2seq",
                        field_concat=FLAGS.field, position_concat=FLAGS.position,
                        fgate_enc=FLAGS.fgate_encoder, dual_att=FLAGS.dual_attention, decoder_add_pos=FLAGS.decoder_pos,
                        encoder_add_pos=FLAGS.encoder_pos, learning_rate=FLAGS.learning_rate)
        
        print '........start.............'
        sess.run(tf.global_variables_initializer())
        # copy_file(save_file_dir)
        if FLAGS.load != '0':
            model.load(save_dir)
        if FLAGS.mode == 'train':
            train(sess, dataloader, model)
        else:
            test(sess, dataloader, model)


if __name__=='__main__':
    # with tf.device('/gpu:' + FLAGS.gpu):
    main()
