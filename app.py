# coding=UTF-8

import pickle
import sqlite3

import tensorflow as tf
import numpy as np
from flask import Flask,render_template,request
from wtforms import Form,TextAreaField,validators
import re
import os
from SeqUnit import *

# from chapter09.flask_web.vectorizer import vect

app = Flask(__name__)


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

if FLAGS.load != "0":
    save_dir = 'results/res/' + FLAGS.load + '/'
    save_file_dir = save_dir + 'files/'
    
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
    
    
def convert_tab(sess, ib_in):
    ib = ib_in.decode('string_escape')
    print('string input is:',ib)
    box_word, box_label, box_pos = [], [], []
    item = ib.split('\t')
    print('item is:', item)
    box_single_word, box_single_label, box_single_pos = [], [], []
    for it in item:
        print('..............input is...........', it)
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
    v = vocab
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
    print('.......................enc_rpos........................', batch_data['enc_rpos'])
        
    model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size, emb_size=FLAGS.emb_size,
                        field_size=FLAGS.field_size, pos_size=FLAGS.pos_size, field_vocab=FLAGS.field_vocab,
                        source_vocab=FLAGS.source_vocab, position_vocab=FLAGS.position_vocab,
                        target_vocab=FLAGS.target_vocab, scope_name="seq2seq", name="seq2seq",
                        field_concat=FLAGS.field, position_concat=FLAGS.position,
                        fgate_enc=FLAGS.fgate_encoder, dual_att=FLAGS.dual_attention, decoder_add_pos=FLAGS.decoder_pos,
                        encoder_add_pos=FLAGS.encoder_pos, learning_rate=FLAGS.learning_rate)
    sess.run(tf.global_variables_initializer())
    if FLAGS.load != '0':
            model.load(save_dir)
    print('.......................input data........................', batch_data)
    
    predictions, atts = model.generate(batch_data, sess)
    print('.......................predict........................',predictions)
    atts = np.squeeze(atts)
    idx = 0

    
    for summary in np.array(predictions):
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
    pred_list = ' '.join(real_sum)
    return pred_list  



@app.route("/")
def index():
    form = ReviewForm(request.form)
    return render_template("index.html",form=form)


@app.route("/main",methods=["POST"])
def main():
    form = ReviewForm(request.form)
    if request.method == "POST" and form.validate():
        review_text = request.form["review"]
#         Y,lable_Y,proba = classify_review([review_text])
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            pred_list  = convert_tab(sess, review_text)
        proba = pred_list
        return render_template("reviewform.html",review=review_text,Y=1,label=None,probability=proba)
    return render_template("index.html",form=form)


@app.route("/tanks",methods=["POST"])
def tanks():
    btn_value = request.form["feedback_btn"]
    review = request.form["review"]
    label_temp = int(request.form["Y"])
    if btn_value == "Correct":
        label = label_temp
    else:
        label = 1 - label_temp

    save_review(review,label)
    return render_template("tanks.html")

class ReviewForm(Form):
    review = TextAreaField("",[validators.DataRequired()])

if __name__ == "__main__":
    app.run(host='0.0.0.0')
