# Copyright 2017 The AI-labs.org Authors. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import re
import random
import sys
import time
import codecs
import numpy as np
import data_utils
import seq2seq_model
import tensorflow as tf
import string
import json
import asyncio
import aiohttp
import copy

from aiohttp import web
from urllib.parse import parse_qsl, urlparse
from six.moves import xrange  # pylint: disable=redefined-builtin
from spacy.en import English

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("inp_vocab_size", 100000, "Input vocabulary size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 100000, "Output vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("rest", False,
                            "Set to True for web server start.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
tf.app.flags.DEFINE_boolean("rewrite", False,
                            "Set to True for decoding from stdin and write to stdout.")
tf.app.flags.DEFINE_boolean("use_fp16", False,
                            "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

sess = None
model = None
templates = None
noun = None
adj = None
verb = None
inp_vocab = None
rev_out_vocab = None

config = tf.ConfigProto(allow_soft_placement=True)

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(6, 6), (7, 7), (8, 8), (10, 10)]

nlp = English()


def read_data(source_path, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
    data_set = [[] for _ in _buckets]
    with codecs.open(source_path, mode="r") as source_file:
        with codecs.open(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    model = seq2seq_model.Seq2SeqModel(
        FLAGS.inp_vocab_size,
        FLAGS.out_vocab_size,
        _buckets,
        FLAGS.size,
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        forward_only=forward_only,
        dtype=dtype)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train():
    """Train model."""
    # Prepare data.
    print("Preparing data in %s" % FLAGS.data_dir)
    inp_train, out_train, inp_dev, out_dev, _, _ = data_utils.prepare_data(
        FLAGS.data_dir, FLAGS.inp_vocab_size, FLAGS.out_vocab_size)

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:1'):
            # Create model.
            print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
            model = create_model(sess, False)

            # Read data into buckets and compute their sizes.
            print("Reading development and training data (limit: %d)."
                  % FLAGS.max_train_data_size)
            dev_set = read_data(inp_dev, out_dev)
            train_set = read_data(inp_train, out_train, FLAGS.max_train_data_size)
            train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
            train_total_size = float(sum(train_bucket_sizes))

            # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
            # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
            # the size if i-th training bucket, as used later.
            train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                                   for i in xrange(len(train_bucket_sizes))]

            # This is the training loop.
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            while True:
                # Choose a bucket according to data distribution. We pick a random number
                # in [0, 1] and use the corresponding interval in train_buckets_scale.
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in xrange(len(train_buckets_scale))
                                 if train_buckets_scale[i] > random_number_01])

                # Get a batch and make a step.
                start_time = time.time()
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    train_set, bucket_id)
                _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                             target_weights, bucket_id, False)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1

                # Once in a while, we save checkpoint, print statistics, and run evals.
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    # Print statistics for the previous epoch.
                    perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                    print("global step %d learning rate %.4f step-time %.2f perplexity "
                          "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                    step_time, perplexity))
                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)
                    # Save checkpoint and zero timer and loss.
                    checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0.0
                    # Run evals on development set and print their perplexity.
                    for bucket_id in xrange(len(_buckets)):
                        if len(dev_set[bucket_id]) == 0:
                            print("  eval: empty bucket %d" % (bucket_id))
                            continue
                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                            dev_set, bucket_id)
                        _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                     target_weights, bucket_id, True)
                        eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                            "inf")
                        print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                    sys.stdout.flush()


def decode():
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:1'):
            # Create model and load parameters.
            model = create_model(sess, True)
            model.batch_size = 1  # We decode one sentence at a time.

            # Load vocabularies.
            inp_vocab_path = os.path.join(FLAGS.data_dir,
                                          "vocab%d.inp" % FLAGS.inp_vocab_size)
            out_vocab_path = os.path.join(FLAGS.data_dir,
                                          "vocab%d.out" % FLAGS.out_vocab_size)
            inp_vocab, _ = data_utils.initialize_vocabulary(inp_vocab_path)
            _, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)

            # Decode from standard input.
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            while sentence:
                # Get token-ids for the input sentence.
                token_ids = data_utils.sentence_to_token_ids(sentence, inp_vocab)
                print('token_ids:', token_ids)
                # Which bucket does it belong to?
                bucket_id = [b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)]
                if bucket_id != []:
                    bucket_id = min(bucket_id)
                elif len(token_ids) >= _buckets[3][0]:
                    bucket_id = 3
                elif len(token_ids) <= _buckets[0][0]:
                    bucket_id = 0
                # Get a 1-element batch to feed the sentence to the model.
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(token_ids, [])]}, bucket_id)
                # Get output logits for the sentence.
                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                # This is a greedy decoder - outputs are just argmaxes of output_logits.
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                # If there is an EOS symbol in outputs, cut them at that point.
                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                print('outputs', outputs)
                # Print out French sentence corresponding to outputs.
                print(" ".join([tf.compat.as_str(rev_out_vocab[output]) for output in outputs]))
                print("> ", end="")
                sys.stdout.flush()
                sentence = sys.stdin.readline()


def make_response(user_words):
    # print(user_words)
    print('__________New request was received__________')
    template = random.choice(templates)
    print('Selected template:', template)
    if ('noun' in user_words) and user_words['noun']:
        i = 0
        for idx, item in enumerate(template):
            if item == '_NN':
                try:
                    template[idx] = user_words['noun'][i]
                    i += 1
                except IndexError:
                    continue
    if ('adj' in user_words) and user_words['adj']:
        j = 0
        for idx, item in enumerate(template):
            if item == '_JJ':
                try:
                    template[idx] = user_words['adj'][j]
                    j += 1
                except IndexError:
                    continue
    if ('verb' in user_words) and user_words['verb']:
        j = 0
        for idx, item in enumerate(template):
            if item == '_VB':
                try:
                    template[idx] = user_words['verb'][j]
                    j += 1
                except IndexError:
                    continue
    print('Template with user modification:', template)
    noun_list = copy.copy(noun)
    adj_list = copy.copy(adj)
    verb_list = copy.copy(verb)
    for idx, item in enumerate(template):
        if item == '_NN' and random.randrange(1, 10) <= 5:
            noun_rand = random.choice(noun_list)
            noun_list.remove(noun_rand)
            template[idx] = noun_rand
        if item == '_JJ' and random.randrange(1, 10) <= 5:
            adj_rand = random.choice(adj_list)
            adj_list.remove(adj_rand)
            template[idx] = adj_rand
        if item == '_VB' and random.randrange(1, 10) <= 5:
            verb_rand = random.choice(verb_list)
            verb_list.remove(verb_rand)
            template[idx] = verb_rand
    sentence = ' '.join(template)
    print('Template with user modification and random words injection:', template)
    print('Input sequence:', sentence)

    token_ids = data_utils.sentence_to_token_ids(sentence, inp_vocab)
    # print('token_ids:', token_ids)
    # Which bucket does it belong to?
    bucket_id = [b for b in xrange(len(_buckets))
                 if _buckets[b][0] > len(token_ids)]
    if bucket_id != []:
        bucket_id = min(bucket_id)
    elif len(token_ids) >= _buckets[3][0]:
        bucket_id = 3
    elif len(token_ids) <= _buckets[0][0]:
        bucket_id = 0
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        {bucket_id: [(token_ids, [])]}, bucket_id)
    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
    # print('outputs', outputs)

    outputs2 = []
    [outputs2.append(tf.compat.as_str(rev_out_vocab[output])) for output in outputs]

    output_str = ""
    for idx, val in enumerate(outputs2):
        if val == "_UNK":
            if idx == 0:
                output_str += 'Somebody'
            else:
                output_str += ' ' + 'somebody'
        if val != "_UNK":
            if idx == 0:
                output_str += val
            else:
                output_str += ' ' + val

    # output_str = " ".join([tf.compat.as_str(rev_out_vocab[output]) for output in outputs])
    # output_str = output_str.replace("_UNK", "Somebody")
    print('Output sequence:', output_str)
    return output_str

async def old_handler(request):
    post_params = await request.json()
    response = web.Response(content_type='text/html')
    response.text = make_response(post_params) + '\n'
    return response

async def single_handler(request):
    post_params = await request.json()
    response = web.Response(content_type='application/json')
    response.text = json.dumps({'sentence': make_response(post_params)})
    return response

async def multiple_handler(request):
    post_params = await request.json()
    response = web.Response(content_type='application/json')
    sent_list = []

    if 'n' in post_params:
        try:
            n = int(post_params['n'])
        except ValueError:
            n = 1
    else:
        n = 1

    for idx in range(n):
        sent_list.append(make_response(post_params))
    response.text = json.dumps({'sentences': sent_list})
    return response

async def init(loop):
    handler = app.make_handler()
    srv = await loop.create_server(handler, '0.0.0.0', 8005)
    print('serving on', srv.sockets[0].getsockname())
    return srv

loop = asyncio.get_event_loop()
app = web.Application()
app.router.add_post('/', old_handler)
app.router.add_post('/single', single_handler)
app.router.add_post('/multiple', multiple_handler)


def rest():
    templates_file = codecs.open('templates.json', encoding='utf-8', mode='r')
    globals()['templates'] = json.load(templates_file)

    noun_file = codecs.open('noun.json', encoding='utf-8', mode='r')
    globals()['noun'] = json.load(noun_file)

    adj_file = codecs.open('adj.json', encoding='utf-8', mode='r')
    globals()['adj'] = json.load(adj_file)

    verb_file = codecs.open('verb.json', encoding='utf-8', mode='r')
    globals()['verb'] = json.load(verb_file)

    # config1 = tf.ConfigProto(device_count={'GPU': 0})

    globals()['sess'] = tf.Session(config=config)
    globals()['model'] = create_model(sess, True)
    globals()['model'].batch_size = 1  # We decode one sentence at a time.
    inp_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.inp" % FLAGS.inp_vocab_size)
    out_vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.out" % FLAGS.out_vocab_size)
    inp_vocab, _ = data_utils.initialize_vocabulary(inp_vocab_path)
    _, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)
    globals()['inp_vocab'] = inp_vocab
    globals()['rev_out_vocab'] = rev_out_vocab

    loop.run_until_complete(init(loop))

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass


def extract(line):
    doc = nlp(line)
    result = []
    for token in doc:
        result.append(token.pos_)
    result_str = ' '.join(result)
    return result_str


def prepare(line, template_similar):
    line = str(line).strip()
    doc = nlp(line)
    result = []
    ent_num = 1
    names_dict = {}
    template = random.choice(template_similar).split()
    template_ = []
    for templ_one in template:
        template_.append('_' + templ_one)
    template_str = ' '.join(template_)
    for token in doc:
        if token.ent_type_ != "":
            tag_ = '_NAME' + str(ent_num)
            result.append(tag_)
            ent_num += 1
            names_dict[tag_] = token.orth_
            template_str = template_str.replace('_PROPN', tag_, 1)
            print(template_str)
        else:
            if token.pos_ is 'NOUN':
                template_str = template_str.replace('_NOUN', token.lemma_, 1)
            if token.pos_ is 'ADV':
                template_str = template_str.replace('_ADV', token.lemma_, 1)
            if token.pos_ is 'VERB':
                template_str = template_str.replace('_VERB', token.lemma_, 1)
            if token.pos_ is 'PRON':
                template_str = template_str.replace('_PRON', token.lemma_, 1)
    # print(template_str)
    # print(names_dict)
    return template_str, names_dict


def prepare2(line, template_similar):
    line = str(line).strip()
    doc = nlp(line)
    result = []
    ent_num = 1
    names_dict = {}
    template = template_similar.split()
    template_ = []
    for templ_one in template:
        template_.append('_' + templ_one)
    template_str = ' '.join(template_)
    for token in doc:
        if token.ent_type_ != "":
            tag_ = '_NAME' + str(ent_num)
            result.append(tag_)
            ent_num += 1
            names_dict[tag_] = token.orth_
            template_str = template_str.replace('_PROPN', tag_, 1)
            print(template_str)
        else:
            if token.pos_ is 'NOUN':
                template_str = template_str.replace('_NOUN', token.lemma_, 1)
            if token.pos_ is 'ADV':
                template_str = template_str.replace('_ADV', token.lemma_, 1)
            if token.pos_ is 'VERB':
                template_str = template_str.replace('_VERB', token.lemma_, 1)
            if token.pos_ is 'PRON':
                template_str = template_str.replace('_PRON', token.lemma_, 1)
    # print(template_str)
    # print(names_dict)
    return template_str, names_dict


def rewrite():
    distances = codecs.open('distances_filtered.json', encoding='utf-8', mode='r')
    templates_dict = json.load(distances)
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:1'):
            # Create model and load parameters.
            model = create_model(sess, True)
            model.batch_size = 1  # We decode one sentence at a time.

            # Load vocabularies.
            inp_vocab_path = os.path.join(FLAGS.data_dir,
                                          "vocab%d.inp" % FLAGS.inp_vocab_size)
            out_vocab_path = os.path.join(FLAGS.data_dir,
                                          "vocab%d.out" % FLAGS.out_vocab_size)
            inp_vocab, _ = data_utils.initialize_vocabulary(inp_vocab_path)
            _, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)

            # Decode from standard input.
            fp1 = codecs.open('output.txt', mode="w")
            fp2 = codecs.open('input.txt', mode="w")
            input_file = sys.stdin
            text = []
            for line in input_file:
                text.append(line.strip())
            text = ' '.join(text)
            text = text.replace("\t", " ")
            text = re.sub(' +', ' ', text)
            doc = nlp(text)
            list_sents = list(doc.sents)
            # print(list_sents)
            for sentence in list_sents:
                # print(sentence)
                template = extract(str(sentence).strip())
                if template in templates_dict.keys():
                    fp2.write('+++ ' + str(sentence) + '\n')
                    sentence, names_dict = prepare(sentence, templates_dict[template])
                else:
                    fp2.write('--- ' + str(sentence) + '\n')
                    sentence, names_dict = prepare2(sentence, template)
                # Get token-ids for the input sentence.
                token_ids = data_utils.sentence_to_token_ids(sentence, inp_vocab)
                # print('token_ids:', token_ids)
                # Which bucket does it belong to?
                bucket_id = [b for b in xrange(len(_buckets))
                             if _buckets[b][0] > len(token_ids)]
                if bucket_id != []:
                    bucket_id = min(bucket_id)
                elif len(token_ids) >= _buckets[3][0]:
                    bucket_id = 3
                elif len(token_ids) <= _buckets[0][0]:
                    bucket_id = 0
                # Get a 1-element batch to feed the sentence to the model.
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(token_ids, [])]}, bucket_id)
                # Get output logits for the sentence.
                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                                 target_weights, bucket_id, True)
                # This is a greedy decoder - outputs are just argmaxes of output_logits.
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                # If there is an EOS symbol in outputs, cut them at that point.
                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                # Print out sentence corresponding to outputs.
                out = [tf.compat.as_str(rev_out_vocab[output]) for output in outputs]
                # print(out)
                names_list = list(names_dict.keys())
                print(names_list)
                out_str = ''
                for word in out:
                    if word in names_list:
                        out_str = out_str + " " + names_dict[word]
                        print(out_str)
                    else:
                        if word not in string.punctuation:
                            out_str = out_str + " " + word
                        else:
                            out_str = out_str + word
                out_str = out_str.strip()
                fp1.write(out_str + '\n')
            print('The result was saved in the files input.txt and output.txt')


def self_test():
    """Test the translation model."""
    with tf.Session(config=config) as sess:
        with tf.device('/gpu:1'):
            print("Self-test for neural translation model.")
            # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
            model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                               5.0, 32, 0.3, 0.99, num_samples=8)
            sess.run(tf.initialize_all_variables())

            # Fake data set for both the (3, 3) and (6, 6) bucket.
            data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                        [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
            for _ in xrange(5):  # Train the fake model for 5 steps.
                bucket_id = random.choice([0, 1])
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    data_set, bucket_id)
                model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                           bucket_id, False)


def main(_):
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    elif FLAGS.rewrite:
        rewrite()
    elif FLAGS.rest:
        rest()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
