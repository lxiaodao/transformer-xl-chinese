import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import math
from vocabulary import Vocab
from absl import flags
from progressbar import ProgressBar
import time
#import tensorflow as tf
import tensorflow.compat.v1 as tf

#tf.disable_eager_execution()
tf.enable_eager_execution()


import model
import data_utils
import random
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.INFO)

#handler = logging.StreamHandler(sys.stdout)
handler = logging.FileHandler('from_tensor_slices.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)

def parserOne(record):
    record_spec = {
                    "inputs": tf.VarLenFeature(tf.int64),
                    "labels": tf.VarLenFeature(tf.int64),
                }
    # retrieve serialized example
    example = tf.parse_single_example(
                serialized=record,
                features=record_spec)
    # cast int64 into int32
    # cast sparse to dense
    for key in list(example.keys()):
                val = example[key]
                if tf.keras.backend.is_sparse(val):
                    val = tf.sparse.to_dense(val)
                if val.dtype == tf.int64:
                    val = tf.to_int32(val)
                example[key] = val
    return example["inputs"], example["labels"]
    
dataset = tf.data.Dataset.from_tensor_slices(["../data/doupo/tfrecords/train.bsz-4.tlen-100.tfrecords"])
dataset = tf.data.TFRecordDataset(dataset)
tf.logging.info("---this is a tfrecord dataset---")
tf.logging.info(dataset)
dataset = dataset.map(parserOne).cache().repeat()
dataset = dataset.batch(4, drop_remainder=True)
dataset = dataset.prefetch(4 * 4)

input_feed, label_feed = dataset.make_one_shot_iterator().get_next()
    #num_core_per_host 4
    # 0维度分割成4个数组
inputs = tf.split(input_feed, 4, 0)
labels = tf.split(label_feed, 4, 0)

inp = tf.transpose(inputs[0], [1, 0])
tgt = tf.transpose(labels[0], [1, 0])
        
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #tf.logging.info(input_feed)
    #tf.logging.info("------label_feed------")
    #tf.logging.info(labels)
    #
    tf.logging.info("------After transpose input is------")
    tf.logging.info(inp)
    tf.logging.info("------After transpose tgt(label) is------")
    tf.logging.info(tgt)
 

 
