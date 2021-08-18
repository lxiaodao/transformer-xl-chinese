import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import math
from vocabulary import Vocab
from absl import flags
from progressbar import ProgressBar
import time
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import model
import data_utils
import random
import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

#handler = logging.StreamHandler(sys.stdout)
handler = logging.FileHandler('tfrecord-demo.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


# GPU config
flags.DEFINE_integer("num_hosts", default=1,
                     help="Number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=4,
                     help="Number of cores per host")

# Experiment (data/checkpoint/directory) config
# 2021-7-8
flags.DEFINE_string("data_dir", default="../data/doupo/tfrecords",
                    help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir", default="../data/doupo/tfrecords/",
                    help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="../data/doupo/corpus-info.json",
                    help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default="EXP-doupo4-1_head-1e4",
                    help="Estimator model_dir.")
flags.DEFINE_bool("do_train", default=True,
                  help="Whether to run training.")
flags.DEFINE_bool("do_eval", default=False,
                  help="Whether to run eval on the dev set.")
flags.DEFINE_bool("do_inference", default=False,
                  help="Whether to run eval on the dev set.")

flags.DEFINE_string("eval_ckpt_path", None,
                    help="Checkpoint path for do_test evaluation."
                         "If set, model_dir will be ignored."
                         "If unset, will use the latest ckpt in model_dir.")
flags.DEFINE_string("warm_start_path", None,
                    help="Checkpoint path for warm start."
                         "If set, will clear Adam states."
                         "Note that the new model_dir should be different"
                         " from warm_start_path.")

# Optimization config
flags.DEFINE_float("learning_rate", default=0.00010,
                   help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
                   help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.004,
                   help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
                     help="Number of steps for linear lr warmup.")

# Training config
flags.DEFINE_integer("train_batch_size", default=4,
                     help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size", default=60,
                     help="Size of valid batch.")
flags.DEFINE_integer("train_steps", default=1000000,
                     help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=200,
                     help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=4000,
                     help="number of steps for model checkpointing.")

# Evaluation config
flags.DEFINE_bool("do_test", default=False,
                  help="Run on the test set.")
flags.DEFINE_integer("max_eval_batch", default=-1,
                     help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_bool("do_eval_only", default=False,
                  help="Run evaluation only.")
flags.DEFINE_integer("start_eval_steps", default=10000,
                     help="Which checkpoint to start with in `do_eval_only` mode.")
flags.DEFINE_string("eval_split", "valid",
                    help="Which data split to evaluate.")

# Model config
flags.DEFINE_integer("tgt_len", default=100,
                     help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=100,
                     help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
                  help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")

flags.DEFINE_integer("n_layer", default=16,
                     help="Number of layers.")
flags.DEFINE_integer("d_model", default=410,
                     help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=410,
                     help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
                     help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=41,
                     help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=2100,
                     help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.0,
                   help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=True,
                  help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True,
                  help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val", default=1,
                     help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=True,
                  help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
                  help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
                   help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

FLAGS = flags.FLAGS


def tfdemo(unused_argv):
    del unused_argv  # Unused
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    # Get input function and model function
    train_input_fn, train_record_info = data_utils.get_input_fn(
        record_info_dir="../data/doupo/tfrecords/",
        split="train",
        per_host_bsz=4,
        tgt_len=100,
        num_core_per_host=4,
        num_hosts=1,
        use_tpu=False)

    tf.logging.info("num of batches {}".format(train_record_info["num_batch"]))

    # Create computational graph
    # 训练用的数据集也创建好
    # train_batch_size 4
    train_set = train_input_fn({
        "batch_size": 4,
        "data_dir": "../data/doupo/tfrecords"})

    input_feed, label_feed = train_set.make_one_shot_iterator().get_next()
    #num_core_per_host 4
    # 0维度分割成4个数组
    inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
    labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

    #print_op = tf.print(inputs)
    inp = tf.transpose(inputs[0], [1, 0])
    tgt = tf.transpose(labels[0], [1, 0])
   
    #mems_i 16个初始值，每个初始值预定[100, 1, 410]，100个维度，1x410的数组
    mems_i = [tf.placeholder(tf.float32,
                                     [FLAGS.mem_len, FLAGS.train_batch_size, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]
    #模拟single_core_graph方法的执行逻辑
    #
    initializer = tf.initializers.random_normal(
                stddev=FLAGS.init_std,
                seed=None)
    proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std,
                seed=None)
      
    r_w_bias = tf.get_variable('r_w_bias', [FLAGS.n_layer, FLAGS.n_head, FLAGS.d_head],
                                       initializer=initializer)
     #获取维度数值
     # 100 100 200
    qlen = tf.shape(inp)[0]
    mlen = tf.shape(mems_i[0])[0] if mems_i is not None else 0
    klen = mlen + qlen
   
    
   # 得到不同字的 embedding
   # 返回放大的y值 和 lookup_table参数，词库数量，嵌入数量度
    embeddings, shared_params = model.mask_adaptive_embedding_lookup(
            x=inp,
            n_token=2596,
            d_embed=FLAGS.d_embed, #410
            d_proj=FLAGS.d_model, #410
            cutoffs=[],
            initializer=initializer,
            proj_initializer=proj_initializer,
            div_val=FLAGS.div_val,
            perms=None,
            proj_same_dim=FLAGS.proj_same_dim)
    # todo 这个mask 有什么用?????????
    attn_mask = model._create_mask(qlen, mlen, same_length=False)
   
     
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.logging.info('---three params are  : {}-{}-{}'.format(qlen.eval(), mlen.eval(), klen.eval()))
        tf.logging.info('---lookup_table embeddings is ---')
        tf.logging.info(sess.run(embeddings))
        tf.logging.info(embeddings.eval())
     
        tf.logging.info('---mask print---')
        tf.logging.info(sess.run(attn_mask))
        tf.logging.info(attn_mask.eval())
       


#




if __name__ == "__main__":
    tf.app.run(tfdemo)

 
