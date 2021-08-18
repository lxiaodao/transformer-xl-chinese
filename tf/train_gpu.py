from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from gpu_utils import assign_to_gpu, average_grads_and_vars

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from visualize_attention import visualize_attention_per_head, visualize_prob, visualize_attention_per_layer
from postprocess import top_one_result, gen_on_keyword, gen_diversity

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


def get_model_fn(n_token, cutoffs):
    def model_fn(inp, tgt, mems, is_training):
        #转换成一维的嵌套数组
        inp = tf.transpose(inp, [1, 0])
        tgt = tf.transpose(tgt, [1, 0])
        print("---the inp is----")
        print(inp)
        if FLAGS.init == "uniform":
            initializer = tf.initializers.random_uniform(
                minval=-FLAGS.init_range,
                maxval=FLAGS.init_range,
                seed=None)
        elif FLAGS.init == "normal":
            #生成一组符合标准正态分布的 tensor 对象
            #stddev：正态分布的标准差
            initializer = tf.initializers.random_normal(
                stddev=FLAGS.init_std,
                seed=None)
            proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std,
                seed=None)

        tie_projs = [False for _ in range(len(cutoffs) + 1)]
        if FLAGS.proj_share_all_but_first:
            for i in range(1, len(tie_projs)):
                tie_projs[i] = True
        # todo  明确loss含义
        # 初始化算法模型
        loss, new_mems = model.transformer(
            dec_inp=inp,
            target=tgt,
            mems=mems,
            n_token=n_token,
            n_layer=FLAGS.n_layer,
            d_model=FLAGS.d_model,
            d_embed=FLAGS.d_embed,
            n_head=FLAGS.n_head,
            d_head=FLAGS.d_head,
            d_inner=FLAGS.d_inner,
            dropout=FLAGS.dropout,
            dropatt=FLAGS.dropatt,
            initializer=initializer,
            proj_initializer=proj_initializer,
            is_training=is_training,
            mem_len=FLAGS.mem_len,
            cutoffs=cutoffs,
            div_val=FLAGS.div_val,
            tie_projs=tie_projs,
            input_perms=None,
            target_perms=None,
            head_target=None,
            same_length=FLAGS.same_length,
            clamp_len=FLAGS.clamp_len,
            use_tpu=False,
            untie_r=FLAGS.untie_r,
            proj_same_dim=FLAGS.proj_same_dim)

        # number of parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        # format_str = '{{:<{0}s}}\t{{}}'.format(
        #     max([len(v.name) for v in tf.trainable_variables()]))
        # for v in tf.trainable_variables():
        #   tf.logging.info(format_str.format(v.name, v.get_shape()))

        if is_training:
            all_vars = tf.trainable_variables()
            grads = tf.gradients(loss, all_vars)
            grads_and_vars = list(zip(grads, all_vars))

            return loss, new_mems, grads_and_vars
        else:
            return loss, new_mems

    return model_fn


def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems):
    model_fn = get_model_fn(
        n_token=n_token,
        cutoffs=cutoffs)
    # tgt 实际是label数组
    model_ret = model_fn(
        inp=inp,
        tgt=tgt,
        mems=mems,
        is_training=is_training)

    return model_ret

#n_token 词汇表数量
#cutoffs 空值
#ps_device  /gpu:0
def train(n_token, cutoffs, ps_device):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

    # Get input function and model function
    train_input_fn, train_record_info = data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split="train",
        per_host_bsz=FLAGS.train_batch_size,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_core_per_host,
        num_hosts=1,
        use_tpu=False)

    tf.logging.info("num of batches {}".format(train_record_info["num_batch"]))

    # Create computational graph
    # 训练用的数据集也创建好
    # train_batch_size 4
    train_set = train_input_fn({
        "batch_size": FLAGS.train_batch_size,
        "data_dir": FLAGS.data_dir})

    input_feed, label_feed = train_set.make_one_shot_iterator().get_next()
    #num_core_per_host 4
    # 0维度分割成4个数组
    inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
    labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

    print_op = tf.print(inputs)

    # 4 // 4
    per_core_bsz = FLAGS.train_batch_size // FLAGS.num_core_per_host

    tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []
    
    # num_core_per_host次循环取值,比如4次
    for i in range(FLAGS.num_core_per_host):
        reuse = True if i > 0 else None
        #todo  review here
        with tf.device(assign_to_gpu(i, ps_device)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            #
           #Placeholder的中文意思就是占位符，用于在会话运行时动态提供输入数据。
           #Placeholder相当于定义了一个位置，在这个位置上的数据在程序运行时再指定。
           #在以后的编程中我们可能会遇到这样的情况：在训练神经网络时需要每次提供一个批量的训练样本，
           #如果每次迭代选取的数据要通过常量表示，那么TensorFlow 的计算图会非常大。
           #因为每增加一个常量，TensorFlow 都会在计算图中增加一个结点，所以说拥有几百万次迭代的神经网络会拥有极其庞大的计算图，
           #而占位符却可以解决这一点，它只会拥有占位符这一个结点，Placeholder机制的出现就是为了解决这个问题，
           #我们在编程的时候只需要把数据通过placeholder传入tensorflow计算图即可
           #16层
            mems_i = [tf.placeholder(tf.float32,
                                     [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]
            # 
            # mems_i 16层的训练图，16个初始值，每个初始值预定[100, 1, 410]，100个维度，1x410的数组
            #
            loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
                n_token=n_token,
                cutoffs=cutoffs,
                is_training=True,
                inp=inputs[i],
                tgt=labels[i],
                mems=mems_i)

            tower_mems.append(mems_i)
            tower_losses.append(loss_i)
            tower_new_mems.append(new_mems_i)
            tower_grads_and_vars.append(grads_and_vars_i)

    # average losses and gradients across towers
    # 跨塔的平均损失和梯度，what is tower?
    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
        grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
    else:
        loss = tower_losses[0]
        grads_and_vars = tower_grads_and_vars[0]
    grads, all_vars = zip(*grads_and_vars)

    # clip gradient
    # 梯度分割，截取比率0.25
    # Gradient Clipping的引入是为了处理gradient explosion或者gradients vanishing的问题。
    #当在一次迭代中权重的更新过于迅猛的话，很容易导致loss divergence。Gradient Clipping的直观作用就是让权重的更新限制在一个合适的范围。

    # global_norm 是所有梯度的平方和 ?
    clipped, gnorm = tf.clip_by_global_norm(grads, FLAGS.clip)
    grads_and_vars = list(zip(clipped, all_vars))

    # configure the optimizer
    # 这个函数主要用于返回或者创建（如果有必要的话）一个全局步数的tensor变量。参数只有一个，就是图，如果没有指定那么就是默认的图。
    global_step = tf.train.get_or_create_global_step()

    # warmup stage: increase the learning rate linearly
    # 预热阶段：线性增加学习率 0
    if FLAGS.warmup_steps > 0:
        warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                    * FLAGS.learning_rate
    else:
        warmup_lr = 0.0

    # decay stage: decay the learning rate using the cosine schedule
    # 衰减阶段：使用余弦表衰减学习率
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step - FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps - FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)

    # choose warmup or decay
    # 选择预热或者衰减，这里应该是衰减
    learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                             warmup_lr, decay_lr)

    # get the train op
    # Adam这个名字来源于Adaptive Moment Estimation，自适应矩估计。
    # RMSprop，Adadelta，Adam是非常相似的优化算法，Adam的bias-correction帮助其在最后优化期间梯度变稀疏的情况下略微战胜了RMSprop。
    # 概率论中矩的含义是：如果一个随机变量 X 服从某个分布，X 的一阶矩是 E(X)，也就是样本平均值，X 的二阶矩就是 E(X^2)，也就是样本平方的平均值。
    # Adam 算法根据损失函数对每个参数的梯度的一阶矩估计和二阶矩估计动态调整针对于每个参数的学习速率。
    #Adam 也是基于梯度下降的方法，但是每次迭代参数的学习步长都有一个确定的范围，不会因为很大的梯度导致很大的学习步长，参数的值比较稳定。
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    # Training loop
    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_core_per_host)
    ]

    saver = tf.train.Saver()

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss', loss)
    # tf.summary.scalar('pplx', math.exp(curr_loss))
    merged = tf.summary.merge_all()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        # todo 放在 此处是因为不用重复的创建trainer目录能显示变量
        train_writer = tf.summary.FileWriter(os.path.join(FLAGS.model_dir, "log"), sess.graph)

        if FLAGS.warm_start_path is not None:
            tf.logging.info("warm start from {}".format(FLAGS.warm_start_path))
            saver.restore(sess, FLAGS.warm_start_path)

        fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op]

        total_loss, prev_step = 0., -1
        while True:
            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            #old
            # fetched = sess.run(fetches, feed_dict=feed_dict)

            # with tf.control_dependencies([print_op]):
            summary, fetched = sess.run([merged, fetches], feed_dict=feed_dict)

            loss_np, tower_mems_np, curr_step = fetched[:3]
            total_loss += loss_np

            if curr_step > 0 and curr_step % FLAGS.iterations == 0:
                curr_loss = total_loss / (curr_step - prev_step)
                tf.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} "
                                "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(curr_step, fetched[-3], fetched[-2], curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
                total_loss, prev_step = 0., curr_step
                train_writer.add_summary(summary, curr_step)

            if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
                save_path = os.path.join(FLAGS.model_dir, "model-{}.ckpt".format(curr_step))
                saver.save(sess, save_path)
                tf.logging.info("Model saved in path: {}".format(save_path))

            if curr_step == FLAGS.train_steps:
                train_writer.close()
                break


def evaluate(n_token, cutoffs, ps_device):
    # Get input function and model function
    eval_input_fn, eval_record_info = data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split=FLAGS.eval_split,
        per_host_bsz=FLAGS.eval_batch_size,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_core_per_host,
        num_hosts=1,
        use_tpu=False)

    num_batch = eval_record_info["num_batch"]
    if FLAGS.max_eval_batch > 0:
        num_batch = FLAGS.max_eval_batch
    tf.logging.info("num of batches {}".format(num_batch))

    # Create computational graph
    eval_set = eval_input_fn({
        "batch_size": FLAGS.eval_batch_size,
        "data_dir": FLAGS.data_dir})

    input_feed, label_feed = eval_set.make_one_shot_iterator().get_next()

    inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
    labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

    per_core_bsz = FLAGS.eval_batch_size // FLAGS.num_core_per_host
    tower_mems, tower_losses, tower_new_mems = [], [], []

    for i in range(FLAGS.num_core_per_host):
        with tf.device(assign_to_gpu(i, ps_device)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            mems_i = [tf.placeholder(tf.float32,
                                     [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]

            loss_i, new_mems_i = single_core_graph(
                n_token=n_token,
                cutoffs=cutoffs,
                is_training=False,
                inp=inputs[i],
                tgt=labels[i],
                mems=mems_i)

            tower_mems.append(mems_i)
            tower_losses.append(loss_i)
            tower_new_mems.append(new_mems_i)

    # sum losses across towers
    if len(tower_losses) > 1:
        loss = tf.add_n(tower_losses) / len(tower_losses)
    else:
        loss = tower_losses[0]

    # Evaluation loop
    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_core_per_host)
    ]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.eval_ckpt_path is None:
            eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
        else:
            eval_ckpt_path = FLAGS.eval_ckpt_path
        tf.logging.info("Evaluate {}".format(eval_ckpt_path))
        saver.restore(sess, eval_ckpt_path)

        fetches = [loss, tower_new_mems, tf.size(label_feed)]

        format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(
            len(str(num_batch)))

        total_loss, total_cnt = 0, 0
        for step in range(num_batch):
            if step % (num_batch // 10) == 0:
                tf.logging.info(format_str.format(step, num_batch))

            feed_dict = {}
            for i in range(FLAGS.num_core_per_host):
                for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                    feed_dict[m] = m_np

            fetched = sess.run(fetches, feed_dict=feed_dict)

            loss_np, tower_mems_np, cnt_np = fetched[:3]
            total_loss += loss_np * cnt_np
            total_cnt += cnt_np

        avg_loss = total_loss / total_cnt
        tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
            avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))


def main(unused_argv):
    del unused_argv  # Unused

    tf.logging.set_verbosity(tf.logging.INFO)

    # Get corpus info
    corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
    n_token = corpus_info["vocab_size"]
    #[1:-1] 从下标为1（从0开始）的元素开始顺序读取
    #[1::-1] 从下标为1（从0开始）的元素开始翻转读取
    cutoffs = corpus_info["cutoffs"][1:-1]
    tf.logging.info("n_token {}".format(n_token))

    if FLAGS.do_train:
        train(n_token, cutoffs, "/gpu:0")
    if FLAGS.do_eval:
        evaluate(n_token, cutoffs, "/gpu:0")
    if FLAGS.do_inference:
        inference(n_token, cutoffs, "/gpu:0")


# new added by pgao
def inference(n_token, cutoffs, ps_device):
    dataset_name = "doupo"
    tmp_Vocab = Vocab()
    tmp_Vocab.count_file("../data/{}/train.txt".format(dataset_name), add_eos=False)
    tmp_Vocab.build_vocab()

    n_token = len(tmp_Vocab)
    # print(tmp_Vocab.idx2sym)

    test_list = tf.placeholder(tf.int64, shape=[1, None])
    dataset = tf.data.Dataset.from_tensors(test_list)
    # dataset = dataset.batch(1, drop_remainder=True)

    iterator = dataset.make_initializable_iterator()
    input_feed = iterator.get_next()

    inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
    # inputs = input_feed

    per_core_bsz = 1
    tower_mems, tower_losses, tower_new_mems = [], [], []
    tower_output = []
    tower_mems_id = []
    tower_new_mems_id = []
    tower_attn_prob = []

    for i in range(FLAGS.num_core_per_host):
        with tf.device(assign_to_gpu(i, ps_device)), \
             tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            mems_i = [tf.placeholder(tf.float32,
                                     [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                      for _ in range(FLAGS.n_layer)]

            mems_i_id = [tf.placeholder(tf.int64,
                                     [FLAGS.mem_len, per_core_bsz])
                      for _ in range(FLAGS.n_layer)]

            new_mems_i, output_i, new_mems_i_id, attn_prob_i = single_core_graph_for_inference(
                n_token=n_token,
                cutoffs=cutoffs,
                is_training=False,
                inp=inputs[i],
                mems=mems_i,
                mems_id=mems_i_id)

            tower_mems.append(mems_i)
            tower_new_mems.append(new_mems_i)
            tower_output.append(output_i)
            tower_mems_id.append(mems_i_id)
            tower_new_mems_id.append(new_mems_i_id)
            tower_attn_prob.append(attn_prob_i)

    # Evaluation loop
    tower_mems_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_core_per_host)
    ]

    tower_mems_id_np = [
        [np.zeros([FLAGS.mem_len, per_core_bsz], dtype=np.float32)
         for layer in range(FLAGS.n_layer)]
        for core in range(FLAGS.num_core_per_host)
    ]

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())

        if FLAGS.eval_ckpt_path is None:
            eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
        else:
            eval_ckpt_path = FLAGS.eval_ckpt_path

        saver.restore(sess, eval_ckpt_path)

        # attention_score = tf.get_variable('transformer/layer_2/rel_attn/transpose_1:0')

        fetches = [tower_new_mems,
                   tower_output,
                   tower_new_mems_id,
                   tower_attn_prob,
                   'transformer/adaptive_embed/lookup_table:0']

        while True:
            input_text = input("seed text >>> ")
            while not input_text:
                print('Prompt should not be empty!')
                input_text = input("Model prompt >>> ")
            encoded_input = tmp_Vocab.encode_sents(input_text, ordered=True)

            with open('{}.txt'.format(dataset_name), 'a') as f:
                f.write('-' * 100+'\n')
                f.write('input:\n')
                f.write(input_text+'\n')

            output_len = 600
            progress = ProgressBar()
            for step in progress(range(output_len)):
                time.sleep(0.01)
                feed_dict = {}
                for i in range(FLAGS.num_core_per_host):
                    for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
                        feed_dict[m] = m_np

                    for id, id_np in zip(tower_mems_id[i], tower_mems_id_np[i]):
                        feed_dict[id] = id_np

                sess.run(iterator.initializer, feed_dict={test_list: [encoded_input]})
                fetched = sess.run(fetches, feed_dict=feed_dict)

                tower_mems_np, output = fetched[:2]

                tower_mems_id_np = fetched[2]

                attn_prob = fetched[3]
                lookup_table = fetched[4]
                # print(attention_score)
                # print(np.array(lookup_table).shape)
                # print(np.array(tower_mems_id_np).shape)

                tmp_list = output[0][-1][0]
                tmp_list = tmp_list.tolist()

                # 下面是对结果的6种处理方式，若需要就保留，然后注释掉其他几种
                # todo 取top1
                index = top_one_result(tmp_list)
                # todo diversity
                # index = gen_diversity(tmp_list)
                # todo base on keyword
                # index = gen_on_keyword(tmp_Vocab, '喜', tmp_list, lookup_table)

                # # todo 可视化候选词
                # visualize_prob(tmp_Vocab, tmp_list,
                # '../exp_result/{}/candidates'.format(dataset_name+'mem_len500'), len(input_text))

                # # # todo 可视化attention per layer
                # visualize_attention_per_layer(tmp_Vocab, tower_mems_id_np, attn_prob, index,
                #                               '../exp_result/{}/attention_per_layer'.format(dataset_name+'mem_len500'),
                #                               len(input_text))

                # # # todo 可视化attention per head
                # visualize_attention_per_head(tmp_Vocab, tower_mems_id_np, attn_prob, index,
                #                              '../exp_result/{}/attention_per_head'.format(dataset_name+'_repeat'),
                #                              len(input_text))

                input_text += tmp_Vocab.get_sym(index) if tmp_Vocab.get_sym(index) != '<eos>' else '\n'
                encoded_input = [index]

            print(input_text)

            with open('{}.txt'.format(dataset_name), 'a') as f:
                f.write('output:\n')
                f.write(input_text+'\n')
                f.write('-'*100+'\n')


def single_core_graph_for_inference(n_token, cutoffs, is_training, inp,  mems, mems_id):
    model_fn = get_model_fn_for_inference(
        n_token=n_token,
        cutoffs=cutoffs)

    model_ret = model_fn(
        inp=inp,
        mems=mems,
        mems_id=mems_id,
        is_training=is_training)

    return model_ret


def get_model_fn_for_inference(n_token, cutoffs):
    def model_fn(inp, mems, mems_id, is_training):
        inp = tf.transpose(inp, [1, 0])

        if FLAGS.init == "uniform":
            initializer = tf.initializers.random_uniform(
                minval=-FLAGS.init_range,
                maxval=FLAGS.init_range,
                seed=None)
        elif FLAGS.init == "normal":
            initializer = tf.initializers.random_normal(
                stddev=FLAGS.init_std,
                seed=None)
            proj_initializer = tf.initializers.random_normal(
                stddev=FLAGS.proj_init_std,
                seed=None)

        tie_projs = [False for _ in range(len(cutoffs) + 1)]
        if FLAGS.proj_share_all_but_first:
            for i in range(1, len(tie_projs)):
                tie_projs[i] = True
        new_mems, output, new_mems_id, attn_prob = model.transformer_inference(
            dec_inp=inp,
            mems=mems,
            mems_id=mems_id,
            n_token=n_token,
            n_layer=FLAGS.n_layer,
            d_model=FLAGS.d_model,
            d_embed=FLAGS.d_embed,
            n_head=FLAGS.n_head,
            d_head=FLAGS.d_head,
            d_inner=FLAGS.d_inner,
            dropout=FLAGS.dropout,
            dropatt=FLAGS.dropatt,
            initializer=initializer,
            proj_initializer=proj_initializer,
            is_training=is_training,
            mem_len=FLAGS.mem_len,
            cutoffs=cutoffs,
            div_val=FLAGS.div_val,
            tie_projs=tie_projs,
            input_perms=None,
            target_perms=None,
            head_target=None,
            same_length=FLAGS.same_length,
            clamp_len=FLAGS.clamp_len,
            use_tpu=False,
            untie_r=FLAGS.untie_r,
            proj_same_dim=FLAGS.proj_same_dim)

        # number of parameters
        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        return new_mems, output, new_mems_id, attn_prob

    return model_fn


if __name__ == "__main__":
    tf.app.run()
