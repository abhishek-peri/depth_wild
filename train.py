# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A training loop for the various models in this directory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import math
import os
import random
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()
from depth_from_video_in_the_wild import model
import cv2


sift = cv2.xfeatures2d.SIFT_create()
gfile = tf.gfile
MAX_TO_KEEP = 1000000  # Maximum number of checkpoints to keep.

flags.DEFINE_string('data_dir', None, 'Preprocessed data.')

flags.DEFINE_string('file_extension', 'png', 'Image data file extension.')

flags.DEFINE_float('learning_rate', 1e-4, 'Adam learning rate.')

flags.DEFINE_float('reconstr_weight', 0.85, 'Frame reconstruction loss weight.')

flags.DEFINE_float('ssim_weight', 3.0, 'SSIM loss weight.')

flags.DEFINE_float('smooth_weight', 1e-2, 'Smoothness loss weight.')

flags.DEFINE_float('depth_consistency_loss_weight', 0.01,
                   'Depth consistency loss weight')

flags.DEFINE_integer('batch_size', 1, 'The size of a sample batch')

flags.DEFINE_integer('img_height', 128, 'Input frame height.')

flags.DEFINE_integer('img_width', 416, 'Input frame width.')

flags.DEFINE_integer('queue_size', 2000,
                     'Items in queue. Use smaller number for local debugging.')

flags.DEFINE_integer('seed', 8964, 'Seed for random number generators.')

flags.DEFINE_float('weight_reg', 1e-2, 'The amount of weight regularization to '
                   'apply. This has no effect on the ResNet-based encoder '
                   'architecture.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory to save model '
                    'checkpoints.')

flags.DEFINE_integer('train_steps', int(1e6), 'Number of training steps.')

flags.DEFINE_integer('summary_freq', 100, 'Save summaries every N steps.')

flags.DEFINE_bool('debug', False, 'If true, one training step is performed and '
                  'the results are dumped to a folder for debugging.')

flags.DEFINE_string('input_file', 'train', 'Input file name')

flags.DEFINE_float('rotation_consistency_weight', 1e-3, 'Weight of rotation '
                   'cycle consistency loss.')

flags.DEFINE_float('translation_consistency_weight', 1e-2, 'Weight of '
                   'thanslation consistency loss.')

flags.DEFINE_integer('foreground_dilation', 8, 'Dilation of the foreground '
                     'mask (in pixels).')

flags.DEFINE_boolean('learn_intrinsics', True, 'Whether to learn camera '
                     'intrinsics.')

flags.DEFINE_boolean('boxify', True, 'Whether to convert segmentation masks to '
                     'bounding boxes.')

flags.DEFINE_string('imagenet_ckpt', None, 'Path to an imagenet checkpoint to '
                    'intialize from.')


FLAGS = flags.FLAGS
flags.mark_flag_as_required('data_dir')
flags.mark_flag_as_required('checkpoint_dir')


def load(filename):
  with gfile.Open(filename) as f:
    return np.load(io.BytesIO(f.read()))


def _print_losses(dir1):
  for f in gfile.ListDirectory(dir1):
    if 'loss' in f:
      print ('----------', f, end=' ')
      f1 = os.path.join(dir1, f)
      t1 = load(f1).astype(float)
      print (t1)

def sift_get_fmat(img1, img2, total=100, ratio = 0.8, algo=cv2.FM_LMEDS,
                  random = False, display = False):
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params  = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    #print('kp1.size: {} '.format(len(kp1)))
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            good.append(m)

    sorted_good_mat = sorted(good, key=lambda m: m.distance)
    for m in sorted_good_mat:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    #print ('pts size: ', pts1.size)
    assert pts1.size > 2 and pts2.size > 2
    F, mask = cv2.findFundamentalMat(pts1,pts2,algo)
    if mask is None or np.linalg.matrix_rank(F) != 2:
        return None, None, None
    # assert np.linalg.matrix_rank(F) == 2

    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    if random:
        # Randomly sample top-[total] number of points
        pts = random.sample(zip(pts1, pts2), min(len(pts1), total))
        pts1, pts2 = np.array([ p for p, _ in pts ]), \
                     np.array([ p for _, p in pts ])
    else:
        pts1 = pts1[:min(len(pts1), total)]
        pts2 = pts2[:min(len(pts1), total)]

    if display:
        draw_matches(img1,pts1,img2,pts2,good)

    return F, pts1, pts2

def main(_):
  # Fixed seed for repeatability
  seed = FLAGS.seed
  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  if not gfile.Exists(FLAGS.checkpoint_dir):
    gfile.MakeDirs(FLAGS.checkpoint_dir)

  train_model = model.Model(
      boxify=FLAGS.boxify,
      data_dir=FLAGS.data_dir,
      file_extension=FLAGS.file_extension,
      is_training=True,
      foreground_dilation=FLAGS.foreground_dilation,
      learn_intrinsics=FLAGS.learn_intrinsics,
      learning_rate=FLAGS.learning_rate,
      reconstr_weight=FLAGS.reconstr_weight,
      smooth_weight=FLAGS.smooth_weight,
      ssim_weight=FLAGS.ssim_weight,
      translation_consistency_weight=FLAGS.translation_consistency_weight,
      rotation_consistency_weight=FLAGS.rotation_consistency_weight,
      batch_size=FLAGS.batch_size,
      img_height=FLAGS.img_height,
      img_width=FLAGS.img_width,
      weight_reg=FLAGS.weight_reg,
      depth_consistency_loss_weight=FLAGS.depth_consistency_loss_weight,
      queue_size=FLAGS.queue_size,
      input_file=FLAGS.input_file)

  _train(train_model, FLAGS.checkpoint_dir, FLAGS.train_steps,
         FLAGS.summary_freq)

  if FLAGS.debug:
    _print_losses(os.path.join(FLAGS.checkpoint_dir, 'debug'))


def _train(train_model, checkpoint_dir, train_steps, summary_freq):
  """Runs a trainig loop."""
  saver = train_model.saver
  sv = tf.train.Supervisor(logdir=checkpoint_dir, save_summaries_secs=0,
                           saver=None)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with sv.managed_session(config=config) as sess:
    logging.info('Attempting to resume training from %s...', checkpoint_dir)
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    logging.info('Last checkpoint found: %s', checkpoint)
    if checkpoint:
      saver.restore(sess, checkpoint)
    elif FLAGS.imagenet_ckpt:
      logging.info('Restoring pretrained weights from %s', FLAGS.imagenet_ckpt)
      train_model.imagenet_init_restorer.restore(sess, FLAGS.imagenet_ckpt)

    logging.info('Training...')
    start_time = time.time()
    last_summary_time = time.time()
    steps_per_epoch = train_model.reader.steps_per_epoch
    step = 1
    #train_model.sess = sess
    #print(train_model.sess)
    #print(flaggni)
    print(np.shape(sess.run(train_model.image_stack)))
    #print(swag)
    while step <= train_steps:
      temp_fetch = {
          'frame1' : train_model.frame1,
          'frame2' : train_model.frame2,
          'frame3' : train_model.frame3,
          'left_image': train_model.left_image,
      }
      fetches = {
          'train': train_model.train_op,
          'global_step': train_model.global_step,
      }
      if step % summary_freq == 0:
        fetches['loss'] = train_model.total_loss
        fetches['summary'] = sv.summary_op

      if FLAGS.debug:
        fetches.update(train_model.exports)

      result_img = sess.run(temp_fetch)
      #print(255*(result_img['left_image'][0,:,:,:]))
      #print(np.shape(result_img['frame1'][0,:,:,:]))
      #print(np.shape(result_img['frame2'][0,:,:,:]))
      #print(np.shape(result_img['frame3'][0,:,:,:]))
      #print(swag)
      image_1 = cv2.cvtColor((256*result_img['frame1'][0,:,:,:]).astype('uint8'), cv2.COLOR_BGR2GRAY)
      image_2 = cv2.cvtColor((256*result_img['frame2'][0,:,:,:]).astype('uint8'), cv2.COLOR_BGR2GRAY) 
      image_3 = cv2.cvtColor((256*result_img['frame3'][0,:,:,:]).astype('uint8'), cv2.COLOR_BGR2GRAY)
      F_gt1 = tf.get_default_graph().get_tensor_by_name("compute_loss/F_gt1:0")
      F_gt2 = tf.get_default_graph().get_tensor_by_name("compute_loss/F_gt2:0")
      #print(np.shape(image_1))
      #print(image_2)
      F1,_,_ = sift_get_fmat(image_1, image_2, total=100, ratio = 0.8, algo=cv2.FM_LMEDS, random= False, display = False)
      F2,_,_ = sift_get_fmat(image_2, image_3, total=100, ratio = 0.8, algo=cv2.FM_LMEDS, random= False, display = False)
      #results = sess.run(fetches)
      if not(F1 is None) and not(F2 is None):
        results = sess.run(fetches, feed_dict = {F_gt1 : F1, F_gt2 : F2})
      global_step = results['global_step']
      print('global_step {}'.format(global_step))
      if step % summary_freq == 0:
        sv.summary_writer.add_summary(results['summary'], global_step)
        train_epoch = math.ceil(global_step / steps_per_epoch)
        train_step = global_step - (train_epoch - 1) * steps_per_epoch
        this_cycle = time.time() - last_summary_time
        last_summary_time += this_cycle
        logging.info(
            'Epoch: [%2d] [%5d/%5d] time: %4.2fs (%ds total) loss: %.3f',
            train_epoch, train_step, steps_per_epoch, this_cycle,
            time.time() - start_time, results['loss'])

      if FLAGS.debug:
        debug_dir = os.path.join(checkpoint_dir, 'debug')
        if not gfile.Exists(debug_dir):
          gfile.MkDir(debug_dir)
        for name, tensor in results.iteritems():
          if name == 'summary':
            continue
          s = io.BytesIO()
          filename = os.path.join(debug_dir, name)
          np.save(s, tensor)
          with gfile.Open(filename, 'w') as f:
            f.write(s.getvalue())
        return

      # steps_per_epoch == 0 is intended for debugging, when we run with a
      # single image for sanity check
      if steps_per_epoch == 0 or step % steps_per_epoch == 0:
        logging.info('[*] Saving checkpoint to %s...', checkpoint_dir)
        saver.save(sess, os.path.join(checkpoint_dir, 'model'),
                   global_step=global_step)

      # Setting step to global_step allows for training for a total of
      # train_steps even if the program is restarted during training.
      step = global_step + 1


if __name__ == '__main__':
  app.run(main)
