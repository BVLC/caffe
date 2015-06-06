#!/usr/bin/env python

from collections import OrderedDict
import json
import numpy as np
import pprint
import cPickle as pickle
import string
import sys

# seed the RNG so we evaluate on the same subset each time
np.random.seed(seed=0)

from coco_to_hdf5_data import *
from captioner import Captioner

COCO_EVAL_PATH = './data/coco/coco-caption-eval'
sys.path.append(COCO_EVAL_PATH)
from pycocoevalcap.eval import COCOEvalCap

class CaptionExperiment():
  # captioner is an initialized Captioner (captioner.py)
  # dataset is a dict: image path -> [caption1, caption2, ...]
  def __init__(self, captioner, dataset, dataset_cache_dir, cache_dir, sg):
    self.captioner = captioner
    self.sg = sg
    self.dataset_cache_dir = dataset_cache_dir
    self.cache_dir = cache_dir
    for d in [dataset_cache_dir, cache_dir]:
      if not os.path.exists(d): os.makedirs(d)
    self.dataset = dataset
    self.images = dataset.keys()
    self.init_caption_list(dataset)
    self.caption_scores = [None] * len(self.images)
    print 'Initialized caption experiment: %d images, %d captions' % \
        (len(self.images), len(self.captions))

  def init_caption_list(self, dataset):
    self.captions = []
    for image, captions in dataset.iteritems():
      for caption, _ in captions:
        self.captions.append({'source_image': image, 'caption': caption})
    # Sort by length for performance.
    self.captions.sort(key=lambda c: len(c['caption']))

  def compute_descriptors(self):
    descriptor_filename = '%s/descriptors.npz' % self.dataset_cache_dir
    if os.path.exists(descriptor_filename):
      self.descriptors = np.load(descriptor_filename)['descriptors']
    else:
      self.descriptors = self.captioner.compute_descriptors(self.images)
      np.savez_compressed(descriptor_filename, descriptors=self.descriptors)

  def score_captions(self, image_index, output_name='probs'):
    assert image_index < len(self.images)
    caption_scores_dir = '%s/caption_scores' % self.cache_dir
    if not os.path.exists(caption_scores_dir):
      os.makedirs(caption_scores_dir)
    caption_scores_filename = '%s/scores_image_%06d.pkl' % \
        (caption_scores_dir, image_index)
    if os.path.exists(caption_scores_filename):
      with open(caption_scores_filename, 'rb') as caption_scores_file:
        outputs = pickle.load(caption_scores_file)
    else:
      outputs = self.captioner.score_captions(self.descriptors[image_index],
          self.captions, output_name=output_name, caption_source='gt',
          verbose=False)
      self.caption_stats(image_index, outputs)
      with open(caption_scores_filename, 'wb') as caption_scores_file:
        pickle.dump(outputs, caption_scores_file)
    self.caption_scores[image_index] = outputs

  def caption_stats(self, image_index, caption_scores):
    image_path = self.images[image_index]
    for caption, score in zip(self.captions, caption_scores):
      assert caption['caption'] == score['caption']
      score['stats'] = gen_stats(score['prob'])
      score['correct'] = (image_path == caption['source_image'])

  def eval_image_to_caption(self, image_index, methods=None):
    scores = self.caption_scores[image_index]
    return self.eval_recall(scores, methods=methods)

  def eval_caption_to_image(self, caption_index, methods=None):
    scores = [s[caption_index] for s in self.caption_scores]
    return self.eval_recall(scores, methods=methods)

  def normalize_caption_scores(self, caption_index, stats=['log_p', 'log_p_word']):
    scores = [s[caption_index] for s in self.caption_scores]
    for stat in stats:
      log_stat_scores = np.array([score['stats'][stat] for score in scores])
      stat_scores = np.exp(log_stat_scores)
      mean_stat_score = np.mean(stat_scores)
      log_mean_stat_score = np.log(mean_stat_score)
      for log_stat_score, score in zip(log_stat_scores, scores):
        score['stats']['normalized_' + stat] = log_stat_score - log_mean_stat_score

  def eval_recall(self, scores, methods=None, neg_prefix='negative_'):
    if methods is None:
      # rank on all stats, and all their inverses
      methods = scores[0]['stats'].keys()
      methods += [neg_prefix + method for method in methods]
    correct_ranks = {}
    for method in methods:
      if method.startswith(neg_prefix):
        multiplier = -1
        method_key = method[len(neg_prefix):]
      else:
        multiplier = 1
        method_key = method
      sort_key = lambda s: multiplier * s['stats'][method_key]
      ranked_scores = sorted(scores, key=sort_key)
      for index, score in enumerate(ranked_scores):
        if score['correct']:
          correct_ranks[method] = index
          break
    return correct_ranks

  def recall_results(self, correct_ranks, recall_ranks=[]):
    num_instances = float(len(correct_ranks))
    assert num_instances > 0
    methods = correct_ranks[0].keys()
    results = {}
    for method in methods:
       method_correct_ranks = \
           np.array([correct_rank[method] for correct_rank in correct_ranks])
       r = OrderedDict()
       r['mean'] = np.mean(method_correct_ranks)
       r['median'] = np.median(method_correct_ranks)
       r['mean (1-indexed)'] = r['mean'] + 1
       r['median (1-indexed)'] = r['median'] + 1
       for recall_rank in recall_ranks:
         r['R@%d' % recall_rank] = \
             np.where(method_correct_ranks < recall_rank)[0].shape[0] / num_instances
       results[method] = r
    return results

  def print_recall_results(self, results):
    for method, result in results.iteritems():
      print 'Ranking method:', method
      for metric_name_and_value in result.iteritems():
        print '    %s: %f' % metric_name_and_value

  def retrieval_experiment(self):
    # Compute image descriptors.
    print 'Computing image descriptors'
    self.compute_descriptors()

    num_images, num_captions = len(self.images), len(self.captions)

    # For each image, score all captions.
    for image_index in xrange(num_images):
      sys.stdout.write("\rScoring captions for image %d/%d" %
                       (image_index, num_images))
      sys.stdout.flush()
      self.score_captions(image_index)
    sys.stdout.write('\n')

    # Compute global caption statistics for normalization.
    for caption_index in xrange(num_captions):
      self.normalize_caption_scores(caption_index)

    recall_ranks = [1, 5, 10, 50]

    eval_methods = ['negative_normalized_log_p']
    # Evaluate caption-to-image retrieval task.
    self.caption_to_image_ranks = [None] * num_captions
    for caption_index in xrange(num_captions):
      sys.stdout.write("\rCaption-to-image evaluation: "
                       "computing recall for caption %d/%d" %
                       (caption_index, num_captions))
      sys.stdout.flush()
      self.caption_to_image_ranks[caption_index] = \
          self.eval_caption_to_image(caption_index, methods=eval_methods)
    sys.stdout.write('\n')
    self.caption_to_image_recall = \
         self.recall_results(self.caption_to_image_ranks, recall_ranks)
    print 'Caption-to-image retrieval results:'
    self.print_recall_results(self.caption_to_image_recall)

    # Evaluate image-to-caption retrieval task.
    self.image_to_caption_ranks = [None] * num_images
    for image_index in xrange(num_images):
      sys.stdout.write("\rImage-to-caption evaluation: "
                       "computing recall for image %d/%d" %
                       (image_index, num_images))
      sys.stdout.flush()
      self.image_to_caption_ranks[image_index] = \
          self.eval_image_to_caption(image_index, methods=eval_methods)
    sys.stdout.write('\n')
    self.image_to_caption_recall = \
        self.recall_results(self.image_to_caption_ranks, recall_ranks)
    print 'Image-to-caption retrieval results:'
    self.print_recall_results(self.image_to_caption_recall)

  def generation_experiment(self, strategy, max_batch_size=1000):
    # Compute image descriptors.
    print 'Computing image descriptors'
    self.compute_descriptors()

    do_batches = (strategy['type'] == 'beam' and strategy['beam_size'] == 1) or \
        (strategy['type'] == 'sample' and
         ('temp' not in strategy or strategy['temp'] in (1, float('inf'))) and
         ('num' not in strategy or strategy['num'] == 1))

    num_images = len(self.images)
    batch_size = min(max_batch_size, num_images) if do_batches else 1

    # Generate captions for all images.
    all_captions = [None] * num_images
    for image_index in xrange(0, num_images, batch_size):
      batch_end_index = min(image_index + batch_size, num_images)
      sys.stdout.write("\rGenerating captions for image %d/%d" %
                       (image_index, num_images))
      sys.stdout.flush()
      if do_batches:
        if strategy['type'] == 'beam' or \
            ('temp' in strategy and strategy['temp'] == float('inf')):
          temp = float('inf')
        else:
          temp = strategy['temp'] if 'temp' in strategy else 1
        output_captions, output_probs = self.captioner.sample_captions(
            self.descriptors[image_index:batch_end_index], temp=temp)
        for batch_index, output in zip(range(image_index, batch_end_index),
                                       output_captions):
          all_captions[batch_index] = output
      else:
        for batch_image_index in xrange(image_index, batch_end_index):
          captions, caption_probs = self.captioner.predict_caption(
              self.descriptors[batch_image_index], strategy=strategy)
          best_caption, max_log_prob = None, None
          for caption, probs in zip(captions, caption_probs):
            log_prob = gen_stats(probs)['log_p']
            if best_caption is None or \
                (best_caption is not None and log_prob > max_log_prob):
              best_caption, max_log_prob = caption, log_prob
          all_captions[batch_image_index] = best_caption
    sys.stdout.write('\n')

    # Compute the number of reference files as the maximum number of ground
    # truth captions of any image in the dataset.
    num_reference_files = 0
    for captions in self.dataset.values():
      if len(captions) > num_reference_files:
        num_reference_files = len(captions)
    if num_reference_files <= 0:
      raise Exception('No reference captions.')

    # Collect model/reference captions, formatting the model's captions and
    # each set of reference captions as a list of len(self.images) strings.
    exp_dir = '%s/generation' % self.cache_dir
    if not os.path.exists(exp_dir):
      os.makedirs(exp_dir)
    # For each image, write out the highest probability caption.
    model_captions = [''] * len(self.images)
    reference_captions = [([''] * len(self.images)) for _ in xrange(num_reference_files)]
    for image_index, image in enumerate(self.images):
      caption = self.captioner.sentence(all_captions[image_index])
      model_captions[image_index] = caption
      for reference_index, (_, caption) in enumerate(self.dataset[image]):
        caption = ' '.join(caption)
        reference_captions[reference_index][image_index] = caption

    coco_image_ids = [self.sg.image_path_to_id[image_path]
                      for image_path in self.images]
    generation_result = [{
      'image_id': self.sg.image_path_to_id[image_path],
      'caption': model_captions[image_index]
    } for (image_index, image_path) in enumerate(self.images)]
    json_filename = '%s/generation_result.json' % self.cache_dir
    print 'Dumping result to file: %s' % json_filename
    with open(json_filename, 'w') as json_file:
      json.dump(generation_result, json_file)
    generation_result = self.sg.coco.loadRes(json_filename)
    coco_evaluator = COCOEvalCap(self.sg.coco, generation_result)
    coco_evaluator.params['image_id'] = coco_image_ids
    coco_evaluator.evaluate()

def gen_stats(prob):
  stats = {}
  stats['length'] = len(prob)
  stats['log_p'] = 0.0
  eps = 1e-12
  for p in prob:
    assert 0.0 <= p <= 1.0
    stats['log_p'] += np.log(max(eps, p))
  stats['log_p_word'] = stats['log_p'] / stats['length']
  try:
    stats['perplex'] = np.exp(-stats['log_p'])
  except OverflowError:
    stats['perplex'] = float('inf')
  try:
    stats['perplex_word'] = np.exp(-stats['log_p_word'])
  except OverflowError:
    stats['perplex_word'] = float('inf')
  return stats

def main():
  MAX_IMAGES = -1  # -1 to use all images
  TAG = 'coco_2layer_factored'
  if MAX_IMAGES >= 0:
    TAG += '_%dimages' % MAX_IMAGES
  eval_on_test = False
  if eval_on_test:
    ITER = 100000
    MODEL_FILENAME = 'lrcn_finetune_trainval_stepsize40k_iter_%d' % ITER
    DATASET_NAME = 'test'
  else:  # eval on val
    ITER = 50000
    MODEL_FILENAME = 'lrcn_finetune_iter_%d' % ITER
    DATASET_NAME = 'val'
  TAG += '_%s' % DATASET_NAME
  MODEL_DIR = './examples/coco_caption'
  MODEL_FILE = '%s/%s.caffemodel' % (MODEL_DIR, MODEL_FILENAME)
  IMAGE_NET_FILE = './models/bvlc_reference_caffenet/deploy.prototxt'
  LSTM_NET_FILE = './examples/coco_caption/lrcn_word_to_preds.deploy.prototxt'
  NET_TAG = '%s_%s' % (TAG, MODEL_FILENAME)
  DATASET_SUBDIR = '%s/%s_ims' % (DATASET_NAME,
      str(MAX_IMAGES) if MAX_IMAGES >= 0 else 'all')
  DATASET_CACHE_DIR = './retrieval_cache/%s/%s' % (DATASET_SUBDIR, MODEL_FILENAME)
  VOCAB_FILE = './examples/coco_caption/h5_data/buffer_100/vocabulary.txt'
  DEVICE_ID = 0
  with open(VOCAB_FILE, 'r') as vocab_file:
    vocab = [line.strip() for line in vocab_file.readlines()]
  coco = COCO(COCO_ANNO_PATH % DATASET_NAME)
  image_root = COCO_IMAGE_PATTERN % DATASET_NAME
  sg = CocoSequenceGenerator(coco, BUFFER_SIZE, image_root, vocab=vocab,
                             align=False, shuffle=False)
  dataset = {}
  for image_path, sentence in sg.image_sentence_pairs:
    if image_path not in dataset:
      dataset[image_path] = []
    dataset[image_path].append((sg.line_to_stream(sentence), sentence))
  print 'Original dataset contains %d images' % len(dataset.keys())
  if 0 <= MAX_IMAGES < len(dataset.keys()):
    all_keys = dataset.keys()
    perm = np.random.permutation(len(all_keys))[:MAX_IMAGES]
    chosen_keys = set([all_keys[p] for p in perm])
    for key in all_keys:
      if key not in chosen_keys:
        del dataset[key]
    print 'Reduced dataset to %d images' % len(dataset.keys())
  if MAX_IMAGES < 0: MAX_IMAGES = len(dataset.keys())
  captioner = Captioner(MODEL_FILE, IMAGE_NET_FILE, LSTM_NET_FILE, VOCAB_FILE,
                        device_id=DEVICE_ID)
  beam_size = 1
  generation_strategy = {'type': 'beam', 'beam_size': beam_size}
  if generation_strategy['type'] == 'beam':
    strategy_name = 'beam%d' % generation_strategy['beam_size']
  elif generation_strategy['type'] == 'sample':
    strategy_name = 'sample%f' % generation_strategy['temp']
  else:
    raise Exception('Unknown generation strategy type: %s' % generation_strategy['type'])
  CACHE_DIR = '%s/%s' % (DATASET_CACHE_DIR, strategy_name)
  experimenter = CaptionExperiment(captioner, dataset, DATASET_CACHE_DIR, CACHE_DIR, sg)
  captioner.set_image_batch_size(min(100, MAX_IMAGES))
  experimenter.generation_experiment(generation_strategy)
  captioner.set_caption_batch_size(min(MAX_IMAGES * 5, 1000))
  experimenter.retrieval_experiment()

if __name__ == "__main__":
  main()
