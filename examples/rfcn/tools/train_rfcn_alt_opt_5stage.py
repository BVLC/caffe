#!/usr/bin/env python

# --------------------------------------------------------
# R-FCN
# Copyright (c) 2016 Yuwen Xiong, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""Train a R-FCN network using alternating optimization.
This tool implements the alternating optimization algorithm described in our
NIPS 2015 paper ("R-FCN: Towards Real-time Object Detection with Region
Proposal Networks." Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun.)
"""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from rpn.generate import imdb_proposals, imdb_rpn_compute_stats
import argparse
import pprint
import numpy as np
import sys, os
import multiprocessing as mp
import cPickle
import shutil


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R-FCN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--net_name', dest='net_name',
                        help='network name (e.g., "ResNet-101")',
                        default=None, type=str)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--imdb_test', dest='imdb_test_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--model', dest='model_name',
                        help='folder name of model',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def get_roidb(imdb_name, rpn_file=None):
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
    if rpn_file is not None:
        imdb.config['rpn_file'] = rpn_file
    roidb = get_training_roidb(imdb)
    return roidb, imdb


def get_solvers(imdb_name, net_name, model_name):
    # R-FCN Alternating Optimization
    # Solver for each training stage
    if imdb_name.startswith('coco'):
        solvers = [[net_name, model_name, 'stage1_rpn_solver360k480k.pt'],
                   [net_name, model_name, 'stage1_rfcn_ohem_solver360k480k.pt'],
                   [net_name, model_name, 'stage2_rpn_solver360k480k.pt'],
                   [net_name, model_name, 'stage2_rfcn_ohem_solver360k480k.pt'],
                   [net_name, model_name, 'stage3_rpn_solver360k480k.pt']]
        solvers = [os.path.join('.', 'models', 'coco', *s) for s in solvers]
        # Iterations for each training stage
        max_iters = [480000, 480000, 480000, 480000, 480000]
        # Test prototxt for the RPN
        rpn_test_prototxt = os.path.join(
            '.', 'models', 'coco', net_name, model_name, 'rpn_test.pt')
    else:
        solvers = [[net_name, model_name, 'stage1_rpn_solver60k80k.pt'],
                   [net_name, model_name, 'stage1_rfcn_ohem_solver80k120k.pt'],
                   [net_name, model_name, 'stage2_rpn_solver60k80k.pt'],
                   [net_name, model_name, 'stage2_rfcn_ohem_solver80k120k.pt'],
                   [net_name, model_name, 'stage3_rpn_solver60k80k.pt']]
        solvers = [os.path.join(cfg.MODELS_DIR, *s) for s in solvers]
        # Iterations for each training stage
        max_iters = [80000, 120000, 80000, 120000, 80000]
        # Test prototxt for the RPN
        rpn_test_prototxt = os.path.join(
            cfg.MODELS_DIR, net_name, model_name, 'rpn_test.pt')
    return solvers, max_iters, rpn_test_prototxt


def _init_caffe(cfg):
    """Initialize pycaffe in a training process.
    """

    import caffe
    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)
    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)


def train_rpn(queue=None, imdb_name=None, init_model=None, solver=None,
              max_iters=None, cfg=None, output_cache=None):
    """Train a Region Proposal Network in a separate training process.
    """

    # Not using any proposals, just ground-truth boxes
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.BBOX_REG = False  # applies only to R-FCN bbox regression
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.IMS_PER_BATCH = 1
    print 'Init model: {}'.format(init_model)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name)
    print 'roidb len: {}'.format(len(roidb))
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    final_caffemodel = os.path.join(output_dir, output_cache)

    if os.path.exists(final_caffemodel):
        queue.put({'model_path': final_caffemodel})
    else:
        model_paths = train_net(solver, roidb, output_dir,
                                pretrained_model=init_model,
                                max_iters=max_iters)
        # Cleanup all but the final model
        for i in model_paths[:-1]:
            os.remove(i)
        rpn_model_path = model_paths[-1]
        # Send final model path through the multiprocessing queue
        shutil.copyfile(rpn_model_path, final_caffemodel)
        queue.put({'model_path': final_caffemodel})


def rpn_generate(queue=None, imdb_name=None, rpn_model_path=None, cfg=None,
                 rpn_test_prototxt=None):
    """Use a trained RPN to generate proposals.
    """

    cfg.TEST.RPN_PRE_NMS_TOP_N = 6000     # no pre NMS filtering
    cfg.TEST.RPN_POST_NMS_TOP_N = 300  # limit top boxes after NMS
    print 'RPN model: {}'.format(rpn_model_path)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    # NOTE: the matlab implementation computes proposals on flipped images, too.
    # We compute them on the image once and then flip the already computed
    # proposals. This might cause a minor loss in mAP (less proposal jittering).
    imdb = get_imdb(imdb_name)
    print 'Loaded dataset `{:s}` for proposal generation'.format(imdb.name)

    # Load RPN and configure output directory
    rpn_net = caffe.Net(rpn_test_prototxt, rpn_model_path, caffe.TEST)
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    rpn_net_name = os.path.splitext(os.path.basename(rpn_model_path))[0]
    rpn_proposals_path = os.path.join(
        output_dir, rpn_net_name + '_proposals.pkl')

    # Generate proposals on the imdb

    # Write proposals to disk and send the proposal file path through the
    # multiprocessing queue
    if not os.path.exists(rpn_proposals_path):
        rpn_proposals = imdb_proposals(rpn_net, imdb)
        with open(rpn_proposals_path, 'wb') as f:
            cPickle.dump(rpn_proposals, f, cPickle.HIGHEST_PROTOCOL)
    queue.put({'proposal_path': rpn_proposals_path})
    print 'Wrote RPN proposals to {}'.format(rpn_proposals_path)


def train_rfcn(queue=None, imdb_name=None, init_model=None, solver=None,
                    max_iters=None, cfg=None, rpn_file=None, output_cache=None):
    """Train a R-FCN using proposals generated by an RPN.
    """

    cfg.TRAIN.HAS_RPN = False           # not generating prosals on-the-fly
    cfg.TRAIN.PROPOSAL_METHOD = 'rpn'   # use pre-computed RPN proposals instead
    cfg.TRAIN.IMS_PER_BATCH = 1
    print 'Init model: {}'.format(init_model)
    print 'RPN proposals: {}'.format(rpn_file)
    print('Using config:')
    pprint.pprint(cfg)

    import caffe
    _init_caffe(cfg)

    roidb, imdb = get_roidb(imdb_name, rpn_file=rpn_file)
    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # Train R-FCN
    # Send R-FCN model path over the multiprocessing queue
    final_caffemodel = os.path.join(output_dir, output_cache)

    if os.path.exists(final_caffemodel):
        queue.put({'model_path': final_caffemodel})
    else:
        model_paths = train_net(solver, roidb, output_dir,
                                pretrained_model=init_model,
                                max_iters=max_iters)
        # Cleanup all but the final model
        for i in model_paths[:-1]:
            os.remove(i)
        rfcn_model_path = model_paths[-1]
        # Send final model path through the multiprocessing queue
        shutil.copyfile(rfcn_model_path, final_caffemodel)
        queue.put({'model_path': final_caffemodel})


def rpn_compute_stats(queue=None, imdb_name=None, cfg=None, rpn_test_prototxt=None):
    """Compute mean stds for anchors
    """
    cfg.TRAIN.HAS_RPN = True
    cfg.TRAIN.BBOX_REG = False  # applies only to R-FCN bbox regression
    cfg.TRAIN.PROPOSAL_METHOD = 'gt'
    cfg.TRAIN.IMS_PER_BATCH = 1

    import caffe
    _init_caffe(cfg)

    # NOTE: the matlab implementation computes proposals on flipped images, too.
    # We compute them on the image once and then flip the already computed
    # proposals. This might cause a minor loss in mAP (less proposal jittering).
    roidb, imdb = get_roidb(imdb_name)
    print 'Loaded dataset `{:s}` for proposal generation'.format(imdb.name)
    mean_file = os.path.join(imdb.cache_path, imdb.name + '_means.npy')
    std_file = os.path.join(imdb.cache_path, imdb.name + '_stds.npy')
    if os.path.exists(mean_file) and os.path.exists(std_file):
        means = np.load(mean_file)
        stds = np.load(std_file)
    else:
        # Load RPN and configure output directory
        rpn_net = caffe.Net(rpn_test_prototxt, caffe.TEST)
        # Generate proposals on the imdb
        print 'start computing means/stds, it may take several minutes...'
        if imdb_name.startswith('coco'):
            means, stds = imdb_rpn_compute_stats(rpn_net, imdb, anchor_scales=(4, 8, 16, 32))
        else:
            means, stds = imdb_rpn_compute_stats(rpn_net, imdb, anchor_scales=(8, 16, 32))
        np.save(mean_file, means)
        np.save(std_file, stds)
    queue.put({'means': means, 'stds': stds})


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    cfg.GPU_ID = args.gpu_id

    # --------------------------------------------------------------------------
    # Pycaffe doesn't reliably free GPU memory when instantiated nets are
    # discarded (e.g. "del net" in Python code). To work around this issue, each
    # training stage is executed in a separate process using
    # multiprocessing.Process.
    # --------------------------------------------------------------------------

    # queue for communicated results between processes
    mp_queue = mp.Queue()
    # solves, iters, etc. for each training stage
    solvers, max_iters, rpn_test_prototxt = get_solvers(args.imdb_name, args.net_name, args.model_name)

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 0 RPN, compute normalization means and stds'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    mp_kwargs = dict(
        queue=mp_queue,
        imdb_name=args.imdb_name,
        cfg=cfg,
        rpn_test_prototxt=rpn_test_prototxt)
    p = mp.Process(target=rpn_compute_stats, kwargs=mp_kwargs)
    p.start()
    stage0_anchor_stats = mp_queue.get()
    p.join()
    cfg.TRAIN.RPN_NORMALIZE_MEANS = stage0_anchor_stats['means']
    cfg.TRAIN.RPN_NORMALIZE_STDS = stage0_anchor_stats['stds']

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.imdb_name,
            init_model=args.pretrained_model,
            solver=solvers[0],
            max_iters=max_iters[0],
            cfg=cfg,
            output_cache='stage1_rpn_final.caffemodel')
    p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
    p.start()
    rpn_stage1_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.imdb_name,
            rpn_model_path=str(rpn_stage1_out['model_path']),
            cfg=cfg,
            rpn_test_prototxt=rpn_test_prototxt)
    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage1_out['proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    mp_kwargs = dict(
        queue=mp_queue,
        imdb_name=args.imdb_test_name,
        rpn_model_path=str(rpn_stage1_out['model_path']),
        cfg=cfg,
        rpn_test_prototxt=rpn_test_prototxt)
    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage1_out['test_proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 1 R-FCN using RPN proposals, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage1'
    mp_kwargs = dict(
            queue=mp_queue,
            imdb_name=args.imdb_name,
            init_model=args.pretrained_model,
            solver=solvers[1],
            max_iters=max_iters[1],
            cfg=cfg,
            rpn_file=rpn_stage1_out['proposal_path'],
            output_cache='stage1_rfcn_final.caffemodel')
    p = mp.Process(target=train_rfcn, kwargs=mp_kwargs)
    p.start()
    rfcn_stage1_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 RPN, init from stage1 R-FCN model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'
    mp_kwargs = dict(
        queue=mp_queue,
        imdb_name=args.imdb_name,
        init_model=str(rfcn_stage1_out['model_path']),
        solver=solvers[2],
        max_iters=max_iters[2],
        cfg=cfg,
        output_cache='stage2_rpn_final.caffemodel')
    p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
    p.start()
    rpn_stage2_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 RPN, generate proposals'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    mp_kwargs = dict(
        queue=mp_queue,
        imdb_name=args.imdb_name,
        rpn_model_path=str(rpn_stage2_out['model_path']),
        cfg=cfg,
        rpn_test_prototxt=rpn_test_prototxt)
    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage2_out['proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    mp_kwargs = dict(
        queue=mp_queue,
        imdb_name=args.imdb_test_name,
        rpn_model_path=str(rpn_stage2_out['model_path']),
        cfg=cfg,
        rpn_test_prototxt=rpn_test_prototxt)
    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage2_out['test_proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 2 R-FCN using Stage-2 RPN proposals, init from ImageNet model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage2'
    mp_kwargs = dict(
        queue=mp_queue,
        imdb_name=args.imdb_name,
        init_model=args.pretrained_model,
        solver=solvers[3],
        max_iters=max_iters[3],
        cfg=cfg,
        rpn_file=rpn_stage2_out['proposal_path'],
        output_cache='stage2_rfcn_final.caffemodel')
    p = mp.Process(target=train_rfcn, kwargs=mp_kwargs)
    p.start()
    rfcn_stage2_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 3 RPN, init from stage1 R-FCN model'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    cfg.TRAIN.SNAPSHOT_INFIX = 'stage3'
    mp_kwargs = dict(
        queue=mp_queue,
        imdb_name=args.imdb_name,
        init_model=str(rfcn_stage2_out['model_path']),
        solver=solvers[4],
        max_iters=max_iters[4],
        cfg=cfg,
        output_cache='stage3_rpn_final.caffemodel')
    p = mp.Process(target=train_rpn, kwargs=mp_kwargs)
    p.start()
    rpn_stage3_out = mp_queue.get()
    p.join()

    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
    print 'Stage 3 RPN, generate test proposals only'
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    mp_kwargs = dict(
        queue=mp_queue,
        imdb_name=args.imdb_test_name,
        rpn_model_path=str(rpn_stage3_out['model_path']),
        cfg=cfg,
        rpn_test_prototxt=rpn_test_prototxt)
    p = mp.Process(target=rpn_generate, kwargs=mp_kwargs)
    p.start()
    rpn_stage3_out['test_proposal_path'] = mp_queue.get()['proposal_path']
    p.join()

    print 'Final model: {}'.format(str(rfcn_stage2_out['model_path']))
    print 'Final RPN: {}'.format(str(rpn_stage3_out['test_proposal_path']))


