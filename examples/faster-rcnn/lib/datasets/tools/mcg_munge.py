import os
import sys

"""Hacky tool to convert file system layout of MCG boxes downloaded from
http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/
so that it's consistent with those computed by Jan Hosang (see:
http://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-
  computing/research/object-recognition-and-scene-understanding/how-
  good-are-detection-proposals-really/)

NB: Boxes from the MCG website are in (y1, x1, y2, x2) order.
Boxes from Hosang et al. are in (x1, y1, x2, y2) order.
"""

def munge(src_dir):
    # stored as: ./MCG-COCO-val2014-boxes/COCO_val2014_000000193401.mat
    # want:      ./MCG/mat/COCO_val2014_0/COCO_val2014_000000141/COCO_val2014_000000141334.mat

    files = os.listdir(src_dir)
    for fn in files:
        base, ext = os.path.splitext(fn)
        # first 14 chars / first 22 chars / all chars + .mat
        # COCO_val2014_0/COCO_val2014_000000447/COCO_val2014_000000447991.mat
        first = base[:14]
        second = base[:22]
        dst_dir = os.path.join('MCG', 'mat', first, second)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        src = os.path.join(src_dir, fn)
        dst = os.path.join(dst_dir, fn)
        print 'MV: {} -> {}'.format(src, dst)
        os.rename(src, dst)

if __name__ == '__main__':
    # src_dir should look something like:
    #  src_dir = 'MCG-COCO-val2014-boxes'
    src_dir = sys.argv[1]
    munge(src_dir)
