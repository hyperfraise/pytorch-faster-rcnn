# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.nms.pth_nms import pth_nms
from lib.nms.pth_nms import pth_nms_batch


def nms_batch(dets, thresh, network_device):
  """Dispatch to either CPU or GPU NMS implementations.
  Accept dets as tensor"""
  return pth_nms_batch(dets, thresh, network_device)

def nms(dets, thresh):
  """Dispatch to either CPU or GPU NMS implementations.
  Accept dets as tensor"""
  return pth_nms(dets, thresh)
