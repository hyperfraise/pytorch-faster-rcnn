# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.model.config import cfg
from lib.model.bbox_transform import bbox_transform_inv, clip_boxes, bbox_transform_inv_batch
from lib.model.nms_wrapper import nms,nms_batch

import torch

def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.view((-1, 4))
  scores = scores.contiguous().view(-1, 1)
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  proposals = clip_boxes(proposals, im_info[:2])

  # Pick the top region proposals
  scores, order = scores.view(-1).sort(descending=True)
  if pre_nms_topN > 0:
    order = order[:pre_nms_topN]
    scores = scores[:pre_nms_topN].view(-1, 1)
  proposals = proposals[order.data, :]

  # Non-maximal suppression
  keep = nms(torch.cat((proposals, scores), 1).data, nms_thresh)

  # Pick th top region proposals after NMS
  if post_nms_topN > 0:
    keep = keep[:post_nms_topN]
  proposals = proposals[keep, :]
  scores = scores[keep,]

  # Only support single image as input
  batch_inds = proposals.new_zeros(proposals.size(0), 1)
  blob = torch.cat((batch_inds, proposals), 1)

  return blob, scores


def proposal_layer_batch(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors, network_device):
  """A simplified version compared to fast/er RCNN
     For details please see the technical report
  """
  if type(cfg_key) == bytes:
      cfg_key = cfg_key.decode('utf-8')
  pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
  post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
  nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

  # Get the scores and bounding boxes
  scores = rpn_cls_prob[:, :, :, num_anchors:]
  rpn_bbox_pred = rpn_bbox_pred.view((rpn_bbox_pred.size(0),-1, 4))
  scores = scores.contiguous().view(scores.size(0),-1, 1)
  proposals = bbox_transform_inv_batch(anchors, rpn_bbox_pred)#here bug
  proposals = list(map(lambda x : clip_boxes(x, im_info[:2]),proposals))

  blobs, scoress = [],[]
  for i in range(scores.size(0)):
      # Pick the top region proposals
      score, order = scores[i].view(-1).sort(descending=True)
      if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
        score = score[:pre_nms_topN].view(-1, 1)
      proposal = proposals[i][order.data, :]

      # Non-maximal suppression
      keep = nms_batch(torch.cat((proposal, score), 1).data, nms_thresh, network_device)
      # Pick th top region proposals after NMS
      if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
      proposal = proposal[keep]
      score = score[keep]

      # Only support single image as input
      batch_inds = proposal.new_zeros(proposal.size(0), 1)
      blob = torch.cat((batch_inds, proposal), 1)
      blobs.append(blob)
      scoress.append(score)
  return blobs, scoress
