# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.model.config import cfg
from lib.model.bbox_transform import bbox_transform_inv, clip_boxes
import numpy.random as npr

import torch
import signal

def proposal_top_layer_batch(rpn_cls_probs, rpn_bbox_preds, im_info, _feat_stride, anchors, num_anchors, network_device):
  """A layer that just selects the top region proposals
     without using non-maximal suppression,
     For details please see the technical report
  """
  rpn_top_n = cfg.TEST.RPN_TOP_N

  scores = rpn_cls_probs[:, :, :, num_anchors:]

  rpn_bbox_preds = rpn_bbox_preds.view(rpn_bbox_preds.size(0),-1, 4)
  scores = scores.contiguous().view(rpn_bbox_preds.size(0),-1, 1)

  blobs, scoress = [],[]
  for i in range(scores.size(0)):
      score = scores[i]
      length = score.size(0)
      if length < rpn_top_n:
        # Random selection, maybe unnecessary and loses good proposals
        # But such case rarely happens
        top_inds = torch.from_numpy(npr.choice(length, size=rpn_top_n, replace=True)).long().cuda(network_device)
      else:
        top_inds = score.sort(0, descending=True)[1]
        top_inds = top_inds[:rpn_top_n]
        top_inds = top_inds.view(rpn_top_n)

      # Do the selection here
      anchor = anchors[top_inds, :].contiguous()
      rpn_bbox_pred = rpn_bbox_preds[i][top_inds, :].contiguous()
      score = score[top_inds].contiguous()

      # Convert anchors into proposals via bbox transformations
      proposal = bbox_transform_inv(anchor, rpn_bbox_pred)
      # Clip predicted boxes to image
      proposal = clip_boxes(proposal, im_info[:2])

      # Output rois blob
      # Our RPN implementation only supports a single input image, so all
      # batch inds are 0
      batch_inds = proposal.new_zeros(proposal.size(0), 1)
      blob = torch.cat([batch_inds, proposal], 1)
      blobs.append(blob)
      scoress.append(score)
  return blobs, scoress

def proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, im_info, _feat_stride, anchors, num_anchors):
  """A layer that just selects the top region proposals
     without using non-maximal suppression,
     For details please see the technical report
  """
  rpn_top_n = cfg.TEST.RPN_TOP_N

  scores = rpn_cls_prob[:, :, :, num_anchors:]

  rpn_bbox_pred = rpn_bbox_pred.view(-1, 4)
  scores = scores.contiguous().view(-1, 1)

  length = scores.size(0)
  if length < rpn_top_n:
    # Random selection, maybe unnecessary and loses good proposals
    # But such case rarely happens
    top_inds = torch.from_numpy(npr.choice(length, size=rpn_top_n, replace=True)).long().to(anchors.device)
  else:
    top_inds = scores.sort(0, descending=True)[1]
    top_inds = top_inds[:rpn_top_n]
    top_inds = top_inds.view(rpn_top_n)

  # Do the selection here
  anchors = anchors[top_inds, :].contiguous()
  rpn_bbox_pred = rpn_bbox_pred[top_inds, :].contiguous()
  scores = scores[top_inds].contiguous()

  # Convert anchors into proposals via bbox transformations
  proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
  #proposals = torch.zeros((5000,4))

  # Clip predicted boxes to image
  proposals = clip_boxes(proposals, im_info[:2])


  # Output rois blob
  # Our RPN implementation only supports a single input image, so all
  # batch inds are 0
  batch_inds = proposals.new_zeros(proposals.size(0), 1)
  blob = torch.cat([batch_inds, proposals], 1)
  return blob, scores
