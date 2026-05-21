from torchvision.ops import nms as tv_nms

def nms(boxes, scores, nms_thresh):
    """Direct wrapper around torchvision NMS.
    Args:
        boxes: can be BoxList or Tensor
        scores: Tensor of scores  
        nms_thresh: float threshold
    """
    # If called with BoxList (from boxlist_ops.py), handle it
    if hasattr(boxes, 'bbox'):
        # boxes is a BoxList object - this is the old-style call
        boxlist = boxes
        score_field = nms_thresh  # in old call: nms(boxlist, nms_thresh, ...)
        # This path should NOT be reached anymore since boxlist_ops calls _box_nms directly
        raise RuntimeError("nms() called with BoxList - check import paths")
    
    # Normal call: boxes is tensor, scores is tensor, nms_thresh is float
    return tv_nms(boxes, scores, float(nms_thresh))
