import tensorflow as tf
from config import _SIZE, _SCORE_THRESHOLD, _IOU_THRESHOLD

def compute_boxes(inputs, anchors, num_classes, ratio):
    ###################################################################
    #              [Batch, Height, Width, Anchors, Prediction]        #
    #    inputs -> [None, (13 | 26 | 52), (13 | 26 | 52), 255]        #
    ###################################################################   
                                                         
    input_shape = tf.cast([_SIZE, _SIZE], dtype=tf.float32)
    with tf.name_scope('GRID_CELL'):
        
        grid_shape = tf.shape(inputs)[1:3]  
        x, y = tf.meshgrid(tf.range(grid_shape[1]), tf.range(grid_shape[0]))        
        grid_x = tf.reshape(x, [grid_shape[0], grid_shape[1], 1, 1])  # shape=[13, 13, 1, 1]
        grid_y = tf.reshape(y, [grid_shape[0], grid_shape[1], 1, 1])  # shape=[13, 13, 1, 1]
        grid = tf.concat([grid_x, grid_y], axis=-1)  # shape=[13, 13, 1, 2]
        grid = tf.cast(grid, dtype=inputs.dtype)

    num_anchors = len(anchors)  # 3
    anchors_tensor = tf.cast(anchors, dtype=inputs.dtype)
    anchors_tensor = tf.reshape(anchors_tensor, [1, 1, 1, num_anchors, 2])  # shape=[1,1,1,3,2]

    # Reshape [None, 13, 13, 255] =>[None, 13, 13, 3, 85]
    inputs_re = tf.reshape(inputs, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    box_xy, box_wh, box_confidence, box_class_probs = tf.split(inputs_re, [2, 2, 1, num_classes], axis=-1)
    
    with tf.name_scope('GRID_LOSS'):  
        
        #b(x,y) = σ(t(x,y)) + GRID_CELL(x,y)
        box_xy = (tf.sigmoid(box_xy) + grid) / tf.cast(grid_shape, dtype=inputs.dtype)
        # b(w,h) = p(w,h) * e^t(w ,h)                                
        box_wh = (anchors_tensor * tf.exp(box_wh))  / tf.cast(input_shape, dtype=inputs.dtype)
        # b(c) = p(object) * IOU
        box_confidence = tf.sigmoid(box_confidence) 
        # b(class) = p(class)
        box_class_probs = tf.sigmoid(box_class_probs)
 
    ##################################################################
    #    Need to reverse the (x, y) --> (y, x) | (w, h) --> (h, w)   # 
    #    Reshape the box_xy -> [507, 2] | box_wh -> [507, 2],        #
    #                 ratio -> [h/416, w/416]                        #
    ##################################################################
    with tf.name_scope('BOX_RECOVER'):
        
    
        box_xy = tf.reshape(box_xy, [-1, 2])
        box_yx = box_xy[..., ::-1]  
        box_wh = tf.reshape(box_wh, [-1, 2])
        box_hw = box_wh[..., ::-1]
    
        box_yx *= ratio[::-1] 
        box_hw *= ratio 
    
        box_min = box_yx - (box_hw / 2.)  
        box_max = box_yx + (box_hw / 2.) 
    
        box = tf.concat([box_min, box_max], axis=-1)
        box = tf.multiply(box, tf.concat([input_shape, input_shape], axis=-1))
        
        box_score = box_confidence * box_class_probs  
        box_score = tf.reshape(box_score, [-1, num_classes])  
         
                                         
    return box, box_score


def predict(inputs, anchors, num_classes, ratio, max_boxes=20, score_threshold=_SCORE_THRESHOLD, iou_threshold=_IOU_THRESHOLD):

    boxes = []
    box_scores = []
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # 3 scale [13, 13, 255] [26, 26, 255], [52, 52, 255]
    for mask in range(3):  
        box, score = compute_boxes(inputs[mask], anchors[anchor_mask[mask]], num_classes, ratio)    
        boxes.append(box)
        box_scores.append(score)
    
    boxes = tf.concat(boxes, axis=0)
    box_scores = tf.concat(box_scores, axis=0)
    
    return boxes, box_scores

