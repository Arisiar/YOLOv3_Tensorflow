# YOLOv3_Tensorflow

A simple tenssorflow implement for yolov3. 
(you can get the weight in the releases)

For more detail:

- [paper](https://arxiv.org/abs/1804.02767)

Here are some result in our test:

<img src="./img/input.jpg" width="900px/">

<img src="./img/result.jpg" width="900px/">


## Setup

- Windows 10
- Tensorflow 1.6.0
- Python 3.6.4

## How to use
``` bash
python main.py --input_img [YOUR INPUT] --output_img [YOUR OUTPUT]
``` 

## Detail

- Network Architecture

┌────────────┬────────────────────────┬───────────────────┐
│    Name    │        Filters         │ Output Dimension  │
├────────────┼────────────────────────┼───────────────────┤
│ Conv 1     │ 7 x 7 x 64, stride=2   │ 224 x 224 x 64    │
│ Max Pool 1 │ 2 x 2, stride=2        │ 112 x 112 x 64    │
│ Conv 2     │ 3 x 3 x 192            │ 112 x 112 x 192   │
│ Max Pool 2 │ 2 x 2, stride=2        │ 56 x 56 x 192     │
│ Conv 3     │ 1 x 1 x 128            │ 56 x 56 x 128     │
│ Conv 4     │ 3 x 3 x 256            │ 56 x 56 x 256     │
│ Conv 5     │ 1 x 1 x 256            │ 56 x 56 x 256     │
│ Conv 6     │ 1 x 1 x 512            │ 56 x 56 x 512     │
│ Max Pool 3 │ 2 x 2, stride=2        │ 28 x 28 x 512     │
│ Conv 7     │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 8     │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 9     │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 10    │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 11    │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 12    │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 13    │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 14    │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 15    │ 1 x 1 x 512            │ 28 x 28 x 512     │
│ Conv 16    │ 3 x 3 x 1024           │ 28 x 28 x 1024    │
│ Max Pool 4 │ 2 x 2, stride=2        │ 14 x 14 x 1024    │
│ Conv 17    │ 1 x 1 x 512            │ 14 x 14 x 512     │
│ Conv 18    │ 3 x 3 x 1024           │ 14 x 14 x 1024    │
│ Conv 19    │ 1 x 1 x 512            │ 14 x 14 x 512     │
│ Conv 20    │ 3 x 3 x 1024           │ 14 x 14 x 1024    │
│ Conv 21    │ 3 x 3 x 1024           │ 14 x 14 x 1024    │
│ Conv 22    │ 3 x 3 x 1024, stride=2 │ 7 x 7 x 1024      │
│ Conv 23    │ 3 x 3 x 1024           │ 7 x 7 x 1024      │
│ Conv 24    │ 3 x 3 x 1024           │ 7 x 7 x 1024      │
│ FC 1       │ -                      │ 4096              │
│ FC 2       │ -                      │ 7 x 7 x 30 (1470) │
└────────────┴────────────────────────┴───────────────────┘

``` bash
boxes, scores = self.sess.run([self.boxes, self.scores], 
                               feed_dict={self.inputs: inputs, self.ratio: ratio})

mask = scores >= _SCORE_THRESHOLD
boxes_ = []
scores_ = []
classes_ = []
for Class in range(len(self.class_names)):          
            
    cls_boxes = boxes[np.array(mask[:, Class]), :]   
    cls_scores = scores[np.array(mask[:, Class]), Class]
    while cls_boxes.shape[0] != 0:
        cls_boxes, cls_scores, max_box, max_score = NMS(cls_boxes, cls_scores, _IOU_THRESHOLD)             
        boxes_.append(max_box)                
        scores_.append(max_score)
        classes_.append(np.ones_like(max_score, dtype=int) * Class)       
```

The outputs from the yoloV3 model are `boxes` and `scores` with shape [10647, 4] and [10647, 80].
(10647 = the number of grid cells * the number of anchor boxes)

The network predicts 4 coordinates(bx, by, bw, bh) for each bounding boxes with 3 scale(13, 26, 52) and 80 class predictions in COCO dataset.

`mask` use to divide `score` into positive and negative with `_SCORE_THRESHOLD = 0.5` then use the `NMS` by computing `IOU` to choose the correct bounding boxes and classification.

- NMS
``` bash
def NMS(cls_boxes, cls_scores, iou_threshold):
            
    max_idx = np.argmax(cls_scores, 0)  
    max_box = cls_boxes[max_idx]
    max_score = np.array(cls_scores[max_idx])
    max_mask = cls_boxes !=  max_box             
    max_mask = np.reshape(max_mask[:, 0:1], [-1])
    cls_boxes = cls_boxes[np.array(max_mask), :]
    cls_scores = cls_scores[np.array(max_mask)]
    ious = [IOU(max_box, x) for x in cls_boxes]
    iou_mask = np.array(ious) <  iou_threshold
    cls_boxes =  cls_boxes[iou_mask, :]
    cls_scores = cls_scores[iou_mask]               
    max_box = np.reshape(max_box, [1,-1])
    max_score = np.reshape(max_score, [-1])   
    return cls_boxes, cls_scores, max_box, max_score 
```



- IOU
``` bash
def IOU(box1, box2):

    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2
    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)
    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)
    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou
```
`box1`: ground-truth bounding boxes

`box2`: predicted bounding boxes

`iou`: area of overlap / area of union

<img src="./img/iou.jpg" width="900px/">
