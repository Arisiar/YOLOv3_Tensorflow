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

The outputs from the yoloV3 model are `boxes` and `scores` which have shape [10647, 4] and [10647, 80].

10647 = the numbers of grid ceils(3 scale: [13x13], [26x26], [52,52]) x the number of anchor boxes

The network predicts 4 coordinates for each bounding box and 80 class predictions in COCO.

``` bash
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
```box1```: ground-truth bounding boxes

```box2```: predicted bounding boxes

```iou```: area of overlap / area of union

<img src="./img/iou.jpg" width="900px/">
