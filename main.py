from PIL import ImageFont, ImageDraw, Image
from config import _ANCHORS, _SCORE_THRESHOLD, _IOU_THRESHOLD, _SIZE, _CLASSNAME
from predictor import predict
from utils import read_classes, resize_image, NMS

import numpy as np
import tensorflow as tf
import cv2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_img', 'input.jpg', 'Input image')
tf.app.flags.DEFINE_string('output_img', 'result.jpg', 'Output image')

class YOLO(object):
    def __init__(self, sess):
        
        self.sess = sess
        self.class_names = read_classes(_CLASSNAME)
        self.anchors = np.array(_ANCHORS).reshape(-1, 2)
        self.score_threshold = _SCORE_THRESHOLD 
        self.iou_threshold = _IOU_THRESHOLD
        self.input_size = (_SIZE, _SIZE)    
        self.ratio = tf.placeholder(tf.float32, [2], name='ratio')  
        self.inputs, self.boxes, self.scores = self.init_opt()
        
    def init_opt(self):

        inputs = self.sess.graph.get_tensor_by_name("input_1_5:0")
        scale1 = self.sess.graph.get_tensor_by_name("output_0:0")
        scale2 = self.sess.graph.get_tensor_by_name("output_1:0")
        scale3 = self.sess.graph.get_tensor_by_name("output_2:0")
    

        boxes, scores = predict([scale1, scale2, scale3], self.anchors, len(self.class_names), self.ratio,
                                score_threshold=self.score_threshold, iou_threshold=self.iou_threshold)
     
        return inputs, boxes, scores

    def detect_image(self, image):
        
        image = Image.fromarray(image)
        self.sess.run(tf.global_variables_initializer())

        ratio = np.array([image.size[0]/_SIZE, image.size[1]/_SIZE])

        boxed_image, image_shape = resize_image(image, self.input_size)
        inputs = np.array(boxed_image, dtype='float32') / 255.
        inputs = np.expand_dims(inputs, 0)
        
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
            
        out_boxes = np.reshape(boxes_, [-1, 4])  
        out_scores = np.reshape(scores_, [-1])  
        out_classes = np.reshape(classes_, [-1])
        
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        
        colors = []
        cls = ''
        color = tuple(np.random.randint(0, 256, 3))
        for i in out_classes:
            if cls != i:
                color = tuple(np.random.randint(0, 256, 3))
                cls = i         
                colors.append(color)
            else:
                colors.append(color)
            
        font = ImageFont.truetype(font='./font/FiraMono-Medium.otf', size=np.floor(3e-2 * image.size[1] + 0.5).astype(np.int32))
        thickness = (image.size[0] + image.size[1]) // 500  # do day cua BB

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype(np.int32))
            left = max(0, np.floor(left + 0.5).astype(np.int32))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype(np.int32))
            right = min(image.size[0], np.floor(right + 0.5).astype(np.int32))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=colors[i])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[i])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        
        cv2.imwrite(FLAGS.output_img, np.array(image))


def main(self, argv=None):
    tf.reset_default_graph()
 
    output_graph_def = tf.GraphDef()
    output_graph_path = './yolov3.pb'
    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    sess = tf.Session()   
    YOLO(sess).detect_image(cv2.imread(FLAGS.input_img))
    sess.close()

if __name__ == '__main__':
    
    tf.app.run()

