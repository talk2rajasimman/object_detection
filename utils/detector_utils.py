# Utilities for object detector.

import numpy as np
import tensorflow as tf
import cv2
from utils import label_map_util
detection_graph = tf.Graph()

TRAINED_MODEL_DIR = 'model'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR +  '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/label_map.pbtxt'

NUM_CLASSES = 90
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Load a frozen infrerence graph into memory
def load_inference_graph():

    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess


def draw_box_on_image(score_thresh, scores, boxes, classes, im_width, im_height, image_np):

    red = (0,0,255)
    green = (0,255,0)
    
    max_boxes_to_draw = 20
    
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        
        if classes[i] in category_index.keys():
            class_name = category_index[classes[i]]['name']            
        else:
            class_name = 'N/A'
            
        (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                        boxes[i][0] * im_height, boxes[i][2] * im_height)
        
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))
        
        if scores[i] > score_thresh:
            cv2.rectangle(image_np, p1, p2, green , 2, 1)            
            cv2.putText(image_np, class_name, (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,255), 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)            
                       
        else:
            cv2.rectangle(image_np, p1, p2, red , 2, 1)            
            cv2.putText(image_np, class_name, (int(left), int(top)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,255), 2)

            cv2.putText(image_np, 'confidence: '+str("{0:.2f}".format(scores[i])),
                        (int(left),int(top)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)            

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')
    images = []
    image = np.expand_dims(image_np, axis=0)
    
    images.append(image)

    image_np_expanded = np.concatenate(images, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)