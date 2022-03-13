import cv2
import argparse
import datetime
from imutils.video import VideoStream
from utils import detector_utils as detector_utils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                default=1, help='Display the detected images using OpenCV. This reduces FPS')

ap.add_argument("-v", "--video" , required = True, help = "Path to the image")

args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    
    # This will return video from the first webcam on your computer.
    cap = cv2.VideoCapture(args['video'])  
    
    # loop runs if capturing has been initialized. 
    im_height, im_width = (None, None)
    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    
    score_thresh = 0.80

    while(True):
        # reads frames from a camera 
        # ret checks return at each frame
        ret, frame = cap.read() 
        
        frame = np.array(frame)
                
        if im_height == None:
            im_height, im_width = frame.shape[:2]  
            
        # Run image through tensorflow graph
        boxes, scores, classes = detector_utils.detect_objects(
            frame, detection_graph, sess)

        detector_utils.draw_box_on_image(
            score_thresh, scores, boxes, classes, im_width, im_height, frame)              
    
        cv2.imshow('Detection', frame)
        
        # Wait for 'q' key to stop the program 
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    
    # Close the window / Release webcam
    cap.release()
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows()
