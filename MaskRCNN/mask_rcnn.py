import numpy as np
import random
import time
import cv2
import os
# $ python mask_rcnn_video.py --input videos/slip_and_slide.mp4 \
# 	--output output/slip_and_slide_output.avi --mask-rcnn mask-rcnn-coco
mask_rcnn = "mask-rcnn-coco"
labelsPath = os.path.sep.join([mask_rcnn, "object_detection_classes_coco.txt"])