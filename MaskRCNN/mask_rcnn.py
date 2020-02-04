import numpy as np
import time
import cv2
import os
import imutils

if __name__ == '__main__':

    # $ python mask_rcnn_video.py --input videos/slip_and_slide.mp4 \
    # 	--output output/slip_and_slide_output.avi
    mask_rcnn = "mask-rcnn-coco"
    input = "video/HorsesEatingGrass.mp4"
    confidence_thre = 0.5
    threshold = 0.5 # minimum threshold for pixel-wise mask segmentation
    writer = None

    labelsPath = os.path.sep.join([mask_rcnn, "object_detection_classes_coco.txt"])
    colorsPath = os.path.sep.join([mask_rcnn, "colors.txt"])
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    weightPath = os.path.sep.join([mask_rcnn, "frozen_inference_graph.pb"])
    configPath = os.path.sep.join([mask_rcnn, "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

    net = cv2.dnn.readNetFromTensorflow(weightPath, configPath)
    vs = cv2.VideoCapture(input)

    # Get the total number of frames in the video files
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
    except:
        print("Could not determine number of frames in video")
        total = -1

    cur_frame = 0
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        # swapRB: swap the red and blue channel
        # return: 1. the bounding box coordinates. 2. the pixel-wise segmentation for each specific object
        blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
        net.setInput(blob) # set a new input for the network
        start = time.time()
        (boxes, masks) = net.forward(["detection_out_final", "detection_masks"]) # Runs forward pass to compute output of layer
        end = time.time()

        # loop detected objects
        for i in range(0, boxes.shape[2]):
            classID = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]
            if confidence > confidence_thre:
                (H, W) = frame.shape[:2]
                box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                boxW = endX - startX
                boxH = endY - startY
                mask = masks[i, classID]
                mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
                mask = (mask > threshold)
                roi = frame[startY:endY, startX:endX][mask] # extract roi

                color = COLORS[classID]
                blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
                frame[startY:endY, startX:endX][mask] = blended

                # draw bounding box
                color = [int(c) for c in color]
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                text = "{}:{:.4f}".format(LABELS[classID], confidence)
                cv2.putText(frame, text, (startX, startY-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(input.replace(".mp4", "-det.mp4"), fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        cur_frame+=1
        print("Current progress {}%".format(cur_frame/total * 100))
        writer.write(frame)

    print("[INFO] cleaning up...")
    writer.release()
    vs.release()