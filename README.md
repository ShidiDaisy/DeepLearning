# Deep Learning
Deep learning study notes

## Image Segmentation with Mask R-CNN
### [Mask R-CNN with OpenCV](https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/)

You can automatically segment and construct pixel-wise masks for every object in an image, thereby to segment the foreground object from the background. We just need perform <b>instance segmentation</b> using the Mask-RCNN architecture.

### Difference between semantic segmentation and instance segmentation
<b>Semantic Segmentation:</b> require us to associate every pixel in an input image with a class label (including background).
<b>Instance Segmentation:</b> compute a pixel-wise mask for every object in the image, even if the objects in the same class.

The Mask R-CNN we are using here were trained on the COCO dataset, which has L=90, thus the resulting volumn size from the mask module of the Mask R-CNN is 100*90*15*15

### Configuration
* mask-rcnn-coco/ : The Mask R-CNN model files
    * [frozen_inference_graph.pb](https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/): The Mask R-CNN model weights. The weights are pre-trained on the COCO dataset.
    * [mask_rcnn_inception_v2_coco_2018_01_28.pbtxt](https://www.google.com/search?client=safari&rls=en&q=mask_rcnn_inception_v2_coco_2018_01_28.pbtxt&ie=UTF-8&oe=UTF-8): The Mask R-CNN model configuration.
    * [object_detection_classes_coco.txt](https://github.com/opencv/opencv/blob/master/samples/data/dnn/object_detection_classes_coco.txt) : All 90 classes are listed in this text file, one per line. Open it in a text editor to see what objects our model can recognize.
    * [colors.txt](https://github.com/spmallick/learnopencv/blob/master/Mask-RCNN/colors.txt) : This text file contains colors to randomly assign to objects found in the image.
## RNN
1. Udemy: Deep Learning A-Z™: Hands-On Artificial Neural Networks

## Dataset
SuperDataScience Deep Learning A-Z™: Download Practice Datasets(https://www.superdatascience.com/pages/deep-learning)
