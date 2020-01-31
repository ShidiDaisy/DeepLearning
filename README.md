# Deep Learning
Deep learning study notes

## Mask R-CNN
### Mask R-CNN with OpenCV(https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/)

You can automatically segment and construct pixel-wise masks for every object in an image, thereby to segment the foreground object from the background. We just need perform <b>instance segmentation</b> using the Mask-RCNN architecture.

#### Difference between semantic segmentation and instance segmentation
<b>Semantic Segmentation:</b> require us to associate every pixel in an input image with a class label (including background).
<b>Instance Segmentation:</b> compute a pixel-wise mask for every object in the image, even if the objects in the same class.

The Mask R-CNN we are using here were trained on the COCO dataset, which has L=90, thus the resulting volumn size from the mask module of the Mask R-CNN is 100*90*15*15

## RNN
1. Udemy: Deep Learning A-Z™: Hands-On Artificial Neural Networks

## Dataset
SuperDataScience Deep Learning A-Z™: Download Practice Datasets(https://www.superdatascience.com/pages/deep-learning)
