# Slicer-TensorFlow
A slicer module to use Tensorflow models for image segmentation.

The module enables the user to :
 - install tensorflow into slicer
 - import personal tensorflow models saved as .h5 and .keras
 - use those models to segment images directly into slicer

 The model input size is detected by the module which performs systematic image resizing and rescaling.
 Only the model path is saved by the module.
