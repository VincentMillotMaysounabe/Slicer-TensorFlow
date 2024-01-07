# Slicer-TensorFlow
## 1. Description
A slicer module to use Tensorflow models for image segmentation.

The module enables the user to :
 - install tensorflow into slicer
 - import personal tensorflow models saved as .h5 and .keras
 - use those models to segment images directly into slicer

 The model input size is detected by the module which performs systematic image resizing and rescaling.
 Only the model path is saved by the module.

## 2. Installation
You can install the extension by downloading the repository and drag & drop the folder in slicer. The extension will be found at Segmentation\TFSegmentation.

## 3. Menu
The extension menu is divided into 5 parts.<br />
![extensionMenu](https://github.com/VincentMillotMaysounabe/Slicer-TensorFlow/assets/114880539/e6a991e0-1084-47a4-9321-16e04db945a2)

1 - Tensorflow version : checks if tensorflow is available on the device. If not, the user can click "install" to pip install tensorflow. When a tensorflow is discovered, the version is displayed and the install button is unenabled.<br /><br />
2 - Input Volume : the volume to be segmented<br /><br />
3 - TensorFlow model : the model to be used. It can be selected among a list of saved models that can be modified through the next part. The "input type" changes the type of process to be done with the model. 2D means a slice by slice segmentation (1 slice as input - 1 slice as output), 2.5D means N slice as input and 1 slice as output, 3D means all volume as input and all volume as output.<br /><br />
4 - Model list options : the tool to manage the model list. Models can be added to the list by clicking on '...', selecting a .h5 or .keras file, and click 'Add model'. The current selected model in (3) will be deleted from the list by clicking 'Delete model'. Models are saved by their path so no copy of them are made.<br /><br />
5 - Advanced menu : to use any pretrained segmentation model in an easy way, to preprocessing steps are made by the extension : resizing to the needed model's input size and rescaling datas into [0, 1] value range. Those options can be unenabled by the user but errors could occur.<br /><br />
6 - Apply Button : to start processing the selected volume.<br /><br />
