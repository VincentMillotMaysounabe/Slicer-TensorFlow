# Examples using MRI prostate segmentation
The following steps shows how the extension works with a 2D U-net trained for prostate segmentation on axial T2 MRI images.

## 0. Needed material
To reproduce the following steps, you will need:
  -  axial T2 prostate MRI images (4 examples can be found in "Example <N>" folders)
  -  the pre-trained 2D U-net segmentation model (can be found in "model" folder)

You can download those files and unzip them to reproduce this tutorial of using the presented slicer extension.

## 1. Importation
After launching slicer, load the MRI images into the app as a Node. Make sure those are axial T2w images so the model can be used for what he is trained for.

## 2. First step into Slicer-TensorFlow extension
Launch the extension by finding it into Modules > Segmentation > TFSegmentation. After clicking, the module can take some time before appearing because he checks for his dependancies (mainly TensorFlow).

If it's your first time using the extension, TensorFlow will not be installed. In this case, you should see this information: <br />
<img width="349" alt="image" src="https://github.com/VincentMillotMaysounabe/Slicer-TensorFlow/assets/114880539/c656d5fe-f2a8-4e58-8698-3932d51a5075"><br />
You can click on "install" button to download and install the latest TensorFlow version. Once it is installed, the version should appear, and the install button will be disabled as shown bellow.<br />
<img width="347" alt="image" src="https://github.com/VincentMillotMaysounabe/Slicer-TensorFlow/assets/114880539/934cec38-fe55-4101-97aa-b34292c61999"><br />

Please note this step will not be required in further usage of this extension as TensorFlow will already be installed into Slicer.

## 3. Chosing the inputs
Input Volume panel should be used to select the prostate MRI to be segmented. If the volume is already loaded into slicer as a node, it should appear as shown bellow with example 1:<br />
<img width="434" alt="image" src="https://github.com/VincentMillotMaysounabe/Slicer-TensorFlow/assets/114880539/fe3c2427-a214-46d9-969c-9502d99e93bd"><br />

## 4. Addind model to list
Right after the extension installation, the model list is empty. To add a path to the model list, click on "..." button in the "Model list options" panel. Select the model to segment with (should be a .h5 or a .keras file). Here you can use the available model "ProstateSegmentation" that can be found in Examples\model folder.

One selected, the path should appear as shown bellow. Simply click on "Add model" to add the selected model to the list.<br />
<img width="437" alt="image" src="https://github.com/VincentMillotMaysounabe/Slicer-TensorFlow/assets/114880539/3ad282e9-762e-4a2e-ac22-74983ae1302d"><br />

Please note that in further usage the added model will be shown in the model list. Any model can be deleted from list by selecting it into the model list and click "Delete Model".

Choose the model "ProstateSegmentation.h5" in the model list to segment the prostate.

## 5. Launch segmentation
If every steps was completed, the extension state should correspond to the following image. <br />
<img width="289" alt="image" src="https://github.com/VincentMillotMaysounabe/Slicer-TensorFlow/assets/114880539/aa2bd275-35e9-4240-a385-cfd0b35946d2"><br />
To segment the prostate, press "Predict". A new segmentation node will appear after the computation, named "Model Segmentation", which contains the model segmentation. On example 1, the result should look like that :<br />
<img width="362" alt="image" src="https://github.com/VincentMillotMaysounabe/Slicer-TensorFlow/assets/114880539/fb701e7b-e59b-4f34-9f5a-5805f33fd171"><br />
