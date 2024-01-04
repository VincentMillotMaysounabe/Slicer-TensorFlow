import logging
import os

import subprocess
import sys

import vtk
import ctk
import numpy as np
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

def CheckForDependencies():
    try :
        import tensorflow as tf
        return True
    except:
        return False

#
# TFSegmentation
#

class TFSegmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "TFSegmentation"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["Segmentation"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Vincent Millot Maysounabe (Personnal project)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is a slicer module to use Tensorflow models for image segmentation.
See more information in <a href="https://github.com/VincentMillotMaysounabe/Slicer-TensorFlow">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was developed by Vincent Millot as a personnal project to learn more about slicer 
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # TFSegmentation1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='TFSegmentation',
        sampleName='TFSegmentation1',
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, 'TFSegmentation1.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames='TFSegmentation1.nrrd',
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums='SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95',
        # This node name will be used when the data set is loaded
        nodeNames='TFSegmentation1'
    )

    # TFSegmentation2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category='TFSegmentation',
        sampleName='TFSegmentation2',
        thumbnailFileName=os.path.join(iconsPath, 'TFSegmentation2.png'),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames='TFSegmentation2.nrrd',
        checksums='SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97',
        # This node name will be used when the data set is loaded
        nodeNames='TFSegmentation2'
    )


#
# TFSegmentationWidget
#

class TFSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

        ########################################################################
        # Developer = VM # Description = Creates userModels variable, list of  #
        # path pointing on tensorFlow models imported by user                  #
        ########################################################################
        self.userModels = None
        # Load model names from local repository
        with open(self.resourcePath('UserModels.txt')) as f:
            self.userModels = f.readlines()

        ########################################################################

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/TFSegmentation.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = TFSegmentationLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        ########################################################################
        # Developer = VM # Description = Connection & init for model selector  #
        ########################################################################

        self.ui.addButton.connect("clicked(bool)", self.onAddButton)
        self.ui.deleteButton.connect("clicked(bool)", self.onDeleteButton)
        self.ui.installButton.connect("clicked(bool)", self.onInstallButton)
        self.ui.typeComboBox.currentIndexChanged.connect(self.onTypeChange)
        self.ui.modelComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        #self.ui.modelComboBox.setNodeTypes('vtkMRMLTextNode')

        if self.userModels :
            self.ui.modelComboBox.enabled = True
            for model in self.userModels:
                self.ui.modelComboBox.addItem(model.split('/')[-1].strip('\n'))
        ########################################################################

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        try:
            import tensorflow as tf
            self.ui.versionLabel.setText(tf.__version__+'  ')
            self.ui.installButton.setEnabled(False)
        except:
            None

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Set if Auto Rescale and Auto Resizingare to be made

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("InputVolume"):
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors
        self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))

        # Developer = VM Note = no update of model selector since only user needs to change it

        # Update buttons states and tooltips
        if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume")\
                and self._parameterNode.GetParameter("Model"):
            self.ui.applyButton.toolTip = "Compute output segmentation"
        else:
            self.ui.applyButton.toolTip = "Select model, input & output volume nodes"


        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID()
        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Run processing when user clicks "Apply" button.
        """
        with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
            ##############################################################################################
            #Developper = VM, Description = Load selected TF model and call logic to perform segmentation#
            #Loading is in safe mode and compile model with substitute optimizer, loss and metrics to    #
            #use the model using the less informations possible                                          #
            ##############################################################################################
            import tensorflow as tf

            # Load TF model
            logging.info('Loading model...')
            with slicer.util.tryWithErrorDisplay(message='Model loading failed', show=True, waitCursor=True):
                model = tf.keras.models.load_model(self.userModels[self.ui.modelComboBox.currentIndex], compile=False)

            # compile model with substitute optimizer, loss and metrics.
            model.compile(optimizer='Adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

            # Compute output
            logging.info("Start Computing")
            with slicer.util.tryWithErrorDisplay(message='Prediction failed', show=True, waitCursor=True):
                typeChoice = self.ui.typeComboBox.currentText
                if typeChoice == "2D":
                    self.logic.process(self.ui.inputSelector.currentNode(), model,
                                   self.ui.resizingCheckBox.isChecked(), self.ui.rescalingCheckBox.isChecked())
                if typeChoice == "2.5D":
                    None
                if typeChoice == "3D":
                    None

            ###############################################################################################
    def onAddButton(self):
        """
        Add a model to the model list
        """
        if self.ui.modelPathLineEdit.currentPath == '':
            print('Error : No model selected !')
            return

        # Check if path already exist in file
        with open(self.resourcePath('UserModels.txt')) as f:
            paths = f.readlines()
            if self.ui.modelPathLineEdit.currentPath in paths:
                return False

        # Add model to list
        self.logic.addModelPath(self.resourcePath('UserModels.txt'), self.ui.modelPathLineEdit.currentPath)
        # Add model to comboBox
        self.ui.modelComboBox.addItem(self.ui.modelPathLineEdit.currentPath.split('/')[-1].strip('\n'))
        # Add model to userModels parameter
        self.userModels.append(self.ui.modelPathLineEdit.currentPath)

    def onDeleteButton(self):
        if slicer.util.confirmOkCancelDisplay("Delete this model from list ?"):
            index = self.ui.modelComboBox.currentIndex
            model = self.userModels[self.ui.modelComboBox.currentIndex]
            # Delete model from txt file
            self.logic.removeModelPath(self.resourcePath('UserModels.txt'), model)

            # Delete model from self.userModels
            self.userModels.pop(index)

            # Delete model from widget
            self.ui.modelComboBox.removeItem(index)

    def onInstallButton(self):
        import pip
        if hasattr(pip, 'main'):
            pip.main(['install', 'tensorflow'])
        else:
            pip._internal.main(['install', 'tensorflow'])
        try :
            import tensorflow as tf
            self.ui.versionLabel.setText(tf.__version__+'  ')
        except :
            print("failed to install tensorflow")

    def onTypeChange(self):
        choice = self.ui.typeComboBox.currentText
        if choice == "2D":
            self.ui.typeDescLabel.setText("The segmentation will be processed slice by slice")
        if choice == "2.5D":
            self.ui.typeDescLabel.setText("The segmentation will be processed slice by slice using a stack of N images"+
                                          " as the model input")
        if choice == "3D":
            self.ui.typeDescLabel.setText("The segmentation will be processed with the full volume as input")

#
# TFSegmentationLogic
#

class TFSegmentationLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        # Pas de parametres pour le moment mais gare pour la semantique
        #if not parameterNode.GetParameter("Threshold"):
        #    parameterNode.SetParameter("Threshold", "100.0")
        #if not parameterNode.GetParameter("Invert"):
        #    parameterNode.SetParameter("Invert", "false")
        CheckForDependencies()



    def addModelPath(self, file_path, path):
        """
        Add a path to the model list in file_path.
        :param file_path: path of txt file where the model list is stored.
        :param path: path to be added.
        """

        # append path to file
        with open(file_path, 'a') as f:
            f.write('\n' + path)

    def removeModelPath(self, file_path, model):
        """
        remove a path from the model list
        :param model: name of the model to be removed.
        """

        # remove model from txt file
        with open(file_path, 'r') as f:
            model_list = f.readlines()
        with open(file_path, 'w') as f:
            for m in model_list:
                if not m.endswith(model):
                    f.write(m)
        None

    def process(self, inputVolume, model, autoResize=True, autoRescale=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param model: tensorFlow model to segment input volume with
        """

        ######################################################################
        #Developper = VM Description = Processing input volume with TF model #
        ######################################################################

        # VM Note = Should make sure the model is correct.
        if not inputVolume:
            raise ValueError("Input volume is invalid")

        if not model:
            raise ValueError("Model is invalid")

        model_input_shape = model.inputs[0].shape.as_list()[1:] #first dim is None

        import time
        startTime = time.time()
        logging.info('Processing started')

        # Getting data array
        name = inputVolume.GetName()
        InputVolumeAsArray = slicer.util.array(name)
        InputVolumeShape = InputVolumeAsArray.shape

        # Pre-processing
        import tensorflow as tf
        resizeProcess = tf.keras.layers.Resizing(model_input_shape[0], model_input_shape[1])
        preprocessedInputArray = []
        for img in InputVolumeAsArray:
            expandedImg = np.expand_dims(img, axis=2)

            # Resizing array if selected
            if autoResize: resizedImg = resizeProcess(expandedImg)
            else: resizedImg = expandedImg

            # Rescaling values if selected
            if autoRescale: rescaledImg = self.autoRescaleImg(resizedImg)
            else: rescaledImg =resizedImg

            preprocessedInputArray.append(rescaledImg)
        preprocessedInputArray = np.array(preprocessedInputArray)

        # Process data
        result = model.predict(preprocessedInputArray)

        # Making sure the output as the same shape as input
        resizeOutputProcess = tf.keras.layers.Resizing(InputVolumeShape[1], InputVolumeShape[2])
        resultResized = []
        for img in result:
            resultResized.append(resizeOutputProcess(img))
        result = np.array(resultResized)
        
        # making sure it is binary
        # Note for later : should take into account more than 2 labels
        result = np.squeeze(result, axis=3) #Always needed ? => ,3 images ?
        result = np.array(result > 0.5, dtype=float)

        # Create segmentation node
        # Note for later : segmentation node should be the selected one
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName(name+" segmentation")

        # Create new segment
        addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment("Model Segmentation")
        slicer.util.updateSegmentBinaryLabelmapFromArray(result, segmentationNode, "Model Segmentation", referenceVolumeNode = inputVolume)
        segmentationNode.SetDisplayVisibility(True)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')

    def autoRescaleImg(self, input_img):
        import tensorflow as tf
        rescale = tf.keras.layers.Rescaling(1. / tf.reduce_max(input_img))
        return rescale(input_img)

    def processSegmentTRANSElements(self, inputVolume):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """
        if not inputVolume :
            raise ValueError("Input or output volume is invalid")

        import time
        startTime = time.time()
        logging.info('Processing started')

        name = inputVolume.GetName()
        if name.startswith('TRANS'):
            # Getting data array
            arr = slicer.util.array(name)

            # Filtering for yellow
            bin_map = arr[:, :, :, 0] > arr[:, :, :, 2]  # extract all added elements
            bin_map = bin_map * (arr[:, :, :, 1] > 150)
            bin_map = bin_map * (arr[:, :, :, 1] > 150)
            bin_map = bin_map * (arr[:, :, :, 2] < 100)
            bin_map = np.array(bin_map, dtype=int)

        # Create segmentation node
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName(name+" segmentation")

        # Create new segment
        addedSegmentID = segmentationNode.GetSegmentation().AddEmptySegment("Firing Zone")
        slicer.util.updateSegmentBinaryLabelmapFromArray(bin_map, segmentationNode, "Firing Zone", referenceVolumeNode = inputVolume)
        segmentationNode.SetDisplayVisibility(True)

        stopTime = time.time()
        logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')


#
# TFSegmentationTest
#

class TFSegmentationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_TFSegmentation1()

    def test_TFSegmentation1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        registerSampleData()
        inputVolume = SampleData.downloadSample('TFSegmentation1')
        self.delayDisplay('Loaded test data set')

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = TFSegmentationLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
