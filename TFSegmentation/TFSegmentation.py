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
import requests
from qt import QWidget, QLineEdit, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,QCursor,Qt
import re

def CheckForDependencies():
    try :
        from tensorflow.keras.models import load_model
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
        self._user_authentication = None

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
        self.ui.installButton.connect("clicked(bool)", self.onImportButton)
        self.ui.typeComboBox.currentIndexChanged.connect(self.onTypeChange)
        self.ui.modelComboBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.rescalingCheckBox.connect("clicked(bool)", self.onRescaleChange)
        self.ui.localComboBox.currentIndexChanged.connect(self.onLocalChange)
        #self.ui.modelComboBox.setNodeTypes('vtkMRMLTextNode')

        if self.userModels :
            self.ui.modelComboBox.enabled = True
            for model in self.userModels:
                self.ui.modelComboBox.addItem(model.split('/')[-1].strip('\n'))
        ########################################################################

        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        if self.logic.computation:
            try:
                TFversion = self.logic.getTFversion()
                self.ui.versionLabel.setText(TFversion + '  ')
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
        ##############################################################################################
        #Developper = VM, Description = Load selected TF model and call logic to perform segmentation#
        #Loading is in safe mode and compile model with substitute optimizer, loss and metrics to    #
        #use the model using the less informations possible                                          #
        ##############################################################################################

        # Compute output
        logging.info("Start Computing")
        with slicer.util.tryWithErrorDisplay(message='Prediction failed', show=True, waitCursor=True):
            #idealProcessingMethod = self.logic.getIdealProcessingMethod()
            ProcessingChoice = self.ui.typeComboBox.currentText

            if not self.logic.computation:
                self.onLocalChange()

            if self.logic.computation:
                self.logic.process(self.userModels[self.ui.modelComboBox.currentIndex],
                    self.ui.typeComboBox.currentText, self.ui.inputSelector.currentNode(),
                    self.ui.resizingCheckBox.isChecked(), self.ui.rescalingCheckBox.isChecked(),
                    self.ui.rescalingComboBox.currentText)

            ###############################################################################################
    def onAddButton(self):
        """
        Add a model to the model list
        """
        if self.ui.modelPathLineEdit.currentPath == '':
            slicer.util.errorDisplay("Selected model path is invalid")
            return

        # Check if path already exist in file
        with open(self.resourcePath('UserModels.txt')) as f:
            paths = f.readlines()
            if self.ui.modelPathLineEdit.currentPath in paths:
                return False

        # Add model to list
        self.logic.addModelPath(self.resourcePath('UserModels.txt'), self.ui.modelPathLineEdit.currentPath)
        # Add model to comboBox & set combobox to this item
        self.ui.modelComboBox.addItem(self.ui.modelPathLineEdit.currentPath.split('/')[-1].strip('\n'))
        self.ui.modelComboBox.setCurrentText(self.ui.modelPathLineEdit.currentPath.split('/')[-1].strip('\n'))
        # Add model to userModels parameter
        self.userModels.append(self.ui.modelPathLineEdit.currentPath)
        # inform user model was correctly added
        slicer.util.infoDisplay("The model was correctly added to list", windowTitle="Model added")

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

            # Inform user the model as beed deleted
            slicer.util.infoDisplay("The model was removed from list", windowTitle="Model removed")


    def onTypeChange(self):
        choice = self.ui.typeComboBox.currentText
        if choice == "2D":
            self.ui.typeDescLabel.setText("The segmentation will be processed slice by slice")
        if choice == "2.5D":
            self.ui.typeDescLabel.setText("The segmentation will be processed slice by slice using a stack of N images"+
                                          " as the model input")
        if choice == "3D":
            self.ui.typeDescLabel.setText("The segmentation will be processed with the full volume as input")

        if choice == "2.5D RGB":
            self.ui.typeDescLabel.setText("The segmentation will be processed slice by slice using a stack of 3 images"+
                                          " forming a RGB image as model input")

    def onRescaleChange(self):
        if self.ui.rescalingCheckBox.isChecked():
            self.ui.rescalingComboBox.setEnabled(True)
        else:
            self.ui.rescalingComboBox.setEnabled(False)

    def onLocalChange(self):
        choice = self.ui.localComboBox.currentText

        if choice.startswith("Local"):
            self.logic.setComputation("Local")
        elif choice.startswith("Distant"):
            if self._user_authentication:
                if not self._user_authentication.isAuthentified:
                    self._user_authentication.show()
            else:
                self._user_authentication = user_authentication()
            self.logic.setComputation("Distant")

        if self.logic:
            try:
                TFversion = self.logic.getTFversion()
                self.ui.versionLabel.setText(TFversion + '  ')
                self.ui.installButton.setEnabled(False)
            except:
                None

    def onImportButton(self):
        with slicer.util.tryWithErrorDisplay("Failed to connect to tensorflow.", waitCursor=True):
            try:
                self.onLocalChange()
            except:
                if self.ui.localComboBox.currentText.startswith("Local"):
                    self.logic.installTF()
            self.onLocalChange()

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
        self.computation = None

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

    def setComputation(self, type: str):
        if type=="Local":
            self.computation = TFComputation()
        elif type=="Distant":
            self.computation = TFSegmentationServ()


    def addModelPath(self, file_path: str, path: str):
        """
        Add a path to the model list in file_path.
        :param file_path: path of txt file where the model list is stored.
        :param path: path to be added.
        """

        # append path to file
        with open(file_path, 'a') as f:
            f.write('\n' + path)

    def removeModelPath(self, file_path: str, model: str):
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

    def installTF(self):
        import pip
        if hasattr(pip, 'main'):
            pip.main(['install', 'tensorflow'])
        else:
            pip._internal.main(['install', 'tensorflow'])
        with slicer.util.tryWithErrorDisplay("Failed to install TensorFlow.", waitCursor=True):
            import tensorflow as tf
            self.ui.versionLabel.setText(tf.__version__+'  ')
            slicer.util.infoDisplay("TensorFlow was succesfully installed", windowTitle="TensorFlow installation")

# ----------------------------------------------------------------------------
# --------------------------------Computation---------------------------------
# ----------------------------------------------------------------------------
    def getTFversion(self):
        if self.computation:
            version = self.computation.getTFversion()
        else:
            version = "TF not reached"
        return version


    def process(self, model_path, method: str, inputVolume, autoResize=True, autoRescale=True, rescaleScale=None):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param model: tensorFlow model to segment input volume with
        """

        ######################################################################
        #Developper = VM Description = Processing input volume with TF model #
        ######################################################################

        # making sure inputs are correct
        if not inputVolume:
            raise ValueError("Input volume is invalid")

        # Getting data array
        name = inputVolume.GetName()
        InputVolumeAsArray = slicer.util.array(name)

        #------------------------Tensorflow needed--------------------------
        result = self.computation.compute(model_path, method, InputVolumeAsArray, autoResize, autoRescale, rescaleScale)
        #----------------------------------------------------------------------

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

        logging.info("Process ended successfully")

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

class user_authentication(QWidget):
    def __init__(self):
        super().__init__()
        self.createUi()

        self.password = None
        self.mail = None
        self.isAuthentified = False

        self.show()

    def createUi(self, display_confirm_password: bool = False):
        self.setWindowTitle("Login to Slicer-TensorFlow servor")
        self.setGeometry(200, 200, 300, 150)

        # Email
        self.email_label = QLabel("Email :   ")
        self.email_edit = QLineEdit()
        self.email_edit.setMaximumWidth(200)

        # Password
        self.password_label = QLabel("Password :   ")
        self.password_edit = QLineEdit()
        self.password_edit.setMaximumWidth(200)
        self.password_edit.setEchoMode(QLineEdit.Password)

        #Confirm Password
        self.confirm_password_label = QLabel("Confirm\npassword :   ")
        self.confirm_password_edit = QLineEdit()
        self.confirm_password_edit.setMaximumWidth(200)
        self.confirm_password_edit.setEchoMode(QLineEdit.Password)

        # Buttons
        self.login_button = QPushButton("Log In")
        self.login_button.setCursor(QCursor(Qt.PointingHandCursor))
        if display_confirm_password:
            self.login_button.setFlat(True)
        self.forgot_password_button = QPushButton("Forgot password ?")
        if display_confirm_password:
            self.forgot_password_button.setText("Cancel")
        self.forgot_password_button.setFlat(True)
        self.forgot_password_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.create_account_button = QPushButton("Create account")
        if not display_confirm_password:
            self.create_account_button.setFlat(True)
        self.create_account_button.setCursor(QCursor(Qt.PointingHandCursor))

        # Information display
        self.info_label = QLabel("welcome !")
        self.info_label.setStyleSheet("font-style: italic;")
        self.info_label.setMaximumHeight(10)

        email_layout = QHBoxLayout()
        email_layout.addWidget(self.email_label)
        email_layout.addWidget(self.email_edit)
        email_layout.setAlignment(self.email_label, Qt.AlignRight)

        password_layout = QHBoxLayout()
        password_layout.addWidget(self.password_label)
        password_layout.addWidget(self.password_edit)
        password_layout.setAlignment(self.password_label, Qt.AlignRight)

        confirm_password_layout = QHBoxLayout()
        confirm_password_layout.addWidget(self.confirm_password_label)
        confirm_password_layout.addWidget(self.confirm_password_edit)
        confirm_password_layout.setAlignment(self.confirm_password_label, Qt.AlignRight)

        info_layout = QHBoxLayout()
        info_layout.addWidget(self.info_label)
        info_layout.setAlignment(self.info_label, Qt.AlignRight)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.create_account_button)
        button_layout.addWidget(self.forgot_password_button)
        button_layout.addWidget(self.login_button)

        if not self.layout():
            main_layout = QVBoxLayout()
            main_layout.addLayout(email_layout)
            main_layout.addLayout(password_layout)
            if display_confirm_password:
                main_layout.addLayout(confirm_password_layout)
            main_layout.addLayout(info_layout)
            main_layout.addLayout(button_layout)
            self.setLayout(main_layout)
        else:
            self.deleteItemsOfLayout(self.layout())
            self.layout().addLayout(email_layout)
            self.layout().addLayout(password_layout)
            if display_confirm_password:
                self.layout().addLayout(confirm_password_layout)
            self.layout().addLayout(info_layout)
            self.layout().addLayout(button_layout)

        # Connections
        self.login_button.connect("clicked(bool)", self.LogIn)
        if not display_confirm_password:
            self.create_account_button.connect("clicked(bool)", self.diplayCreateAccount)
        else:
            self.create_account_button.connect("clicked(bool)", self.createAccount)
        if not display_confirm_password:
            self.forgot_password_button.connect("clicked(bool)", self.passwordForgot)
        else:
            self.forgot_password_button.connect("clicked(bool)", self.cancel_create_account)
        if display_confirm_password:
            self.email_edit.textChanged.connect(self.caracterChanged)
            self.password_edit.textChanged.connect(self.caracterChanged)
            self.confirm_password_edit.textChanged.connect(self.caracterChanged)
            self.create_account_button.setEnabled(False)
            self.login_button.setEnabled(False)

    def deleteItemsOfLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    self.deleteItemsOfLayout(item.layout())

    def cancel_create_account(self):
        self.createUi(display_confirm_password=False)

    def diplayCreateAccount(self):
        self.createUi(display_confirm_password=True)

    def createAccount(self):
        resp = TFSegmentationServ().create_account(mail=self.email_edit.text, password=self.password_edit.text)
        if resp.json()["is_user_added"]:
            self.createUi(display_confirm_password=False)
            self.info_label.setText("Account successfuly created, please Log In")
        else:
            self.info_label.setText("Unable to create account, mail already linked to an account")

    def caracterChanged(self):
        self.create_account_button.setEnabled(False)

        if not '@' in self.email_edit.text:
            self.info_label.setText("Email seems invalid")
            return

        is_strong, msg = self.check_password_strength(self.password_edit.text)
        if not is_strong:
            self.info_label.setText(msg)
            return

        if self.password_edit.text != self.confirm_password_edit.text:
            self.info_label.setText("Passwords are different")
            return

        self.info_label.setText("Requirements checked !")
        self.create_account_button.setEnabled(True)

    def check_password_strength(self, password: str) -> [bool, str]:
        """Vérifie la force d'un mot de passe."""
        # Vérifie si le mot de passe a au moins 8 caractères
        if len(password) < 8:
            return False, "Password must contain at least 8 characters " #"Le mot de passe doit contenir au moins 8 caractères."

        # Vérifie s'il y a au moins une lettre majuscule
        if not any(char.isupper() for char in password):
            return False, "Password must contain at least 1 capital letter" #"Le mot de passe doit contenir au moins une lettre majuscule."

        # Vérifie s'il y a au moins un caractère spécial
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False, "Password must contain at least 1 special character" # "Le mot de passe doit contenir au moins un caractère spécial."

        # Si toutes les conditions sont satisfaites, le mot de passe est considéré comme fort
        return True, "Password is strong" # "Le mot de passe est fort."

    def passwordForgot(self):
        ...

    def LogIn(self):
        resp = TFSegmentationServ().authenticate_user(mail=self.email_edit.text, password=self.password_edit.text)
        if resp.json()["is_authenticated"]:
            self.mail = self.email_edit.text
            self.password = self.email_edit.text
            self.hide()
        else :
            self.info_label.setText("Unable to Log In, please check your mail and password")


class TFComputation():
    def __init__(self):
        self.model = None

    def getTFversion(self):
        from tensorflow import __version__ as TFversion
        return TFversion

    def loadModel(self, modelPath):
        # import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.losses import BinaryCrossentropy

        # Load TF model
        logging.info('Loading model...')
        self.model = load_model(modelPath, compile=False)

        # compile model with substitute optimizer, loss and metrics.
        self.model.compile(optimizer='Adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

    def compute(self, model_path, method, InputVolumeAsArray, autoResize, autoRescale, rescaleScale):
        self.loadModel(model_path)
        InputVolumeShape = InputVolumeAsArray.shape
        InputModelShape = self.model.inputs[0].shape.as_list()[1:]  # first dim is None

        # Pre-processing
        preprocessedInputArray = self.preProcessing(InputVolumeAsArray, InputModelShape, autoResize, autoRescale,
                                                    rescaleScale)
        inputs = preprocessedInputArray
        if method=="2.5D RGB":
            # Converting Grayscale to RVB
            inputs = np.zeros((len(preprocessedInputArray), InputModelShape[0], InputModelShape[1], 3))
            for j in range(len(preprocessedInputArray)):
                inputs[j, :, :, 1] = np.squeeze(preprocessedInputArray[j])
                if j != 0:
                    inputs[j, :, :, 0] = np.squeeze(preprocessedInputArray[j - 1])
                if j != (len(preprocessedInputArray) - 1):
                    inputs[j, :, :, 2] = np.squeeze(preprocessedInputArray[j + 1])

        # Process data using model
        result = self.model.predict(inputs)

        # Making sure the output as the same shape as input
        from tensorflow.keras.layers import Resizing

        resizeOutputProcess = Resizing(InputVolumeShape[1], InputVolumeShape[2])
        resultResized = []
        for img in result:
            resultResized.append(resizeOutputProcess(img))
        result = np.array(resultResized)

        return result


    def preProcessing(self, InputVolumeAsArray, model_2D_input_size,  autoResize=True, autoRescale=True, rescaleScale=None):
        # Pre-processing
        from tensorflow.keras.layers import Resizing
        resizeProcess = Resizing(model_2D_input_size[0], model_2D_input_size[1])
        preprocessedInputArray = []
        for img in InputVolumeAsArray:
            expandedImg = np.expand_dims(img, axis=2)

            # Resizing array if selected
            if autoResize: resizedImg = resizeProcess(expandedImg)
            else: resizedImg = expandedImg

            # Rescaling values if selected
            if autoRescale: rescaledImg = self.autoRescaleImg(resizedImg, rescaleScale)
            else: rescaledImg = resizedImg

            preprocessedInputArray.append(rescaledImg)
        preprocessedInputArray = np.array(preprocessedInputArray)

        return preprocessedInputArray


    def autoRescaleImg(self, input_img, scale):
        import tensorflow as tf
        if scale is None or scale.startswith("[0,1]"):
            rescale = tf.keras.layers.Rescaling(1. / tf.reduce_max(input_img))
            return rescale(input_img)
        elif scale.startswith("[-1,1]"):
            rescale = tf.keras.layers.Rescaling(2. / tf.reduce_max(input_img), offset=-1)
            return rescale(input_img)


    def getIdealProcessingMethod(self):
        """
        returns a label containing the ideal processing method based on volume input shape and model input shape
        """
        InputModelShape = self.model.inputs[0].shape.as_list()[1:] #first dim is None
        OutputModelShape = self.model.outputs[0].shape.as_list()[1:] #first dim is None

        if len(InputModelShape) == 3:
            if InputModelShape[-1] == 3:
                return "2.5D RGB"
            else:
                return "2D"

        if len(InputModelShape) == 4:
            return "3D"

        if len(InputModelShape) == 4 and len(OutputModelShape) == 3:
            return "2.5D"

        return None

class TFSegmentationServ():
    def __init__(self):
        self.url = r"https://slicertensorflow.eu.pythonanywhere.com"

    def authenticate_user(self, mail:str, password:str) -> bool:
        user = {"mail": mail, "password": password}
        r = requests.post(self.url + '/auth/authenticate', json=user)
        return r

    def change_password(self, mail: str)->str:
        ...

    def create_account(self, mail: str, password: str)->bool:
        user = {"mail": mail, "password": password}
        r = requests.post(self.url + '/auth/adduser', json=user)
        return r



