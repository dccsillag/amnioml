import os
import vtk, qt, ctk, slicer
import logging
from SegmentEditorEffects import *
import sys
import requests
import json
import subprocess
import numpy as np
import time


class SegmentEditorEffect(AbstractScriptedSegmentEditorEffect):

    def __init__(self, scriptedEffect):
        scriptedEffect.name = 'AmnioML'
        scriptedEffect.perSegment = False # this effect operates on all segments at once (not on a single selected segment)
        scriptedEffect.requireSegments = True # this effect requires segment(s) existing in the segmentation
        AbstractScriptedSegmentEditorEffect.__init__(self, scriptedEffect)

    def clone(self):
        # It should not be necessary to modify this method
        import qSlicerSegmentationsEditorEffectsPythonQt as effects
        clonedEffect = effects.qSlicerSegmentEditorScriptedEffect(None)
        clonedEffect.setPythonSource(__file__.replace('\\','/'))
        return clonedEffect

    def icon(self):
        # It should not be necessary to modify this method
        iconPath = os.path.join(os.path.dirname(__file__), 'SegmentEditorEffect.png')
        if os.path.exists(iconPath):
            return qt.QIcon(iconPath)
        return qt.QIcon()

    def helpText(self):
        # the first <br> is used by Slicer to split the header from the body.
        return """<html>Segment amniotic fluid from a fetal MRI exam with machine learning.<br>
<p>Option:</p>
<ul style="margin: 0">
<li><b>Use CUDA:</b> enable CUDA (GPU) acceleration, if available. In most cases, the best option will be selected automatically.</li>
</ul>

<p>Buttons:</p>
<ul style="margin: 0">
<li><b>Run:</b> compute the segmentation, which may take a few minutes on slower systems.</li>
</ul>
</html>
"""

    def setupOptionsFrame(self):
        self.pluginDir = os.path.dirname(__file__)
        self.predictExecutableDir = os.path.abspath(self.pluginDir + "../../../predict")

        # useCUDA checkbox (try to guess best option)
        self.useCUDA = qt.QCheckBox("Use CUDA acceleration")
        self.useCUDA.setToolTip("Accelerate the segmentation using CUDA (GPU)")
        boolCudaIsAvailable = False
        self.pathCudaIsAvailablePreComputed = slicer.app.temporaryPath + "/" + "AmnioML-cudaIsAvailablePreComputed.txt"

        # try to read from cache first
        if os.path.exists(self.pathCudaIsAvailablePreComputed):
            with open(self.pathCudaIsAvailablePreComputed, "r") as f:
                if f.read()[:4] == "True":
                    boolCudaIsAvailable = True

        # if not in cache, check if cuda is available again
        else:
            subprocess_command = [f"{self.predictExecutableDir}/predict.exe", "--cuda_is_available"]
            print(subprocess_command)
            cudaIsAvailable = subprocess.check_output(subprocess_command)
            print(f"cudaIsAvailable[:4]={cudaIsAvailable[:4]}")
            boolCudaIsAvailable = cudaIsAvailable[:4] == b"True"
            with open(self.pathCudaIsAvailablePreComputed, "w") as f:
                if boolCudaIsAvailable:
                    f.write("True")
                else:
                    f.write("False")
            print("done running executable")
        self.useCUDA.setEnabled(True)
        if boolCudaIsAvailable:
            self.useCUDA.setChecked(True)
        else:
            self.useCUDA.setChecked(False)
        self.useCUDA.toggled.connect(self.onUseCudaToggled)
        self.scriptedEffect.addOptionsWidget(self.useCUDA)


        # Volume
        self.volumeLineEdit = qt.QLineEdit("N/A")
        self.volumeLineEdit.setReadOnly(True)
        self.volumeFrame = qt.QFrame() #self.scriptedEffect.addLabeledOptionsWidget("Segmented volume: ", self.volumeLineEdit)
        self.volumeFrame.setLayout(qt.QHBoxLayout())
        self.volumeFrame.layout().addWidget(qt.QLabel("Best estimate: "))
        self.volumeFrame.layout().addWidget(self.volumeLineEdit)

        self.confidenceIntervalLineEdit = qt.QLineEdit("N/A")
        self.confidenceIntervalLineEdit.setReadOnly(True)
        self.confidenceIntervalFrame = qt.QFrame() #self.scriptedEffect.addLabeledOptionsWidget("Segmented volume: ", self.volumeLineEdit)
        self.confidenceIntervalFrame.setLayout(qt.QHBoxLayout())
        self.confidenceIntervalFrame.layout().addWidget(qt.QLabel("Confidence interval (90%): "))
        self.confidenceIntervalFrame.layout().addWidget(self.confidenceIntervalLineEdit)

        self.volumeDataGroupBox = ctk.ctkCollapsibleGroupBox()
        self.volumeDataGroupBox.setTitle("Volume Data (Last Run)")
        self.volumeDataGroupBox.setLayout(qt.QVBoxLayout())
        self.volumeDataGroupBox.layout().addWidget(self.volumeFrame)
        self.volumeDataGroupBox.layout().addWidget(self.confidenceIntervalFrame)
        self.volumeDataGroupBox.collapsed = False
        self.scriptedEffect.addOptionsWidget(self.volumeDataGroupBox)


        # Run button
        self.applyButton = qt.QPushButton("Run")
        self.applyButton.objectName = self.__class__.__name__ + 'Run'
        self.applyButton.setToolTip("Run the prediciton algorithm")
        self.applyButton.connect('clicked()', self.onRun)


        # Buttons frame
        buttonsFrame = qt.QHBoxLayout()
        buttonsFrame.addWidget(self.applyButton)
        self.scriptedEffect.addOptionsWidget(buttonsFrame)



    def createCursor(self, widget):
        # Turn off effect-specific cursor for this effect
        return slicer.util.mainWindow().cursor


    def onUseCudaToggled(self):
        if os.path.exists(self.pathCudaIsAvailablePreComputed):
            os.remove(self.pathCudaIsAvailablePreComputed)

    def onRun(self):

        self.time_start_run = time.time()

        # Get list of visible segment IDs, as the effect ignores hidden segments.
        segmentationNode = self.scriptedEffect.parameterSetNode().GetSegmentationNode()
        visibleSegmentIds = vtk.vtkStringArray()
        segmentationNode.GetDisplayNode().GetVisibleSegmentIDs(visibleSegmentIds)
        if visibleSegmentIds.GetNumberOfValues() == 0:
            logging.info("Smoothing operation skipped: there are no visible segments")
            return

        # This can be a long operation - indicate it to the user
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

        # Allow users revert to this state by clicking Undo
        self.scriptedEffect.saveStateForUndo()

        # Export master image data to temporary new volume node.
        # Note: Although the original master volume node is already in the scene, we do not use it here,
        # because the master volume may have been resampled to match segmentation geometry.
        import vtkSegmentationCorePython as vtkSegmentationCore
        masterVolumeNode = slicer.vtkMRMLScalarVolumeNode()
        slicer.mrmlScene.AddNode(masterVolumeNode)
        masterVolumeNode.SetAndObserveTransformNodeID(segmentationNode.GetTransformNodeID())
        slicer.vtkSlicerSegmentationsModuleLogic.CopyOrientedImageDataToVolumeNode(self.scriptedEffect.masterVolumeImageData(), masterVolumeNode)
        spacing = masterVolumeNode.GetSpacing()
        self.scale = spacing[0] * spacing[1] * spacing[2] / 1000.0
        print("spacing="+str(masterVolumeNode.GetSpacing()))

        # Generate merged labelmap of all visible segments, as the filter expects a single labelmap with all the labels.
        mergedLabelmapNode = slicer.vtkMRMLLabelMapVolumeNode()
        slicer.mrmlScene.AddNode(mergedLabelmapNode)
        slicer.vtkSlicerSegmentationsModuleLogic.ExportSegmentsToLabelmapNode(segmentationNode, visibleSegmentIds, mergedLabelmapNode, masterVolumeNode)

        # Run segmentation algorithm
        import SimpleITK as sitk
        import sitkUtils
        # Read input data from Slicer into SimpleITK
        labelImage = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(mergedLabelmapNode.GetName()))
        backgroundImage = sitk.ReadImage(sitkUtils.GetSlicerITKReadWriteAddress(masterVolumeNode.GetName()))

        import uuid
        filename = str(uuid.uuid1())
        masterVolumeNode_filepath = slicer.app.temporaryPath + "/" + filename    + ".nrrd"
        labelVolumeNode_filepath = slicer.app.temporaryPath + "/" + filename    + "_label.nrrd"
        print("Saving masterVolumeNode to "+masterVolumeNode_filepath)
        slicer.util.saveNode(masterVolumeNode, masterVolumeNode_filepath)
        print("done saving")

        print("running executable")
        print(self.predictExecutableDir)
        gpu_flag = ""
        subprocess_command = [os.path.join(f"{self.predictExecutableDir}", "predict.exe") ,"-cp" , os.path.join(f"{self.predictExecutableDir}", "amnioml.ckpt"), "-ep" , f"{masterVolumeNode_filepath}","-o", f"{labelVolumeNode_filepath}"]
        if self.useCUDA.checkState():
            subprocess_command.append("--use_gpu")
        print(subprocess_command)
        print(subprocess.run(subprocess_command))
        print("done running executable")

        rawImage = sitk.ReadImage(labelVolumeNode_filepath)
        labelImage = rawImage > .5
        rawImageArray = sitk.GetArrayFromImage(rawImage)
        bestVolume = np.count_nonzero(rawImageArray > .5) * self.scale
        self.volumeLineEdit.setText('{:.2f}'.format(bestVolume)+" mL")


        with open(self.pluginDir+"/threshold_conformal_prediction-c=90.txt", "r") as f:
            min_threshold = float(f.readline())
            max_threshold = float(f.readline())

        minVolume = np.count_nonzero(rawImageArray > max_threshold) * self.scale
        maxVolume = np.count_nonzero(rawImageArray > min_threshold) * self.scale
        self.confidenceIntervalLineEdit.setText('{:.2f}'.format(minVolume)+" mL to "+"{:.2f}".format(maxVolume) + " mL")

        print(labelImage)
        sitk.WriteImage(labelImage, sitkUtils.GetSlicerITKReadWriteAddress(mergedLabelmapNode.GetName()))
        mergedLabelmapNode.GetImageData().Modified()
        mergedLabelmapNode.Modified()

        # Update segmentation from labelmap node and remove temporary nodes
        slicer.vtkSlicerSegmentationsModuleLogic.ImportLabelmapToSegmentationNode(mergedLabelmapNode, segmentationNode, visibleSegmentIds)
        slicer.mrmlScene.RemoveNode(masterVolumeNode)
        slicer.mrmlScene.RemoveNode(mergedLabelmapNode)

        os.remove(masterVolumeNode_filepath)
        os.remove(labelVolumeNode_filepath)

        qt.QApplication.restoreOverrideCursor()

        self.time_end_run = time.time()

        print(self.time_end_run-self.time_start_run)
