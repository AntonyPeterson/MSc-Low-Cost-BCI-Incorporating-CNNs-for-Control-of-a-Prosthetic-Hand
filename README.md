# Low-Cost-BCI-Incorporating-CNNs-for-Control-of-a-Prosthetic-Hand
All data and code used for MSc dissertation titled "Low-cost Brain-Computer Interface Incorporating Convolutional Neural Netwokrs for Control of a Prosthetic Hand" by Antony Wade Peterson.

The study investigated using CNNs with minimal pre-processing to classify multiple MI and EMG tasks and enable live control of a 3D printed prosthetic hand to perform a variety of typical hand gestures.

All data and code used in this study is included in this repository and the instructions for it's use are detailed below. For ease of use, it is recommended that all code, data and results are saved in the same directory.

## Requirements
All code was developed in python 3.8.12 and the following libraries are required to be installed:

- [BrainFlow](https://brainflow.readthedocs.io/en/stable/BuildBrainFlow.html)	4.8.0

- [Jupyter notebook](https://jupyter.org/install) 6.1.4

- [Keras Tuner](https://keras.io/keras_tuner/)	1.1.0

- [Kivy](https://kivy.org/)	1.11.1

- [Matplotlib](https://matplotlib.org/stable/index.html) 3.3.1

- [MNE](https://mne.tools/stable/install/index.html)	0.24.0

- [Numpy](https://numpy.org/install/)	1.21.2

- [Pandas](https://pandas.pydata.org/getting_started.html)	1.1.3

- [pyFirmata](https://pypi.org/project/pyFirmata/)	1.1.0

- [Scikit-Learn](https://scikit-learn.org/stable/install.html)	0.23.2

- [SciPy](https://scipy.org/install/)	1.7.1

- [Seaborn](https://seaborn.pydata.org/installing.html) 0.11.0

- [TensorFlow](https://www.tensorflow.org/install)	2.5.0

If using a GPU:

- [TensorFlow GPU](https://www.tensorflow.org/install/pip) 2.5.0

- [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit-archive) 11.2

- [cuDNN](https://developer.nvidia.com/cudnn) 8.1.0

An environment manager such as [Anaconda](https://www.anaconda.com/) is recommended and newer versions of the libraries should be considered if there are no clashing requirements between the various libraries.

## EEG Data Acquisition

Prior to data acquisition, the Cyton board on the UltraCortex headset must be switched on and the USB dongle must be connected to the laptop.

The [OpenBCI GUI](https://docs.openbci.com/Software/OpenBCISoftware/GUIDocs/) was first used to monitor the quality of the EEG signals. Note for active electrodes, 
the electrode gains must be changed from x24 to x8 in the GUI. A custom python GUI was developed to aid data acquisition and the OpenBCI GUI session must be ended before using the python GUI.

The GUI programme is: [TestingGUI.py](Data%20Acquisition%20GUI/TestingGUI.py)

The task or tasks to be performed are first selected followed by the number of trials and their lengths and intervals. The "Participant:" field is used to specify the name of the folder where the data will be saved.
The GUI must be run in the same directory as the accompanying pictures of each MI or EMG task in [Data Acquisition GUI](Data%20Acquisition%20GUI) to function correctly.

## Datasets

Low-cost, self-gathered EEG data was gathered using an OpenBCI UltraCortex Mark IV headset with 8 active ThinkPulse electrodes positioned at FP1, FP2, C3, C4, CP1, CP2, PO3, PO4 and reference electrodes at FPZ and on the earlobe.
The active electrodes in that order correspond to channels 1-8 in the data respectively. Data was gathered for 4 classes of motor imagery (Foot, LH, RH, Tongue) and 4 EMG classes (Blinking, Close eyes, Eyebrow Raise and Teeth Clench). 
The dataset consists of 24 5-second trials of each class per day gathered over eight days scattered over three weeks. 
The first 6 days were used as the training set ([05to20-08 Active 8chan FPZ Bias](05to20-08%20Active%208chan%20FPZ%20Bias))
and the last 2 days as the validation set ([23&24-08 Active 8chan FPZ Bias](23%2624-08%20Active%208chan%20FPZ%20Bias)). 
The datasets comprise of raw 5-second-long EEG data.
The Close Eyes data was not used in the study. Data for an idle "Nothing" class gathered over the last 4 days is included but was not used in the study.

[BCI Competition IV Dataset 2a](BCI%20Competition%20IV%20Dataset%202a) [1] is a popular public dataset that contains 4 class motor imagery data from 9 subjects and was used to validate the CNN training and tuning method.
The folder contains the raw .mat training and validation data files for each subject.

## CNN Training and Tuning

CNN training and tuning for both datasets was performed using the methods in the notebooks provided in [CNN Training and Tuning](CNN%20Training%20and%20Tuning).
These are run in Jupyter Notebook.
The notebooks were run in the same directory as the datasets and were used to process and filter the EEG data and train and tune CNNs to classify MI or EMG tasks. The notebooks provided sometimes only demonstrate a single example of a classification scenario that was tested e.g. subject 9
for BCI Comp IV data or left and right hand MI classification for OpenBCI data. They can be simply modified as indicated to test a different classification scenario e.g. different subject or
combination of MI/EMG tasks.
The best default and tuned models obtained in the study are provided in the subfolder [Best Models](CNN%20Training%20and%20Tuning/Best%20Models).

## Band Power Analysis

The band powers for the electrodes closest to the motor cortex were analysed for the OpenBCI MI data using [Band Power Analysis of OpenBCI MI data.ipynb](Band%20Power%20Analysis%20of%20OpenBCI%20MI%20data.ipynb) which can be run in Jupyter Notebook.

## Prosthetic Hand Control

[Prosthetic Hand Control](Prosthetic%20Hand%20Control) includes the programmes developed for the two live control scenarios that were investigated:

[5classlivehandcontroltrials.py](Prosthetic%20Hand%20Control/5classlivehandcontroltrials.py) was used for the live control trials where the subject attempted to perform a selected gesture for 100 predictions. The recorded CNN predictions and their 
probabilities are included in [5class tuned rhtng 8chan fpz bias hand control trials](5class%20tuned%20rhtng%208chan%20fpz%20bias%20hand%20control%20trials).

[sequencehandlivecontrol.py](Prosthetic%20Hand%20Control/sequencehandlivecontrol.py) was used for the second control scenario where the subject attempted self-paced control of the hand to perform a set sequence of gestures. The predictions, probabilities and response times from the 5 trials 
performed are included in [5class tuned rhtng 8chan fpz bias hand control sequence](Prosthetic%20Hand%20Control/5class%20tuned%20rhtng%208chan%20fpz%20bias%20hand%20control%20sequence).

The [Live Prediction and Response Time Analysis.ipynb](Prosthetic%20Hand%20Control/Live%20Prediction%20and%20Response%20Time%20Analysis.ipynb) notebook contains code used to assess online classifcation accuracy during the live control trials and to analyse the prediction probabilities 
of the live predictions to assess whether incorrect predictions were made with low confidence. The system response time was also evaluated for the sequence trials. 


## References:

[1] C. Brunner, R. Leeb, G. Müller-Putz, A. Schlögl and G. Pfurtscheller, “BCI Competition 2008 - Graz data set A,” Institute for Knowledge Discovery (Laboratory of Brain-Computer Interfaces), Graz University of Technology, Austria, 2008.

