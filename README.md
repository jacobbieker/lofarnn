# lofarnn
Second Master's Research Project for Leiden University, focused on attempting to use machine learning to identify radio sources and optical counterparts in LOFAR data

# Installation

The easiest way to install this package is with ```pip``` with ```pip install lofarnn```.

Otherwise, the lastest code can be built with ```pip install git+https://github.com//jacobbieker/lofarnn.git```

# Usage

The different PyTorch models and datasets can be easily imported from the ```lofarnn``` package. To preprocess LOFAR data into the correct format for either CNN or Detectron2 models, example code can be found under ```analysis``` folder.

# Models
PyTorch models used in the thesis are available here: https://drive.google.com/drive/folders/1lCFcQT7WRTiMxfd8jL2ReCoJrNAhj4BW?usp=sharing. The best performing model is the ```multi_cnn.pth``` model.
