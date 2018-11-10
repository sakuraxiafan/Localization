  # Localization
  This is an implementation of three-dimensional localization microscopy through deep learning on Matlab (Matconvnet) as well as Python (Keras and TensorFlow). The model is based on a deep convolutional neural networks (CNNs) to retrieve the 3D locations of the fluorophores from a single 2D image captured by a conventional wide-field fluorescence microscope. The challenging 3D localization is converted to a  multi-label classification problem through two cascaded CNNs.
  
  The repository includes:
  
  * Source code of training and testing for bead / cell / particle localization.
  * Source code of simulating training dataset.
  * Dataset of zebra fish blood moving collected by wide-field fluorescence microscopy for cell localization / tracking.
  * Examples of training and testing on simulated / collected dataset.
  
  The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below). 
  
  # Getting Started
  The dataset of zebra fish or system PSF can be obtained from the [Google Drive]('https://drive.google.com/xxx').
  
  For Python 3.5.5 user, please install [TensorFlow 1.4.0](https://github.com/tensorflow/tensorflow/tree/v1.4.0) as well as [Keras 2.1.3](https://github.com/keras-team/keras/tree/2.1.3). For Matlab user, please install [Matconvnet v1.0-beta24](https://github.com/vlfeat/matconvnet/tree/v1.0-beta24).
  
  ## Training cascaded CNNs
  `train_network_A.py(.m)` provides an example of training the first CNN (lateral detection CNN) to determine whether there exist diffraction patterns at the central lateral position of each patch.
  
  `train_network_B.py(.m)` provides an example of training the second CNN (axial localization CNN) to estimates the 3D positions of the predicted positive samples of lateral detection CNN.
  
  ## Testing well trained CNNs
  `test_network.py(.m)` provides an example of testing the well trained CNNs to achieve 3D localization given a fluorescence image captured by wide-field microscopy. Specifically, the testing codes in both bead and particle (SR) folders estimate the 3D locations given a simulated wide-field fluorescence image. The testing codes in cell folder estimates the 3D locations given the captured zebra fish blood motion movie.
  
  # Training on Your Own Dataset
  To train the model on your own dataset you'll need to change the PSF file utilized for simulating training samples.
  
  To test the well trained model you'll need to change the image / video file utilized for testing.
  
  ## Citation
  Use this bibtex to cite this repository:
```
@misc{xxx,
  title={},
  author={},
}
```
  
