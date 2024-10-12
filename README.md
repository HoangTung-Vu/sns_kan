This repository contains an implementation of the Kolmogorov-Arnold Network (KAN). The original KAN implementation can be found at: https://github.com/KindXiaoming/pykan.


## Simple KAN

`Simple KAN` is a streamlined version of the `pykan` library, designed to simplify the implementation of KAN (Kolmogorov Arnold Network) by removing the complexities of pruning, symbolic regression, and other advanced features. This makes it easier to mod and customize.

- **Layer Definition**: Easily define neural network layers, facilitating the creation of hybrid models.
- **Training Functions**: 
  - `train_pykan()`: Randomly selects batches from the training data and iterates through many random batches across multiple steps.
  - `fit()`: Allows you to specify the `batch_size`, divides the training set into batches, and trains through epochs by processing the entire training set.

`Simple KAN` provides a straightforward approach to building and training neural networks, focusing on essential functionalities for effective model training and modification.

## Wavelet KAN
The implementation for Wavelet-KAN is taken from https://github.com/zavareh1/Wav-KAN
