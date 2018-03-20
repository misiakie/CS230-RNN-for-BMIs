# CS230-RNN-for-BMIs
CS230 project: Recurrent Neural Network for brain-machine interfaces: decoding arm kinematics from intracortical signal

*Authors: Julien Boussard, Maxime Cauchois and Theodor Misiakiewicz*

## Introduction

Brain-machine interfaces attempt to translate neural activity into control signals and create a direct pathway between the brain and an external device. In particular, BMIs give hope to disabled patients for restoring some of their mobility through neural prosthetics. The main challenge lies in reliably decoding neural activity, a highly non-linear and noisy time series data, into a particular stimulus. In this project, we build a recurrent neural network to decode the arm kinematic of a monkey from an intracortical array of 96 electrodes. The monkey were trained to move a cursor on a screen to reach targets. Our goal is to infer the instantaneous hand position and velocity using our RNN model and the neural signal as an input.



## Task
Given the premovement neural activity of 192 neurons monitored for a given period of time, we aim to predict the target attained by the animal in an experiment, among 48 targets divided into three concentric circles.


## Quickstart (~10 min)

1. __Build__ tfrecords files from the Matlab data recorded in MatlabAnimalData
```
python build_data.py
```

1Bis. __Data Already processed for tensorflow__ If your data is already under the format .tfrecords, you only need to put it in the folder `data`, and move on to step 2.

2. __Your first experiment__ We created a `base_model` directory for you under the `experiments` directory. It countains a file `params.json` which sets the parameters for the experiment. It looks like
```json
{
    "learning_rate": 1e-3,
    "batch_size": 5,
    "num_epochs": 2
}
```
For every new experiment, you will need to create a new directory under `experiments` with a `params.json` file.

3. __Train__ your experiment. If your data in in `data/my_folder`, simply run
```
python train.py --model_dir experiments/base_model --data_dir data/my_folder
```
It will instantiate a model and train it on the training set following the parameters specified in `params.json`. It will also evaluate some metrics on the development set, especially the accuracy, the 'Top 3' accuracy and so on.

If you want to reuse the weights from a past experiment you can use the `--restore_dir` option, precising the folder in which your weights have been saved (generally `experiments/my_model/my_weights`), for instance:
```
python train.py --model_dir experiments/base_model --data_dir data/my_folder --restore_dir experiments/base_model/best_weights
```
## Credits

The code architecture was taken from the "Named Entity Recognition with Tensorflow" project by Guillaume Genthial and Olivier Moindrot (see https://cs230-stanford.github.io).

