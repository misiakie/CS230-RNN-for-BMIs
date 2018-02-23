# CS230-RNN-for-BMIs
CS230 project: Recurrent Neural Network for brain-machine interfaces: decoding arm kinematics from intracortical signal

*Authors: Julien Boussard, Maxime Cauchois and Theodor Misiakiewicz*

## Introduction

Brain-machine interfaces attempt to translate neural activity into control signals and create a direct pathway between the brain and an external device. In particular, BMIs give hope to disabled patients for restoring some of their mobility through neural prosthetics. The main challenge lies in reliably decoding neural activity, a highly non-linear and noisy time series data, into a particular stimulus. In this project, we build a recurrent neural network to decode the arm kinematic of a monkey from an intracortical array of 96 electrodes. The monkey were trained to move a cursor on a screen to reach targets. Our goal is to infer the instantaneous hand position and velocity using our RNN model and the neural signal as an input.

## Guide

