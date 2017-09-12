# [CS 7643 Deep Learning - Homework 1][5]

In this homework, we will learn how to implement backpropagation (or backprop) for 
“vanilla” neural networks (or Multi-Layer Perceptrons) and ConvNets.   
You will begin by writing the forward and backward
passes for different types of layers (including convolution and pooling),
and then go on to train a shallow ConvNet on the CIFAR-10 dataset in Python.   
Next you’ll learn to use [PyTorch][3], a popular open-source deep learning framework,
and use it to replicate the experiments from before.

This homework is divided into the following parts:

- Implement a neural network and train a ConvNet on CIFAR-10 in Python.
- Learn to use PyTorch and replicate previous experiments in PyTorch (2-layer NN, ConvNet on CIFAR-10).

Download the starter code [here]({{site.baseurel}}/assets/f17cs7643_hw1_starter.zip).

## Part 1

Starter code for part 1 of the homework is available in the `1_cs231n` folder.
Note that this is [assignment 2 from the Stanford CS231n course][1].

### Setup

Dependencies are listed in the `requirements.txt` file. If working with Anaconda, they should all be installed already.

Download data.

```bash
cd 1_cs231n/cs231n/datasets
./get_datasets.sh
```

Compile the Cython extension. From the `cs231n` directory, run the following.

```bash
python setup.py build_ext --inplace
```

### Q1.1: Two-layer Neural Network (10 points)

The IPython notebook `two_layer_net.ipynb` will walk you through implementing a
two-layer neural network on CIFAR-10. You will write a hard-coded 2-layer
neural network, implement its backward pass, and tune its hyperparameters.

### Q1.2: Modular Neural Network (16 points)

The IPython notebook `layers.ipynb` will walk you through a modular neural network
implementation. You will implement the forward and backward passes of many
different layer types, including convolution and pooling layers.

### Q1.3: ConvNet on CIFAR-10 (8 points)

The IPython notebook `convnet.ipynb` will walk you through the process of training
a (shallow) convolutional neural network on CIFAR-10.

**Deliverables**

Zip the completed ipython notebooks and relevant files.

```bash
cd 1_cs231n
./collect_submission.sh
```

Submit the generated zip file `1_cs231n.zip`.

## Part 2

This part is similar to the first part except that you will now be using [PyTorch][3] to 
implement the two-layer neural network and the convolutional neural network. In part 1
you implemented core operations given significant scaffolding code. In part 2 these core
operations are given by PyTorch and you simply need to figure out how to use them.

If you haven't already, install PyTorch (__please use PyTorch vesion >=0.2__). This will probably be as simple as running the
commands in the [Get Started][3] section of the PyTorch page, but if you run in to problems
check out the [installation section][10] of the github README, search Google, or come to
office hours. You may want to go through the [PyTorch Tutorial][12] before continuing.
This homework is not meant to provide a complete overview of Deep Learning framework
features or PyTorch features.

You probably found that your layer implementations in Python were much slower than
the optimized Cython version. Open-source frameworks are becoming more and more
optimized and provide even faster implementations. Most of them take advantage of
both GPUs, which can offer a significant speedup (e.g., 50x). A library of highly optimized Deep
Learning operations from Nvidia called the [CUDA® Deep Neural Network library (cuDNN)][9]
also helps.

You will be using existing layers and hence, this part should be short and simple. To get
started with PyTorch you could just jump in to the implementation below or read through
some of the documentation below.

- What is PyTorch and what distinguishes it from other DL libraries? (github [README][11])
- PyTorch [Variables](http://pytorch.org/docs/master/autograd.html#variable) (needed for autodiff)
- PyTorch [Modules](http://pytorch.org/docs/master/nn.html)
- PyTorch [examples][8]

The necessary files for this section are provided in the `2_pytorch` directory.
You will only need to write code in `train.py` and in each file in the `models/` directory.

### Q2.1: Softmax Classifier using PyTorch (6 points)

The`softmax-classifier.ipynb` notebook will walk you through implementing a softmax
classifier using PyTorch. Data loading and scaffolding for a train loop are provided.
In `filter-viz.ipynb` you will load the trained model and extract its weight so they can be visualized.

### Q2.2: Two-layer Neural Network using PyTorch (4 points)

By now, you have an idea of working with PyTorch and may proceed to implementing a two-layer neural network. Go to 
`models/twolayernn.py` and complete the `TwoLayerNN` `Module`. Now train the neural network using

```bash
run_twolayernn.sh
```
    
You will need to adjust hyperparameters in `run_twolayernn.sh` to achieve good performance.
Use the code from `softmax-classifier.ipynb` to generate a __loss vs iterations__ plot for train
and val and a __validation accuracy vs iterations__ plot. Make suitable modifications in `filter-viz.ipynb`
and save visualizations of the weights of the first hidden layer called `twolayernn_gridfilt.png`.

### Q2.3: ConvNet using PyTorch (6 points)

Repeat the above steps for a convnet. Model code is in `models/convnet.py`. Remember to save the filters learned. 

**Deliverables**

Submit the results by uploading a zip file called `2_pytorch.zip` created with

```bash
cd 2_pytorch/
./collect_submission.sh
```

The following files should be included:

1. Model implementations `models/*.py`
2. Training code `train.py`
3. All of the shell scripts used to train the 3 models (`run_softmax.sh`, `run_twolayernn.sh`, `run_convnet.sh`)
3. Learning curves (loss) and validation accuracy plots from Q2.2 and Q2.3.
4. The version of `filter-viz.ipynb` used to generate convnet filter visualizations
5. Images of the visualized filters for each model: `softmax_gridfilt.png`, `twolayernn_gridfilt.png`, and `convnet_gridfilt.png`
6. Log files for each model with test accuracy reported at the bottom


### Experiment (Extra credit: up to 10 points)

Experiment and try to get the best performance that you can on CIFAR-10 using a ConvNet.

- Filter size: In part 1 we used 7x7; this makes pretty pictures but smaller filters may be more efficient
- Number of filters: In part 1 we used 32 filters. Do more or fewer do better?
- Network depth: Some good architectures to try include:
    - [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]
    - [conv-relu-pool]xN - [affine]xM - [softmax or SVM]
    - [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]
- Alternative update steps: AdaGrad, AdaDelta, Adam

**Deliverables**

Be sure to include the following in the `2_pytorch.zip` file

- Model definition file `models/mymodel.py`
- Training log, loss plot, and validation accuracy plot as above
- List and describe all that you tried in a text file called `extra.md`

References:

1. [CS231n Convolutional Neural Networks for Visual Recognition][2]

[1]: http://cs231n.github.io/assignment2/
[2]: http://cs231n.stanford.edu/
[3]: http://pytorch.org/
[4]: http://bvlc.eecs.berkeley.edu/
[5]: https://www.cc.gatech.edu/classes/AY2018/cs7643_fall/
[8]: https://github.com/pytorch/examples
[9]: https://developer.nvidia.com/cudnn
[10]: https://github.com/pytorch/pytorch#installation
[11]: https://github.com/pytorch/pytorch
[12]: http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html


---

&#169; 2017 Georgia Tech
