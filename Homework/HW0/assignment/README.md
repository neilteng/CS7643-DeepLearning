### Part 1: Getting Started

In this course, we will be using python often (most assignments will need a good amount of python).

#### Anaconda

Although many distributions of python are available, we recommend that you use the [Anaconda Python](https://store.continuum.io/cshop/anaconda/). Here are the advantages of using Anaconda:

- Easy seamless install of [python packages](http://docs.continuum.io/anaconda/pkg-docs) (most come standard)
- It does not need root access to install new packages
- Supported by Linux, OS X and Windows
- Free!

We suggest that you use either Linux (preferably Ubuntu) or OS X.
Follow the instructions [here](http://docs.continuum.io/anaconda/install) to install Anaconda python.
Remember to make Anaconda python the default python on your computer.
Common issues are addressed here in the  [FAQ](http://docs.continuum.io/anaconda/faq).

#### Python
If you are comfortable with python, you can skip this section!

If you are new to python and have sufficient programming experience in using languages like C/C++, MATLAB, etc., you should be able to grasp the basic workings of python necessary for this course easily.

We will be using the [Numpy](http://www.numpy.org/) package extensively as it is the fundamental package for scientific computing providing support for array operations, linear algebra, etc. A good tutorial to get you started is [here](http://cs231n.github.io/python-numpy-tutorial/). For those comfortable with the operations of MATLAB, [this](https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html) might prove useful.

For some assignments, we will be using the [Jupyter Notebook](https://jupyter.org/).
Jupyter is a web app for interactive computing developed originally developed as part of the [IPython](https://ipython.org/) interactive shell.
The notebook is a useful environment where text can be embedded with code enabling us to set a flow while you do the assignments.
If you have installed Anaconda and made it your default python, you should be able to start the Jupyter notebook environment with:

```sh
jupyter notebook
```

Now you should be able to see the Jupyter home page when you navigate to `http://localhost:8888/` in your browser.
It shows you a listing of files in the directory you ran the `jupyter notebook` command from and allows you to create new notebooks.
Jupyter notebook files have `.ipynb` extensions.

This homework is a warm up for the rest of the course. As part of this homework you will:

- Implement Softmax Regression (SR)
    - vectorized loss function
    - vectorized gradient computation
- Implement Stochastic Gradient Descent

You will train the classifiers on images in the [CIFAR-10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html).
Download and unzip the [starter code](https://www.cc.gatech.edu/classes/AY2018/cs7643_fall/assets/hw0_q8_starter.zip) then start a Jupyter notebook in the resulting directory.
CIFAR-10 is a toy dataset with 60000 images of size 32 X 32, belonging to 10 classes.
You need to implement logistic regression in `softmax.ipynb`.

This homework is based on [assignment 1](http://cs231n.github.io/assignments2017/assignment1/) of the CS231n course at Stanford.

#### Getting the dataset

Make sure you are connected to the internet. Navigate to the `f17cs7643/data` folder and run the following:

```sh
./get_datasets.sh
```

This script will download the python version of the database for you and put it in `f17cs7643/data/cifar-10-batches-py` folder.

### Part 2: Softmax Regression

As you might already know, Softmax Regression is a generalization of logistic regression to multiple classes. Here is a brief summary and if you need a detailed tutorial to brush up your knowledge, [this](http://cs231n.github.io/linear-classify/) is a nice place.

Before we go into the details of a classifier, let us assume that our training dataset consists of \\(N\\) instances \\(x\_i \in \mathbb{R}^D \\) of dimensionality \\(D\\). 
Corresponding to each of the training instances,
we have labels \\(y\_i \in \{1,2,\dotsc ,K \}\\), where \\(K\\) is the number of classes. 
In this homework, we are using the CIFAR-10 database where \\(N=50,000\\), \\(K=10\\), \\(D= 32 \times 32 \times 3\\) 
(image of size  \\(32 \times 32\\) with \\(3\\) channels - Red, Green, and Blue).

Classification is the task of assigning a label to the input from a fixed set of categories or classes. A classifier consists of two important components:

**Score function:** This maps every instance \\(x_i\\) to a vector \\(z\_i\\) of dimensionality \\(K\\). Each of these entries represent the class scores for that image:

\\[ z\_i = Wx\_i + b \\]

Here, W is a matrix of weights of dimensionality \\(K \times D\\) and b is a vector of bias terms of dimensionality \\(K \times 1\\). The process of training is to find the appropriate values for W and b such that the score corresponding to the correct class is high. In order to do this, we need a function that evaluates the performance. Using this evaluation as feedback, the weights can be updated in the right 'direction' to improve the performance of the classifier.

Before proceeding, we'll incorporate the bias term into \\(W\\), making it of dimensionality \\(K \times (D+1)\\). Also let a superscript \\(j\\) denote the \\(j^{th}\\) element of \\(z\_i\\) and \\(w\_j\\) be the \\(j^{th}\\) row of W so that \\(z\_i^j = w\_j^Tx\_i\\). Finally apply the softmax function to compute probabilities (for the \\(i\\)th example and \\(j\\)th class):

\\[ p_i^j = \frac{e^{z\_i^{j}}}{\sum\_k e^{z^k\_i}} \\]

**Loss function:** This function quantifies the correspondence between the predicted scores and ground truth labels. Softmax regression uses the cross-entropy loss:

\\[ L = - \frac{1}{N}\sum\_{i=1}^{N}\log \left( p_i^{y_i} \right) \\]

If the weights are allowed to take values as high as possible, the model can overfit to the training data. To prevent this from happening a regularization term \\(R(W)\\) is added to the loss function. The regularization term is the squared some of the weight matrix \\(W\\). Mathematically,

\\[ R(W) = \sum\_{k}\sum\_{l}W\_{k,l}^2 \\]

The final loss is

\\[ \mathcal{L}(W) = L(W) + R(W) \\]

The regularization term \\(R(W)\\) is usually multiplied by the regularization strength \\(\lambda\\) before adding it to the loss function. \\(\lambda\\) is a hyper parameter which needs to be tuned so that the classifier generalizes well over the training set.

The next step is to update the weight parts such that the loss is minimized. This is done by Stochastic Gradient Descent (SGD). The weight update is done as:

\\[ W := W - \eta \nabla_W \mathcal{L}(W) \\]

Here, \\(\nabla_W \mathcal{L}\\) is the gradient of the loss function and the factor \\(\eta\\) is the learning rate. SGD is usually performed by computing the gradient w.r.t. a randomly selected batch from the training set.
This method is more efficient than computing the gradient w.r.t the whole training set before each update is performed.

**Deliverables**

Work through `softmax.ipynb` and implement the Softmax classifier. Run the `collect_submission.sh` script and upload the resulting zip file when done.

#### References

1. [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu)
