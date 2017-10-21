# [CS 7643 Deep Learning - Homework 3][1]

In this (short) homework, we will implement vanilla recurrent neural networks (RNNs) and Long-Short Term Memory (LSTM) RNNs and apply them to image captioning on [COCO][3].

Note that this homework is adapted from [the Standford CS231n course][2].

Download the starter code [here]({{site.baseurl}}/assets/hw3_starter.zip).

## Setup

Assuming you already have homework 2 dependencies installed, here is some prep work you need to do. First, download the data:

```
cd cs231n/datasets
./get_assignment_data.sh
```

## Part 1: Captioning with Vanilla RNNs (25 points)

Open the `RNN_Captioning.ipynb` Jupyter notebook, which will walk you through implementing the forward and backward pass for a vanilla RNN, first 1) for a single timestep and then 2) for entire sequences of data. Code to check gradients has already been provided.

You will overfit a captioning model on a tiny dataset and implement sampling from the softmax distribution and visualize predictions on the training and validation sets.

## Part 2: Captioning with LSTMs (25 points)

Open the `LSTM_Captioning.ipynb` Jupyter notebook, which will walk you through the implementation of Long-Short Term Memory (LSTM) RNNs, and apply them to image captioning on MS-COCO.

## Part 3: Train a good captioning model (Extra credit: up to 15 points)

Using the pieces you implement in parts 1 and 2, train a captioning model that gives decent qualitative results (better than the random garbage you saw with the overfit models) when sampling on the validation set.

Code for evaluating models using the [BLEU][4] unigram precision metric has already been provided. Feel free to use PyTorch for this section if you'd like to train faster on a GPU.

Here are a few pointers:
- Attention-based captioning models
    + [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. Xu et al., 2015][5]
    + [Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning. Lu et al., CVPR 2017][6]
- Discriminative captioning
    + [Context-aware Captions from Context-agnostic Supervision. Vedantam et al., CVPR 2017][7]
- Novel object captioning
    + [Deep Compositional Captioning: Describing Novel Object Categories without Paired Training Data. Hendricks et al., CVPR 2016][8]
    + [Captioning Images with Diverse Objects. Venugopalan et al., CVPR 2017][9]

**Deliverables**

Submit the notebooks and code you wrote with all the generated outputs. Run `collect_submission.sh` to generate the zip file for submission.

References:

1. [CS231n Convolutional Neural Networks for Visual Recognition][2]

[1]: https://www.cc.gatech.edu/classes/AY2018/cs7643_fall/
[2]: http://cs231n.github.io/assignments2017/assignment3/
[3]: http://cocodataset.org/
[4]: http://www.aclweb.org/anthology/P02-1040.pdf
[5]: https://arxiv.org/abs/1502.03044
[6]: https://arxiv.org/abs/1612.01887
[7]: https://arxiv.org/abs/1701.02870
[8]: https://arxiv.org/abs/1511.05284
[9]: https://arxiv.org/abs/1606.07770
