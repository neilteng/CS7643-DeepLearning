import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        channels, height, width = im_size
        # 3, 32, 32
        self.conv1_1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 64, 16, 16
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # 128, 8, 8
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # 256, 4, 4
        # self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # # 512, 2, 2
        # self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        # # 512, 1, 1
        self.fc1 = nn.Linear(int(256*(height/2**3)*(width/2**3)), 4096)
        self.fc2 = nn.Linear(4096, 4096)
        # 4096
        self.fc3 = nn.Linear(4096, 10)

        self.activations = nn.LeakyReLU()
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        images = self.activations(self.conv1_1(images))
        images = self.activations(self.conv1_2(images))
        images = F.max_pool2d(images, (2,2))

        images = self.activations(self.conv2_1(images))
        images = self.activations(self.conv2_2(images))
        images = F.max_pool2d(images, (2,2))

        images = self.activations(self.conv3_1(images))
        images = self.activations(self.conv3_2(images))
        images = self.activations(self.conv3_3(images))
        images = F.max_pool2d(images, (2,2))

        # images = self.activations(self.conv4_1(images))
        # images = self.activations(self.conv4_2(images))
        # images = self.activations(self.conv4_3(images))
        # images = F.max_pool2d(images, (2,2))

        # images = self.activations(self.conv5_1(images))
        # images = self.activations(self.conv5_2(images))
        # images = self.activations(self.conv5_3(images))
        # images = F.max_pool2d(images, (2,2))

        images = images.view(images.size(0), -1)

        images = self.activations(self.fc1(images))
        images = F.dropout(images, training=self.training)
        images = self.activations(self.fc2(images))
        images = F.dropout(images, training=self.training)
        scores = self.fc3(images)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores
