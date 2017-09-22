# NOTE: The scaffolding code for this part of the assignment
# is adapted from https://github.com/pytorch/examples.
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from cifar10 import CIFAR10

# You should implement these (softmax.py, twolayernn.py, convnet.py)
import models.softmax
import models.twolayernn
import models.convnet
import models.mymodel

# Training settings
parser = argparse.ArgumentParser(description='CIFAR-10 Example')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M',
                    help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--model',
                    choices=['softmax', 'convnet', 'twolayernn', 'mymodel'],
                    help='which model to train/evaluate')
parser.add_argument('--hidden-dim', type=int,
                    help='number of hidden features/activations')
parser.add_argument('--kernel-size', type=int,
                    help='size of convolution kernels/filters')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of batches between logging train status')
parser.add_argument('--cifar10-dir', default='data',
                    help='directory that contains cifar-10-batches-py/ '
                         '(downloaded automatically if necessary)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load CIFAR10 using torch data paradigm
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# CIFAR10 meta data
n_classes = 10
im_size = (3, 32, 32)
# Subtract the mean color and divide by standard deviation. The mean image
# from part 1 of this homework was essentially a big gray blog, so
# subtracting the same color for all pixels doesn't make much difference.
# mean color of training images
cifar10_mean_color = [0.49131522, 0.48209435, 0.44646862]
# std dev of color across training images
cifar10_std_color = [0.01897398, 0.03039277, 0.03872553]
transform = transforms.Compose([
                 transforms.ToTensor(),
                 transforms.Normalize(cifar10_mean_color, cifar10_std_color),
            ])
# Datasets
train_dataset = CIFAR10(args.cifar10_dir, split='train', download=True,
                        transform=transform)
val_dataset = CIFAR10(args.cifar10_dir, split='val', download=True,
                        transform=transform)
test_dataset = CIFAR10(args.cifar10_dir, split='test', download=True,
                        transform=transform)
# DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                 batch_size=args.batch_size, shuffle=True, **kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset,
                 batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,
                 batch_size=args.batch_size, shuffle=True, **kwargs)

# Load the model
if args.model == 'softmax':
    model = models.softmax.Softmax(im_size, n_classes)
elif args.model == 'twolayernn':
    model = models.twolayernn.TwoLayerNN(im_size, args.hidden_dim, n_classes)
elif args.model == 'convnet':
    model = models.convnet.CNN(im_size, args.hidden_dim, args.kernel_size,
                               n_classes)
elif args.model == 'mymodel':
    model = models.mymodel.MyModel(im_size, n_classes)
else:
    raise Exception('Unknown model {}'.format(args.model))
# cross-entropy loss function
criterion = F.cross_entropy
if args.cuda:
    model.cuda()

#############################################################################
# TODO: Initialize an optimizer from the torch.optim package using the
# appropriate hyperparameters found in args. This only requires one line.
#############################################################################
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
def adjust_lr(optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // 1))
    print('[{}] learning rate is {}'.format(epoch,lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################

def train(epoch):
    '''
    Train the model for one epoch.
    '''
    # Some models use slightly different forward passes and train and test
    # time (e.g., any model with Dropout). This puts the model in train mode
    # (as opposed to eval mode) so it knows which one to use.
    model.train()
    # train loop
    for batch_idx, batch in enumerate(train_loader):
        # prepare data
        images, targets = Variable(batch[0]), Variable(batch[1])
        if args.cuda:
            images, targets = images.cuda(), targets.cuda()
        #############################################################################
        # TODO: Update the parameters in model using the optimizer from above.
        # This only requires a couple lines of code.
        #############################################################################
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, targets)
        loss.backward()
        #  for param in model.parameters():
        #      print(param.grad)
        optimizer.step()
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        if batch_idx % args.log_interval == 0:
            val_loss, val_acc = evaluate('val', n_batches=4)
            train_loss = loss.data[0]
            examples_this_epoch = batch_idx * len(images)
            epoch_progress = 100. * batch_idx / len(train_loader)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {}'.format(
                epoch, examples_this_epoch, len(train_loader.dataset),
                epoch_progress, train_loss, val_loss, val_acc))
            model.train()

def evaluate(split, verbose=False, n_batches=None):
    '''
    Compute loss on val or test data.
    '''
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    if split == 'val':
        loader = val_loader
    elif split == 'test':
        loader = test_loader
    for batch_i, batch in enumerate(loader):
        data, target = batch
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss += criterion(output, target, size_average=False).data[0]
        # predict the argmax of the log-probabilities
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_examples += pred.size(0)
        if n_batches and (batch_i >= n_batches):
            break

    loss /= n_examples
    acc = 100. * correct / n_examples
    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            split, loss, correct, n_examples, acc))
    return loss, acc


# train the model one epoch at a time
for epoch in range(1, args.epochs + 1):
    train(epoch)
    adjust_lr(optimizer, epoch)
evaluate('test', verbose=True)

# Save the model (architecture and weights)
torch.save(model, args.model + '.pt')
# Later you can call torch.load(file) to re-load the trained model into python
# See http://pytorch.org/docs/master/notes/serialization.html for more details

