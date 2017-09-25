– Summary:

This paper mainly investigates the fully convolutional networks for dense
prediction tasks. The main contribution is the novel skip structure for finer
pixel-wise classification. They basically modify previous famous networks, such
as VGG net, GoogLeNet and AlexNet, replacing classification layers with fully
convolutional layers and fine tune based on published supervised learning model.
They accomplish state-of-the-art segmentation of PASCAL VOC segmentation task by
around 20% of mean IU improvement and five times inference speed.

– List of positive points/Strengths:

Fully convolutional layers mentioned in the paper could handle variable size of
images and get rid of patch-wise training. On the other hand, since fully
convolutional layers only care about locally overlapping patches and computation
could further accelerated. Authors emphasize that they could obtain up to five
times speed up.

Decapitating final classification layer and converting fully connected layers to
convolutional layers, and then appending upsampling/deconvolutional layers to
restore back to original resolution, could scoring high on certain standard
metrics. But the semantic segmentation result is far from satisfying. I like
their idea of skipping certain layers and concatenate lower layers to later
layers. Basically it will provide more fine details of original images. And high
layers will provide global semantic information. Combining these two kinds of
information could induce fine semantic segmentation.

They also publish their code, which I really appreciate because it will make
reproduction of their work much easier.

– List of negative points/Weaknesses:

I believe it will be much better if they could spare one paragraph in section IV
to articulate their network structure, although it may be too low level
explanation. Or they could provide some technical tips for understanding their
network structure. This is of vital importance since the base network they are
using have completely different structure.

– Reflections

I believe they could apply their work on video and some kind of dynamic images.
If we have better semantic segmentation for each frame, it would be easier for
tracking some objects in the video.
