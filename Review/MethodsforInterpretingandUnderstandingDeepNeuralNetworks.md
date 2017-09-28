– Summary:

This paper focuses on interpretation of deep neural networks model, relevance
and its propagation. It doesn't provide any prior tips for training a neural
networks since it is mainly about post hoc interpretation. The authors walk
through interpretation of DNN models, such as activation maximization, expert
and GAN coding, and decision making for DNN, such as sensitivity analysis and
relevance propagation. Later on, they introduce layer-wise relevance
propagation, which will induce deeper interpretation. In the end, they
introduce potential application of their work.

– Listofpositivepoints/Strengths:

When they are talking about interpretation of DNN model, they first go through
activation maximization, which is basically like regularizer. However, it is
based on the data instead of model weights. Then they introduce an expert
probability distribution, which can compulsively obtain strong classification
and reproduce original data distribution. Furthermore, they go to Generative
Adversarial Networks to regularize on GAN code, which render better
interpretation. Authors also investigate the relevance propagation through
networks. Based on that, they find use simple Talyor Decomposition and deep
Talyor Decomposition can somehow show how the data features influence final
classification result.

– Listofnegativepoints/Weaknesses:

This paper is good start for demystify deep learning framework. The methods and
tools proposed by authors are novel and kind of abstract. If they can give some
low level explanations, it would be much better for understanding.

– Reflections

Based on the idea of this paper, we should be less painful to debug our network
and maybe invent new networks structure, which can gender better results.
