
– Summary:

This paper study a problem where an agent can conceive an image and a piece of
instruction at each time step, and predict an action according to its internal
policy. They use policy gradient, as well as reward shaping to avoid the reward
delay problem in general reinforcement learning. From their results, their
method outperforms other methods, such as DQN and REINFORCE.

– List of positive points/Strengths:

The task they are trying to solve is quite hard since the reward is not
available until the end of rollout. Based on original reward, it is hard to
train a good policy. In this paper, the authors propose a set of additional
rewards to boost policy training.

– List of negative points/Weaknesses:

1. Task: why instructions are necessary. Images, start state and end state seem
   enough to define a reinforcement learning problem.

2. Why use a stack of previous images, instead of using CNN-LSTM on top of each
   frame of iamges?

3. How stochastic policy is defined? Is it the probability after softmax layer?
   Why block policy independent from direction policy?

– Reflections

This paper gives an effective method to solve reward delayed reinforcement
learning problem
