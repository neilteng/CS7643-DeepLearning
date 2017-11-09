informative, coheerince, and easy to answer
diversity and length, human judges

integrates the power of SEQ2SEQ systems to learn compositional semantic meanings of utterances with the strengths of reinforcement learn- ing in optimizing for long-term goals across a conver- sation.

Problem:
1 foruse on informative feedbacks that keep both user agents engaged.

2 infinte loop

Challenge
(1) integrate developer-defined rewards that better mimic the true goal of chatbot development and 

(2) model the long- term influence of a generated response in an ongoing dialogue.

Our model uses the encoder- decoder architecture as its backbone, 
and simulates conversation between two virtual agents to explore the space 
of possible actions while learning to maxi- mize expected reward.

rewards:
good conversations are forward-looking (Allwood et al., 1992) or 
interactive (a turn suggests a following turn), informative, 
and coheren

Optimiztion:
policy gradient instead of MLE

Pretrain using MLE model (? MLE vs Q learing) each action has q value. But MLE
will basially evalute the result reward of a sequence of actions. So MLE object
should larger than Q leanring objective. (or the action space dimension. The
actino space is infinite since we need arbitary length sequence.

Summation:
Action
State
  markvian, since we want to long term coherence.. Bescally we need all the
  history as state?
Policy
  random
Reward
  + easy to answer --- forward-looing function. simply min log prob of null
      response
  + information flow --- each agent must contribute new stuff each round. simply
      penlize of repeating previous utterance (cos similarity)
  + semantic coherence --- un- grammatical or not coherent
      Another RL network

Result

Refelction:

P_seq2seq is fixed why it is in reward

P_seq2seq bacward is another network? why not use bayeasian rule

scaledy by length. simple normlization or root? This make sense because we are
talking about log probaitliy

Using LSTM in DRL, is it markovian?

curriculum learning strategy?

n-gram
