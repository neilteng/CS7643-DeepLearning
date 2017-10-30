– Summary:

This paper introduces an end-to-end framework to generate caption for natural images. They stack LSTM unit on CNN image embedder. Training the whole network via stochastic gradient descent, the authors find the metrics of inference on test data set is beyond the state of art results. They further discuss generalization ability, transfer learning and word embedding learning.

– List of positive points/Strengths:

The authors gain insight from machine translation where LSTM network has been trained to predict the probability of target sentence given the whole source sentence. Here they treat the image embedding from CNN part as a special word embedding to LSTM part. After time step 0, they will feed in word sequence and output the maximum log probability of next word. The training process is pretty routine. It is found that their trained model can beat the state of art image captioning result.

They don't use bag of words to embed sentence, which makes sense since bag of words embedding, although sometimes effective and easy to implement, loss a lot of spatial information.

– List of negative points/Weaknesses:

The authors use image embedding as a kind of word embedding into LSTM and treat them as x_0. It would be better if they could report the result if they treat image embedding as initial hidden state h0 into LSTM.

The next step they could do is to try to stack more LSTMs or use bidirectional LSTM to see whether the image captioning can be further improved.

– Reflections

It seems necessary to load pre-trained model from large task as inital point for any new task to reduce over-fitting as indicated by the authors.
