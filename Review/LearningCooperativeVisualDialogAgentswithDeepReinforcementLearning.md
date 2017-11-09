– Summary:

This paper aims at building a framework where two virtual agents, Q-bot can ask question to A-bot to eventually guess the image A-bot has access to see. Instead of training both agents in supervised manner, authors use deep reinforcement learning to train them. In the end, they show that DRL-based agents outperform previous models.

– Listofpositivepoints/Strengths:

The paper proposes very delicate and nice structure for two agents. Question encoder and decoder can relate a sequence of words to a fixed length embedding. Combining the image embedding, A-bot can response the question properly. Both bots will encode the qeustion and answers since they can be treated as a single RL agent in high level. At last, Q-bot will hold another regression network to output a continous embedding of the image.

– Listofnegativepoints/Weaknesses:

1. Is the A-bot running CNN image embedding at every round? Why or why not? Is it better to only encode image at the very beginning and start as the very first hidden state h0 like image captioning? Or is it better to run CNN forward each round to gather enought information?

2. Is it possible to change competitive agents where A-bot doesn't want Q guess the correct image so that it will given wrong answers. This should be much harder problem.

3. Can A-bot and Q-bot share the fact embedding network? Can they share history encoder network?

4. How do authors synthesize toy examples? Are they image or just symbol encoded by values?

– Reflections

Why not use continuous language model? It is argued in paper that continuous variable will encode full imformation of the image Q-bot needs to know to guess the image. But isn't it the goal of the work to let A-bot convey enough information so that Q-bot can guess the image?
