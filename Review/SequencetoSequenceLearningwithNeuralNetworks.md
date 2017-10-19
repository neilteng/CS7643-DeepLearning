– Summary:

This paper introduced an end-to-end sequence to sequence network structure to address language translation problem. They use a stack of LSTM to embed the whole word sequences into one fixed length vector representation and later on use another stack of LSTM to decode sequence of information from aforementioned embedding. Their result is close to state-of-art language translation result. Furthermore they argue that by reversing the word order of original source languages can ultimately improve training results.

– List of positive points/Strengths:

The result and methodology of this paper is extremely impressive. It makes sense to separate sequence to sequence learning into two parts, encoding and decoding. They only keep the last hidden state as the embedded information. This vector will essentially cover all information from previous source sentences. Using another LSTM can decode information from this representation until end of sentence. This also helps when we plug in a third language interpreter/decoder. We can keep the first part. The dimesionality of this representation space can be treated as a hyper-parameter to denote the representation power of the whole model.

The biggest discovery is that reversing the word order of source language can improve training result remarkably. Authors also provide their insight of this phenomenon. While the overall lag distance between source word and target word is nearly the same, the minimum lag distance between the first few words is greatly reduced. Although the maximum lag distance between the last few words is also increased, we could imagine that first few words are more important for senetence evolution.

– List of negative points/Weaknesses:

The authors give too many abbreviations without explanation them. It would be better if they could provide such appendix.

Reversing the order is a novel contribution. They compare the results of reversing order and raw data. But they don't provide the results of both kinds of input, which from my perspective, could further improve the result, since the effective length of source sentences are halved. They should also be clear how they handle the EOS token for reverse order input.

– Reflections

The paper pushes the machine translation into seq2seq era. I am willing to see how this finding could be used in real life.



