# GPUDMM

The implementations of the GPUDMM topic models, as described in 2016 SIGIR paper:

**Topic Modeling for Short Texts with Auxiliary Word Embeddings.**
Haoran Wang, Chenliang Li, Zhiqian Zhang, Aixin Sun, Zongyang Ma.

## Description

This repository doesn't contain the preprocess steps. So if you want to use this code, you should prepare the data by yourself. 

Also this repository doesn't contain the metric code for classification and PMI score. The classification algorithm we used is `SVM` provided by [scikit](http://scikit-learn.org/stable/).

The data format describe here:
> docid \t category | content

> 0	business|manufacture manufacturers suppliers supplier china directory products taiwan manufacturer

***
Anonther file you should prepare is the `words' similarity` file. In our paper, we use the cosine similarity calculated on word embeddings. This can be prepared in advance.

## Parameter Explanation

`beta`: the hyper-parameter beta, and the alpha is calculated as 50/numTopic.

`similarityFileName`: the file of words' similarity

`weight`: the promotion of similar word

`threshold`: the threshold for filtering similar word set

`filterSize`: the filter size for filtering similar word set

`numIter`: the number of iteration for gibbs sampling progress

## Model Result Explanation
`*_pdz.txt`: the topic-level representation for each document. Every line is a topic distribution for one document. This is used for classification task.

`*_phi.txt`: the word-level representation for each topic. Every line is a word distribution for one topic. This is used for PMI Coherence task.

`*_words.txt`: word, wordid map information. This is used for PMI Coherence task.

The PMI Coherence should calculated in external corpus, such as Wikipedia for English or Baidu Baike for Chinese.

