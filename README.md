# A Machine Learning Framework for Supervised Distributional Lexical Relation Classification

### By Chengyu Wang (https://chywang.github.io)

**Introduction:** It is a general framework with easy-to-use Python APIs to build supervised distributional models for lexical relation classification. It includes several types of projection models (i.e., linear projection, orthogonal linear projection, piecewise linear projection and orthogonal piecewise linear projection) and word embedding based features (i.e., concatenation, subtraction, addition of two terms' embeddings). Besides, it can work in two modes: fully supervised and semi-supervised. Users can inject different types of projections and features combinations for different types of lexical relations.

**Frameworks**

+ Supervised: the algorithm is implemented in supervised.py, with a demo in the demo package

Inputs

1. model.vec: The embeddings of all words in word2vec format. The start of the first line is the number of words and the dimensionality of word embeddings. After that, each line contains the words and its embeddings. All the values in a line are separated by a blank (' '). In practice, the embeddings can be learned by all deep neural language models.

> NOTE: Due to the large size of neural language models, we do not upload the embedding vectors of words. Please use your own neural language model instead, if you would like to try the algorithm over your datasets.

2. train.tsv: The path of the training set. The format is "word1 \t word2 \t relation" triples.

3. test.tsv: The path of the testing set. The format is "word1 \t word2 \t relation" triples.

> We upload a subset of the BLESS dataset here.

Codes (A brief guide on how to use the demo)

1. Load the input of our algorithm.

```python
learner= RelationLearner()
learner.load_word_embeddings('model.vec')
learner.load_training_set('train.tsv')
learner.load_testing_set('test.tsv')
```
2. The algorithm learns how to map the relation subject to the relation object in the word embedding space. The first parameter is the relation label. The second is the type of projection that is applied to the type of lexical relations. There are four types of projections:

Input | Projection
:-: | :-: 
 'linear'  | Linear projection| 
 'orth_linear'  | Orthogonal linear projection| 
 'piecewise'  | Piecewise linear projection| 
 'orth_piecewise'  | Orthogonal piecewise linear projection| 

Refer to the papers for details. For the latter two types of projections, you will need to input the value of parameter `k`.

```python
learner.impose_project('attri','linear')
learner.impose_project('coord','linear')
learner.impose_project('event','orth_linear')
learner.impose_project('mero','orth_linear')
learner.impose_project('random','piecewise',k=4)
learner.impose_project('hyper','orth_piecewise',k=4)
```

3. The embeddings of the two terms can be also added as features. All possible types are:

Input | Features
:-: | :-: 
 'subject'  | The embeddings of the relation subject| 
 'object'  | The embeddings of the relation object| 
 'offset'  | The offset of the two terms' embeddings| 
 'addition'  | The sum of the two terms' embeddings| 
 'product'  | The element product of the two terms' embeddings| 

```python
learner.set_default_feature_types('subject','object','offset')
```

4. Finally, we train the model and report the performance over the test set.

```python
model=learner.train_model()
learner.test_model(model)
```
---

+ Semi-supervised: the algorithm is implemented in semi_supervised.py, with a demo in the demo package

Inputs: The same as the supervised version

Codes (A brief guide on how to use the demo)

1. We set the max number of iterations and the confidence threshold in the semi-supervised learning process.

```python
learner= SemiSupervisedRelationLearner(2,0.8)
```

2. The rest of the steps are the same as the supervised version.

---

**Dependencies**

The main Python packages that we use in this project include but are not limited to:

1. gensim: 2.0.0
2. numpy: 1.15.4
3. scikit-learn: 0.18.1

The codes can run properly under the packages of other versions as well.

**References**

This software is built based on some of our own work and others.

1. Fu et al. Learning Semantic Hierarchies via Word Embeddings. ACL 2014
2. Wang and He. Chinese Hypernym-Hyponym Extraction from User Generated Categories. COLING 2016
3. Wang et al. Transductive Non-linear Learning for Chinese Hypernym Prediction. ACL 2017
4. Wang et al. A Family of Fuzzy Orthogonal Projection Models for Monolingual and Cross-lingual Hypernymy Prediction. WWW 2019
5. Wang et al. SphereRE: Distinguishing Lexical Relations with Hyperspherical Relation Embeddings. ACL 2019

More research works can be found here: https://chywang.github.io.



