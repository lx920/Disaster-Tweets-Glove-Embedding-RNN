# Disaster-Tweets-Glove-Embedding-RNN
This notebook aims to use GloVe Embedding and RNN network to predict Kaggle Competition, Disaster Tweets

# Introduction
This is an ongoing Kaggle Competition: https://www.kaggle.com/c/nlp-getting-started/overview \
The aim of this project is to accurately predict whether a tweet is truly related to a natural disaster or not

# Dataset Brief
The original data can be found here: https://www.kaggle.com/c/nlp-getting-started/data \
There are 7613 rows of training data\
Independent variable is unstructed text data, tweets. Varying from 1 to 140 word length \
Dependent variable is the 1: Disaster, or 0: non-Disaster.

# Main Packages
- Basic Manipulation: Pandas, Numpy
- Text processing: re, nltk, contractions
- Embedding: tensorflow.kera.preprocessing
- Model Training: tensorflow.keras.Sequential

# Steps of project
- 1: Data Preprocessing and Cleaning
- 2: Glove Vector Embedding
- 3: RNN network model training

# Data Preprocessing and Cleaning
I want to remove all the garbage before doing word embedding
- RegExp to remove http, URL, signs. Also convert to lower case.
- Removal of stopwords
- WordNet Lemminization. Word contraction.

# GloVe Vector Embedding
In this section I want to create GloVe Embedding for later network\
GloVe vector is an open project by Stanford and the pre-trained Vector representations available for download can be found here: https://nlp.stanford.edu/projects/glove/ \
You can choose to use 50d, 100d, 200d or 300d vector reprensetation. I chose to use 200d for my project.\
Each line of the file starts with a word, and follows by the vector representation of it\
There are 400,000 English vocabulary stored in the file.\
After downloading and choosing 200d file:
- Create dictionary to store word:vector pairs.
- Use tensorflow.keras.preprocessing to tokenize the tweets. Select the top 120 tokens using pad sequence
- Use the word/vec dictionary and transform my corpus to a GloVe embedding matrix.

Now, I have created the first two layers for the RNN network
- 1: a 7613 X 120 matrix representation of original cleaned tweets
- 2: a n X 200 matrix representation of word embedding

# Model Building
Here I choose to use a sequential model with 3 hidden layers\
There may be other models or layers that can outperform this. But I have not tested\
- set up Adam optimizer. This is the most recommended for RNN. It is most accurate and least time consuming for RNN.
- set up RELU as activation functions. There maybe other activation functions that can perform better, though.
- set up the loss function as BinaryCrossEntropy. This is predicting 1 or 0 so it should be binary.
- set up monitors for accuracy and loss for each Epoch. So we can visualize at each training where the model is at
- set up threshold for early stopping. If 3 continuous Epoch do not improve loss. Then training can stop
- set up a validation split of 0.8:0.2 as train and test.

I tried different layers, dropout rates, as well as batch sizes. \
It seems the accuracy approximately max out at around 0.85. Loss minimize around 0.336\

# Conclusion and Results
- The final RNN network has 3 layers. The first two uses RELU activation. I chose to use sigmoid at the last layer, as it is a binary classification
- I chose a batch size of 100, with this size each Epoch will run on 61 iterations.

The Kaggle Competition Submission has accuracy score of 0.825

