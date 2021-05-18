# Disaster-Tweets-Glove-Embedding-RNN
This notebook aims to use GloVe Embedding and RNN network to predict Kaggle Competition, Disaster Tweets

# Steps of project
- 1: Data Preprocessing and Cleaning
- 2: Glove Vector Embedding
- 3: RNN network Hyperparameter Tuning
- 4: RNN network model training

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
- Hyperparameter Tuning: keras tuner
- Model Training: tensorflow.keras


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

# Hypeparameter Tuning
I used keras tuner and Hyperband search method to find:
- Optimal elements in the first dense layer
- Optimal learning rate of Adam optimizer

The element range is set for 32 to 512, the learning rate range is set for [1e-2, 1e-3, 1e-4].\
Set up objective as val_accuracy, max_epoch, early stopping and validation split\
After training, Hyperband search gives:
- The optimal units is 288 and learning rate 0.0001

# Model Building/Training
- The model incoporate the best Hyperparameters from the previous section.
- The model has input layer specified as the GloVe embedding
- A layer of bidirectional LSTM
- Layer with best Dense elements, Relu activation function
- Layer with 1 unit and sigmoid activation function. As this is a binary classification
- loss function as binary cross entrophy. Adam optimizer with 0.0001 learning rate.

In 15 Epoch of training, the loss function minimizes around 0.336, the val accuracy approches 0.8201.

# Conclusion and Results
- The final RNN network has 2 layers. The firstuses RELU activation. I chose to use sigmoid at the last layer, as it is a binary classification
- I chose a batch size of 100, with this size each Epoch will run on 61 iterations.

The Kaggle Competition Submission has accuracy score of 0.825

