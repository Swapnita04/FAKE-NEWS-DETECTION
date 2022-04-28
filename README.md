
# Fake News Detection

## Overview
next-day newspaper. However, with the growth of online newspapers who update news almost instantly, people have found a better and faster way to be informed of the matter of his/her interest. Nowadays social-networking systems, online news portals, and other online media have become the main sources of news through which interesting and breaking news are shared at a rapid pace. However, many news portals serve special interest by feeding with distorted, partially correct, and sometimes imaginary news that is likely to attract the attention of a target group of people. Fake news has become a major concern for being destructive sometimes spreading confusion and deliberate disinformation among the people.

The Aims of this projects is to use the Natural Language Processing and Machine learning to detect the Fake news based on the text content of the Article.And after building the suitable Machine learning model to detect the fake/true news
.
# Prerequisites:
1. Python
2. You will also need to download and install the required packages after you install python
* Sklearn (scikit-learn)
* Numpy
* Pandas
* Matplotlib
* Seaborn
* NLTK
# Dataset:
The Dataset is collected from Kaggle (https://www.kaggle.com/)

# Text Preprocessing
Machine learning data only works with numerical features so we have to convert text data into numerical columns. So we have to preprocess the text and that is called natural language processing.
In-text preprocess we are cleaning our text by steaming, remove stopwords, remove special symbols and numbers.
 
# Bag Of Words
Bag of words is a Natural Language Processing technique of text modelling. In technical terms, we can say that it is a method of feature extraction with text data. This approach is a simple and flexible way of extracting features from documents.

A bag of words is a representation of text that describes the occurrence of words within a document. We just keep track of word counts and disregard the grammatical details and the word order. It is called a “bag” of words because any information about the order or structure of words in the document is discarded. 

# Tf-Idf Vectorizer
TF (Term Frequency): The number of times a word appears in a document is its Term Frequency. A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.

IDF (Inverse Document Frequency): Words that occur many times a document, but also occur many times in many others, may be irrelevant. IDF is a measure of how significant a term is in the entire corpus.

The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.

# Hashing Vectorizer
hashing vectorizer is a vectorizer which uses the hashing trick to find the token string name to feature integer index mapping. Conversion of text documents into matrix is done by this vectorizer where it turns the collection of documents into a sparse matrix which are holding the token occurence counts. Advantages for hashing vectorizer are:

As there is no need of storing the vocabulary dictionary in the memory, for large data sets it is very low memory scalable. As there in no state during the fit, it can be used in a streaming or parallel pipeline.
# ML model Traning and Building
1. Multinomial Naive Bayes:It is used for classification when is in discrete form. It is very useful in text processing. Each text will be converted to a vector of word count. It cannot deal with negative numbers.
It is predefined in Scikit Learn Library. So we can import that class in our project then we create an object of Multinomial Naive Bayes Class.

2. PassiveAggressiveClassifier:Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. 
