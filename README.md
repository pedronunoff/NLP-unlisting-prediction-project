# Text Mining: predicting unlisting airbnb listings from their reviews and characteristics

This project was developed for the Text Mining course in the Master's Degree in Data Science and Advanced Analytics (NOVA IMS).

The main goal was to do binary classification (will be unlisted or not) from text data with customer reviews and airbnb characteristics in multiple languages. All code in Python.

## Main Steps (and methods/algorithms)

- Data Exploration with checks for missing values, imbalances, word counts
- Data Preprocessing with language detection and text cleaning (lowercasing, removing tags, numerical data, ponctuation, emojis, stopwords and rare words, doing lemmatization)
- Feature Engineering with Word Embeddings (TF-IDF, GLoVe, STSB-XLM-R-Multilingual, DistilUSE multilingual)
- Modelling and performance comparison, including word embeddings' contribution to performance, and hyperparameter tuning (kNN, Logistic Regression, MLP)
- Choosing the best model to do final predictions on unlabeled data

It is worth to note that our model, despite being simple, got the award for having the highest f1 score from all projects on later released results from the predictions on unlabeled data.
