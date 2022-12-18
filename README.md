# Cheat Detection Model
 NLP RNN Model for predicting cheat statements 

This natural language processing text classification model uses various algorithms to predict if a sentence is closely related to a student trying to ask for answers in a group chat. 

Developed this model for a Social Network used to connect with students with same majors [Afterhours]() developed by Tarleton Computer Society

Libraries used for this include pandas and numpy for data manipulation and seaborn, matplotlib, and pickle for data visualization. The model also utilizes various nltk (Natural Language Toolkit) libraries for text preprocessing, including word_tokenize for tokenizing text into individual words and stopwords for removing common stopwords. The model also uses the SnowballStemmer from nltk for stemming words and the WordNetLemmatizer for lemmatizing words.

For model building and evaluation, the model utilizes several sklearn (Scikit-learn) libraries, including train_test_split for dividing the data into training and testing sets using a specified test size and random state. The model also uses various classifiers, including Logistic Regression, SGDClassifier, and MultinomialNB, which are all based on different mathematical algorithms. The model also utilizes various evaluation metrics, including the classification report, F1 score, accuracy score, confusion matrix, ROC curve, AUC, and ROC AUC score, to assess the performance of the model.

In addition to these traditional machine learning algorithms, the model also utilizes feature extraction techniques such as bag of words representation and word embedding. For bag of words representation, the model uses TfidfVectorizer and CountVectorizer from sklearn's feature_extraction library to convert the text data into numerical feature vectors. For word embedding, the model uses the Word2Vec algorithm from gensim to learn dense vector representations of words based on their contexts in the text data.

Once the data is fully preprocessed and the features are extracted, the model trains and evaluates the different classifiers using the extracted features and the corresponding labels. The classifier with the highest evaluation metric scores is chosen as the final model, which is then saved using pickle for future use.
