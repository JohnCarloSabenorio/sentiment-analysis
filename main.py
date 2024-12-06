import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score 
from sklearn.pipeline import make_pipeline
import joblib
import nltk
from nltk.corpus import stopwords
import re
import numpy
import matplotlib.pyplot as plt


import string
punct = string.punctuation
#download the stopwords library
stop_words = set(stopwords.words('english'))

nltk.download('stopwords')
# retrieves csv data
def load_data(path):
    data = pd.read_csv(path)
    data = data[['Review', 'Rating']]
    return data
    

# def text_data_cleaning(sentence):
#     # doc = nlp(sentence)                        
    
#     # tokens = [] # list of tokens
#     # for token in doc:
#     #     print(f"TOKEN: {token}")
#     #     if token.lemma_ != "-PRON-":
#     #         temp = token.lemma_.lower().strip()
#     #     else:
#     #         temp = token.lower_
#     #     print(f"TEMP: {temp}")
#     #     tokens.append(temp)

#     # cleaned_tokens = []
#     # for token in tokens:
#     #     if token not in stop_words and token not in punct:
#     #         cleaned_tokens.append(token)

#     # return ' '.join(cleaned_tokens)


#     return ' '.join(word for word in sentence.split() if word not in stop_words) # Adds the word to the string if the word is not a stop word

    
def preprocess_data(data):
    # data['sentiment'] = data['Rating'].apply(lambda x: 'positive' if x > 3  else ('neutral' if x == 3 else 'negative'))
    # convert to all lower case
    data['Review'] = data['Review'].str.lower()
    
    # remove special characters
    data['Review'] = data['Review'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    
    #remove stopwords
    # data['Review'] = data['Review'].apply(lambda x: text_data_cleaning(x))
    data['Review'] = data['Review'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

    data['Rating'] = data['Rating'].apply(lambda x: 1 if x > 3 else 0)

    # CHECKS FOR NULL VALUES
    print(data)
    return data
    
def train_model(data):
    X = data['Review']
    y = data['Rating']
    
    # data splitting for testing and training
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
    # pipeline build
    model = make_pipeline(TfidfVectorizer(), LinearSVC())
    cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    model.fit(X, y)    
    # model evaluation
    # predictions = model.predict(X_test)
    # print(f"Model Accuracy: {accuracy_score(y_test, predictions):.2f}")
    
    # save the model to a file
    joblib.dump(model, 'sentiment_model.pk1')
    print("Sentiment model was created.")
    return model, cross_val_scores

def predict_rating(model, user_review):
    user_review = user_review.lower()
    user_review = re.sub(r'[^a-z\s]', '', user_review)
    user_review = ' '.join(word for word in user_review.split() if word not in set(stopwords.words('english')))

    # this variable is a list 
    prediction = model.predict([user_review])
    return prediction[0]

def plot_accuracy(cross_val_scores):
    plt.figure(figsize = (8, 6))
    plt.plot(range(1, len(cross_val_scores) + 1), cross_val_scores, marker='o',linestyle='-', color='b')
    plt.title('Malupet na Analysis ni Sabenorio', fontsize= 16)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0,1)
    plt.grid(True)
    plt.show()
    plt.savefig('Accuracy_Sabenorio.jpg')
# main class
file = 'spotify_reviews.csv'
data = load_data(file)
pre_data = preprocess_data(data)

# training the model
model, cross_val_scores = train_model(pre_data)
user_review1 = 'Really buggy and terrible to use as of recently'
predicted_rating = predict_rating(model, user_review1)

# Display results
print("Sample Prediction of User Review: ", user_review1, "\nPredicted Rating: ", predicted_rating)
plot_accuracy(cross_val_scores)


'''
1. research vectorizer alternatives
2. change algorithm
3. limit language to english
'''