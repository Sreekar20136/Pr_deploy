import sys
import os
import pickle
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Data Collection and Pre Processing
# 1 Load Data from csv file to a pandas datafram
raw_mail_data = pd.read_csv('./mail_data.csv')
raw_mail_data.head()
# Replace the null values with a null string
mail_data = raw_mail_data.where(pd.notnull(raw_mail_data),'')
# Printing first 5 row of mail data
mail_data.head()
# Checking the number of rows and columns
mail_data.shape
#  Label Encoding
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
mail_data.head()
# Seperating the text as texts and label
X = mail_data['Message']
Y = mail_data['Category']
X.head()
Y.head()
X_Train,X_test,Y_Train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)
X.shape
X_Train.shape
Y_test.shape
# Feature Extraction 
# Transform text data to feature vectors that can be used as input to the logistic regression
feature_extraction = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_feature = feature_extraction.fit_transform(X_Train)
X_test_feature = feature_extraction.transform(X_test)

#  Convert Y_train and T_test as Integers

Y_Train = Y_Train.astype('int')
Y_test = Y_test.astype('int')
print(X_train_feature)
X_Train
# Training the Model
# Logistic Regression
model = LogisticRegression()
Y_Train
model.fit(X_train_feature,Y_Train)
# Evaluating the Trained Model
# Predition on Training Model
prediction_on_Training_Data = model.predict(X_train_feature)
accuracy_on_training_data = accuracy_score(Y_Train,prediction_on_Training_Data)
print("Accuracy for Training : ",accuracy_on_training_data * 100)
# Predict on Test Data
prediction_on_Test_Data = model.predict(X_test_feature)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_Test_Data)
print("Accuracy for Training : ",accuracy_on_test_data * 100)
#  Building a Predictable System
input_mail = ["As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a £1500 Bonus Prize, call 09066364589"]

# Convert Text to feature vectors
input_data_feature = feature_extraction.transform(input_mail)

# Making Prediction
prediction = model.predict(input_data_feature)

print(prediction)

if(prediction == [1]):
    print("This is the Ham Mail.")
else:
    print("This is the Spam Mail.")

