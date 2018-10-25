#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 18:47:59 2018

@author: shekhar
"""

#importing the dataset
import pandas as pd
from bs4 import BeautifulSoup
import re
train = pd.read_csv("labeledTrainData.tsv", header=0 , delimiter='\t', quoting = 3) #Labeled training dataset header starts from the beginning to read the file, delimeter differentiates words by spaces, and quoting is ignoring the double quotes
train.shape  #reading the 25k rows and 3 columns
train.columns.values

#using nltk
import nltk
from nltk.corpus import stopwords  # i'll remove the stop keywords using nltk

def review_words(raw_review):
    
    #using beautifulsoup to remove the markups instead of regular expressions
    all_rev = BeautifulSoup(raw_review)
    
    #getting raw output
    all_rev.get_text()   #NO TAGS OR NO MARKUPS
    letters_only = re.sub("[^a-zA-Z]"," ",all_rev.get_text()) #find and replace
    #print(letters_only)
    
    #now this is tokenization converting to lower and splitting it by space
    lower_case = letters_only.lower() #lower case
    words = lower_case.split() #splitting up the words
    
    #by Google searching a set is much faster than searching a list so i will convert stopwords to sets
    stop_set = set(stopwords.words("english"))
    
    #removing the stop words Finally!!!!
    words_with_meaning = [w for w in words if not w in stop_set] 
    
    # Joining the words at the end
    return(" ".join(words_with_meaning))
    
    
#getting the number of reviews from the datsets
all_reviews = train["review"].size
print(all_reviews)

#empty list to hold the cleaned ones
final_reviews = []

#looping over the each reviews
for i in range(0,all_reviews):
    if (i % 1000 == 0):
        print("review {} of {}".format(i,all_reviews))
    final_reviews.append(review_words(train["review"][i]))
    
#now comes bag of words which is basically converting the words into numeric before giving it to the machine
print("Creeating the bag of words \n")

#using countvectorizer tool (BOW tool)

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

#have an object
my_vect = CountVectorizer(analyzer = "word",max_features=5000)

#fitting and transforming the model

trained_features = my_vect.fit_transform(final_reviews)

#at the end obviously everything has to be converted into ARRAAAAY!!... so

trained_features = trained_features.toarray()

#vocabulary
vocabs = my_vect.get_feature_names()
print(vocabs)

#counting the total number of vocabs

total_vocabs = np.sum(trained_features, axis=0)

#printing the vocabs and number of vocabs
for tag, count in zip(vocabs, total_vocabs):
    print(count,tag)
    
#RANdOM FOREST
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100)

#fitting the forest to training set
forest = forest.fit(trained_features, train["sentiment"])

#for TEST data
test = pd.read_csv("testData.tsv", header = 0, delimiter='\t',quoting=3)
print(test.shape) #test size

#empty list for cleaned reviews
all_reviews = len(test["review"])
final_test_reviews=[]

#looping over the each reviews
for i in range(0,all_reviews):
    if (i % 1000 == 0):
        print("review {} of {}".format(i,all_reviews))
    final_test_reviews.append(review_words(test["review"][i]))

#fitting and transforming the model
tested_features = my_vect.fit_transform(final_test_reviews)
tested_features=tested_features.toarray()

#once again random forest for sentiment finalizatrion
final_result=forest.predict(tested_features)

# Output to new dataframe with new column
opt=pd.DataFrame(data={"id":test["id"] , "sentiment":final_result})

#FINAL outputfile
opt.to_csv("Popcorn_OutPut.csv",index=False, quoting=3)
