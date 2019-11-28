#!/usr/bin/env python
# coding: utf-8

# # 0. Package Dependency
# 
# - [nltk](https://www.nltk.org)
# - [sklearn](http://scikit-learn.org/stable/)

# In[214]:


# Load packages
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WordPunctTokenizer
import numpy as np
# Load data
trn_texts = open("trn-reviews.txt").read().strip().split("\n")
trn_labels = open("trn-labels.txt").read().strip().split("\n")
print("Training data ...")
print("%d, %d" % (len(trn_texts), len(trn_labels)))

dev_texts = open("dev-reviews.txt").read().strip().split("\n")
dev_labels = open("dev-labels.txt").read().strip().split("\n")
print("Development data ...")
print("%d, %d" % (len(dev_texts), len(dev_labels)))


# In[215]:


glove = pd.read_csv("glove.6b.50d.txt", sep=' ', header=None, quotechar=None,  quoting=3)

wtrn_data = numpy.array([WordPunctTokenizer().tokenize(i.lower()) for i in trn_texts])
wdev_data = numpy.array([WordPunctTokenizer().tokenize(i.lower()) for i in dev_texts])


# In[216]:


def txt_rep(data):
    new_data = np.array([[0]*50])
    k = 0
    for i in data:
        addition = [0]*50
        count = 0
        df = glove[glove[0].isin(i)]
        for j in i:
            try:
                temp = np.array(df[df[0] == j])
                addition = np.sum([temp[0][1:], addition], axis=0)
                count+=1
            except: pass #print(j)
        new_data = np.append(new_data,[addition/count], axis=0)
#         if k == 5:
#             break
        k+=1
        
    return new_data
gtrn_data = txt_rep(wtrn_data)[1:]
gdev_data = txt_rep(wdev_data)[1:]


# In[ ]:


print(gtrn_data.shape)


# In[ ]:


from sklearn.linear_model import LogisticRegression


print(gtrn_data.shape)
# Define a LR classifier
classifier = LogisticRegression(solver="liblinear", multi_class="auto")
classifier.fit(gtrn_data, trn_labels)

# Measure the performance on training and dev data
print("Training accuracy = %f" % classifier.score(gtrn_data, trn_labels))
print("Dev accuracy = %f", classifier.score(gdev_data, dev_labels))


# # 2. Feature Extraction
# 
# Please refer to the document of [_CountVectorizer_](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) for the parameters of this function. 

# In[ ]:


choice = 1

if choice == 1:
    print("Preprocessing without any feature selection")
    vectorizer = CountVectorizer(lowercase=False)
    # vocab size 77166
elif choice == 2:
    print("Lowercasing all the tokens")
    vectorizer = CountVectorizer(lowercase=True)
    # vocab size 60610
else:
    raise ValueError("Unrecognized value: choice = %d" % choice)

trn_data = vectorizer.fit_transform(trn_texts)
print(trn_data.shape)
dev_data = vectorizer.transform(dev_texts)
print(dev_data.shape)


# In[ ]:


def comb(data, gdata):
    new_data = np.array([[0]*77216])
    for i,j in zip(data,gdata):
        new_data = np.append(new_data, [np.append(i.toarray(), j )], axis=0)    
    return new_data
ctrn_data = comb(trn_data, gtrn_data)[1:]    
cdev_data = comb(dev_data, gdev_data)[1:]    


# In[ ]:


ctrn_data.shape


# # 3. Logistic Regression
# 
# Please refer to the document of [_LogisticRegression_](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for the parameters of this function. 

# In[ ]:


from sklearn.linear_model import LogisticRegression

# Define a LR classifier
classifier = LogisticRegression(solver="liblinear", multi_class="auto")
classifier.fit(ctrn_data, trn_labels)

# Measure the performance on training and dev data
print("Training accuracy = %f" % classifier.score(ctrn_data, trn_labels))
print("Dev accuracy = %f", classifier.score(cdev_data, dev_labels))


# In[ ]:




