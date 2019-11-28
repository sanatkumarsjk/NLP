

# Load packages
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WordPunctTokenizer
import numpy as np

# Load data
print("Loading the Data")
trn_texts = open("trn-reviews.txt").read().strip().split("\n")
trn_labels = open("trn-labels.txt").read().strip().split("\n")
print("Training data ...")
print("%d, %d" % (len(trn_texts), len(trn_labels)))

dev_texts = open("dev-reviews.txt").read().strip().split("\n")
dev_labels = open("dev-labels.txt").read().strip().split("\n")
print("Development data ...")
print("%d, %d" % (len(dev_texts), len(dev_labels)))


# In[215]:

print("Processing the glove file")
glove = pd.read_csv("glove.txt", sep=' ', header=None, quotechar=None,  quoting=3)
dict = {}
for i, row in glove.iterrows():
    if len(dict)%50000 ==0:
        print(len(dict), "words completed")
    dict[row[0]] = np.array(row.drop([0]))
	
	
wtrn_data = np.array([WordPunctTokenizer().tokenize(i.lower()) for i in trn_texts])
wdev_data = np.array([WordPunctTokenizer().tokenize(i.lower()) for i in dev_texts])


# In[216]:

print("processing input data using glove embeddings")
def txt_rep(data):
    new_data = np.array([[0]*50])
    k = 0
    for i in data:
        addition = np.array([0]*50)
        count = 0
        for j in i:
            try:
                temp = dict[j]
                addition = np.sum([temp, addition], axis=0)
                count+=1
            except: pass
        count = max(1,count)
        new_data = np.append(new_data,[addition/count], axis=0)
        if k%1000 == 0:
            print(k)
        k+=1
    return new_data


gtrn_data = txt_rep(wtrn_data)[1:]
gdev_data = txt_rep(wdev_data)[1:]
print('this block is done')



print("training the LR model with glove data")
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

print("extrating count vectors")
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

print("preping final data with count vectors and the glove")
cdev_data = np.concatenate((dev_data.toarray(),gdev_data), axis=1)
ctrn_data = np.concatenate((trn_data.toarray(),gtrn_data), axis=1)

# def comb(data, gdata):
#     new_data = np.array([[0]*77216])
#     for i,j in zip(data,gdata):
#         new_data = np.append(new_data, [np.append(i.toarray(), j )], axis=0)
#     return new_data
# ctrn_data = comb(trn_data, gtrn_data)[1:]
# cdev_data = comb(dev_data, gdev_data)[1:]


# In[ ]:


ctrn_data.shape


# # 3. Logistic Regression
# 
# Please refer to the document of [_LogisticRegression_](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) for the parameters of this function. 

# In[ ]:

print("training the LR model for 1.2")
from sklearn.linear_model import LogisticRegression

# Define a LR classifier
classifier = LogisticRegression(solver="liblinear", multi_class="auto")
classifier.fit(ctrn_data, trn_labels)

# Measure the performance on training and dev data
print("Training accuracy = %f" % classifier.score(ctrn_data, trn_labels))
print("Dev accuracy = %f", classifier.score(cdev_data, dev_labels))


# In[ ]:




