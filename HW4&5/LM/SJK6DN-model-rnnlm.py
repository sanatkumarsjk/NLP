#!/usr/bin/env python
# coding: utf-8

# # Importing packages and processing the data

# In[14]:


import math
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WordPunctTokenizer

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
print("Using:", device)
# In[15]:


# generating vocab
vocab = {}
id = 0

for i in open("trn-wiki.txt", encoding="utf8"):
    tokens = i.split()
    for j in tokens:
        if j not in vocab:
            vocab[j] = id
            id+=1


# In[16]:


def generate_indices(file):
    final_data = []

    for i in file:
        tokens = i.split()
        sample = []
        for j in tokens:
            sample.append(vocab[j])    
        final_data.append(sample)
    return final_data

trn_data = generate_indices(open("trn-wiki.txt", encoding="utf8"))
dev_data = generate_indices(open("dev-wiki.txt", encoding="utf8"))


# # Defining the model
# ## Hyper parameters
#     
# - Hidden dimensions = 32
# - Embedding Dimensions = 32
# - Word embedding = nn.Embedding
# - Batch Size = 1
# - LSTM layers = 1
# - Optimizer = SGD
# - Sentence length = Any

# In[17]:


class LM_LSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LM_LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        prob_space = self.hidden(lstm_out.view(len(sentence), -1))
        prob_scores = F.log_softmax(prob_space, dim=1)
        return prob_scores


# In[18]:


def train(data, model):
    #hyper parameters
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1): 
        iterations = 1
        for sentence in data:
            if iterations%5000 == 0:
                print("Epoch number",epoch, " Iteration number", iterations)
            iterations += 1

            model.zero_grad()

            prob_scores = model( torch.LongTensor(sentence[:-1]).to(device))

            loss = loss_function(prob_scores,  torch.LongTensor(sentence[1:]).to(device))
            loss.backward()
            optimizer.step()
    return model


# # Calculating perplexity 

# In[19]:


def cal_perlexity(model, data):
    sum = 0 
    samp_count = 0
    count = 1
    for sample in data:
        if count%5000 == 0:
            print("Analzed", count,"samples for perplexity")
        count+=1    
        samp_count += len(sample)
        with torch.no_grad():       
            prob_scores = model( torch.LongTensor(sample[:-1]).to(device))
            
            for index in range(len(prob_scores)):
                sum += prob_scores[index][sample[index]]
    return math.exp(-(sum/samp_count))           


# In[20]:



dimensions = [64, 128, 256]

for i in dimensions:
    print("Training model with embedding dimensions:", i, " and hidden dimensions:", i)
    model = LM_LSTM(i, i, len(vocab), len(vocab)).to(device)
    model = train(trn_data, model)
    trn_per = cal_perlexity(model, trn_data)
    print("Train perplexity is:", trn_per)
    dev_per = cal_perlexity(model, dev_data)
    print("Dev perplexity is:", dev_per)


# In[ ]:




