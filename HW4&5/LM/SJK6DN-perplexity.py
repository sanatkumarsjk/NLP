#!/usr/bin/env python
# coding: utf-8

# # Importing packages and processing the data

# In[8]:


import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WordPunctTokenizer


# In[9]:


# generating vocab
vocab = {}
id = 0

for i in open("trn-wiki.txt", encoding="utf8"):
    tokens = i.split()
    for j in tokens:
        if j not in vocab:
            vocab[j] = id
            id+=1


# In[10]:


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

# In[11]:


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
        prob_scores = F.softmax(prob_space, dim=1)
        return prob_scores


# In[40]:


def train(data):
    #hyper parameters
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 32
    model = LM_LSTM(EMBEDDING_DIM, HIDDEN_DIM, len(vocab), len(vocab))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1): 
        iterations = 1
        for sentence in data:
            if iterations%2 == 0:
                print("Epoch number",epoch, " Iteration number", iterations)
            iterations += 1

            model.zero_grad()

            prob_scores = model( torch.LongTensor(sentence[:-1]))

            loss = loss_function(prob_scores,  torch.LongTensor(sentence[1:]))
            loss.backward()
            optimizer.step()
    return model
model = train(trn_data)


# # Calculating perplexity 

# In[56]:


def cal_perlexity(model, data):
    sum = 0 
    samp_count = 0
    
    for sample in data:
        samp_count += len(sample)
        
        with torch.no_grad():       
            prob_scores = model( torch.LongTensor(sample[:-1]))
            
            for index in range(len(prob_scores)):
                sum += prob_scores[index][sample[index]]
    return sum/samp_count           


# In[ ]:


trn_per = cal_perlexity(model, trn_data)
print("Train perplexity is:", trn_per)
dev_per = cal_perlexity(model, dev_data)
print("Dev perplexity is:", trn_per)

