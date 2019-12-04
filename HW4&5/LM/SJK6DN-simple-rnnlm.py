#!/usr/bin/env python
# coding: utf-8

# # Importing packages and processing the data

# In[1]:


import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import WordPunctTokenizer


# In[2]:


# generating vocab
vocab = {}
id = 1

for i in open("trn-wiki.txt", encoding="utf8"):
    tokens = i.split()
    for j in tokens:
        if j not in vocab:
            vocab[j] = id
            id+=1


# In[3]:


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

# In[4]:


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


# In[1]:


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
            if iterations%5000 == 0:
                print("Epoch number",epoch, " Iteration number", iterations)
            iterations += 1

            model.zero_grad()

            prob_scores = model( torch.LongTensor(sentence[:-1]))

            loss = loss_function(prob_scores,  torch.LongTensor(sentence[1:]))
            loss.backward()
            optimizer.step()
    return model
model = train(trn_data)


# In[ ]:




