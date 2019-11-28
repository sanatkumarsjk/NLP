#!/usr/bin/env python
# coding: utf-8

# In[465]:


import pandas as pd
from csv import reader

def read(filename):
    data = open("data.conll", "r")
    return data


# In[466]:


visited = set()
def process_Data(data):
    new_sentence = False
    parsing_data = []
    head = [0] * 100
    res = []
    count = 0
    for line in data:
        if line[0] == '#':
            continue
        d = line.split('\t')

        if line == '\n':
            count+=1
            if count !=100000:
                res.append(find_actions(parsing_data, head))
            parsing_data = []
            head = [0]*100
            visited.clear()
            continue
        parsing_data.append(d)
        head[int(d[6])] +=1
        head[int(d[0])] +=1
    return res

def find_actions(data,head):
    seq = []
    stack = []
    i = -1
    while i < len(data)-1: 
        i+=1
        if len(stack) < 1:
            seq.append('Shift')
            stack.append(data[i])
            if len(stack) < 2:
                continue
            i+=1
        seq.append('Shift')
        stack.append(data[i])   
        seq+=process(stack,head)    
    return seq

def delete_elm(stack,head):
    continue_flg = False
    if head[int(stack[-1][0])] <= 0:
        del stack[-1]
        continue_flg = True
        if head[int(stack[-1][0])] <= 0:
            del stack[-1]
            continue_flg = True     
    elif head[int(stack[-2][0])] <= 0:
        del stack[-2]
        continue_flg = True  
    return continue_flg


# In[467]:


def process(stack, head):
    seq = []   
    flag = True
    while flag and len(stack) > 1:
        continue_flg = False
        if (stack[-1][0],stack[-2][0]) in visited:
            return seq
        if stack[-1][6] == stack[-2][0]:
            visited.add((stack[-1][0],stack[-2][0]))
            seq.append("RightArc"+stack[-2][0]+'-'+ stack[-1][0])
            head[int(stack[-1][0])] -= 1
            head[int(stack[-2][0])] -= 1
            continue_flg = delete_elm(stack, head)
                
        elif stack[-2][6] == stack[-1][0]:
            visited.add((stack[-1][0],stack[-2][0]))
            seq.append("LeftArc"+stack[-1][0]+'-'+ stack[-2][0])
            head[int(stack[-2][0])] -= 1
            head[int(stack[-1][0])] -= 1
            continue_flg = delete_elm(stack, head)           
            
        if continue_flg: continue
        flag = False
    return seq
    
   
data = read('data.conll')  


# In[468]:


with open('sjk6dn-parsing-actions.txt', 'w') as filehandle:
    for listitem in process_Data(data):
        filehandle.write('%s\n' % listitem)


# In[ ]:





# In[ ]:





# In[ ]:




