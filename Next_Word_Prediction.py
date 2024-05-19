#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding ,LSTM , Dense


# In[2]:


with open("C:\\Users\\pankaj singh mehta\\Desktop\\mughal.txt", 'r',encoding='utf-8') as myfile:
    mytext= myfile.read()


# In[3]:


mytext


# In[4]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
mytokenizer = Tokenizer()
mytokenizer.fit_on_texts([mytext])


# In[5]:


mytokenizer.word_index


# In[26]:


len(mytokenizer.word_index)#unique word present in corpus


# In[7]:


total_words = len(mytokenizer.word_index)+1


# In[8]:


for line in mytext.split('\n'):
    print(line)


# In[37]:


for line in mytext.split('\n'):
    token_list = mytokenizer.texts_to_sequences([line])[0]
    print(token_list)


# In[9]:


my_input_sequences =[]
for line in mytext.split('\n'):
    
    token_list = mytokenizer.texts_to_sequences([line])[0]
    
    for i in range(1,len(token_list)):
        my_n_gram_sequence = token_list[:i+1]
        print(my_n_gram_sequence)
        my_input_sequences.append(my_n_gram_sequence)


# In[10]:


max([len(x)for x in my_input_sequences]) #maximum word present in a sentence


# In[11]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
max_sequence_len = max([len(seq) for seq in my_input_sequences])
input_sequences = np.array(pad_sequences(my_input_sequences, maxlen=max_sequence_len, padding='pre'))


# In[12]:


input_sequences


# In[13]:


X = input_sequences[:,:-1]
Y = input_sequences[:,-1]


# In[14]:


X.shape


# In[15]:


X[0]


# In[22]:


print(X.shape,Y.shape)


# In[16]:


Y.shape


# In[17]:


Y[1]


# In[18]:


Y = np.array(tf.keras.utils.to_categorical(Y, num_classes=total_words))


# In[19]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding ,LSTM , Dense
model = Sequential()
model.add(Embedding(total_words,100, input_length =max_sequence_len-1))
model.add(LSTM(150))
model.add(Dense(total_words,activation='softmax'))
print(model.summary())


# In[20]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X,Y ,epochs=55,verbose=1)


# In[40]:


input_text ='babur son'
predict_next_words = 3

for _ in range(predict_next_words):
    token_list = mytokenizer.texts_to_sequences([input_text])[0]  #using tokenizer
    print(token_list)
    token_list = pad_sequences ([token_list],maxlen=max_sequence_len-1,padding='pre') # padding
    predicted = np.argmax(model.predict(token_list),axis=1)
    output_word = ""
    for word, index in mytokenizer.word_index.items():
        if index== predicted:
            output_word = word
            break
    input_text +=" " + output_word
    
print(input_text)    


# In[ ]:





# In[ ]:




