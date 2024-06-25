#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS= set(stopwords.words('english'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import re


# In[3]:


get_ipython().run_line_magic('pip', 'install xgboost')


# In[4]:


data=pd.read_csv(r"C:\Users\abhis\Downloads\amazon_alexa.tsv",delimiter='\t',quoting=3)


# # Exploratory Data Analysis

# In[5]:


# Exploratory Data Analysis (EDA) is a crucial initial step in data science projects. 
#  It involves analyzing and visualizing data to understand its key characteristics, uncover patterns, 
#  and identify relationships between variables refers to the method of studying and exploring record 
# sets to apprehend their predominant traits, discover patterns, locate outliers, and 
# identify relationships between variables 


# In[6]:


print(f"Dataset shape:{data.shape}")


# In[7]:


data.head()


# In[8]:


print(data.columns.values)


# In[9]:


data.isnull().sum()


# In[10]:


data[data["verified_reviews"].isna()==True]


# In[11]:


data.dropna(inplace=True)


# In[12]:


data.isna().sum()


# In[13]:


data.shape


# In[14]:


#creating a new column "Length" that will maintain the length of the string in "verified_reviews" column


# In[15]:


data['verified_reviews'].apply(len)


# In[16]:


data["length"]=data['verified_reviews'].apply(len)


# In[17]:


data.describe()


# In[18]:


data.head()


# In[19]:


#veryfying length

print(f"'verified reviewes:' column_name : {data.iloc[4]['verified_reviews']}")


# In[59]:


len(data.iloc[4]['verified_reviews'])


# In[60]:


data.iloc[0:4]['length']


# In[22]:


#Datatypes of features


# In[23]:


data.dtypes


# In[24]:


#Analyzing "rating " column


# In[25]:


print(f"Rating value count: {data['rating'].value_counts()}")


# In[26]:


data['rating'].value_counts().plot.bar(color='violet')

plt.title('rating distribution count')
plt.xlabel('Rating')
plt.ylabel('count')
plt.show()


# In[27]:


print(f"% of count per rating:\n {round(data['rating'].value_counts()/len(data['rating'])*100,2)}")


# In[28]:


len(data['rating'])


# In[62]:


fig=plt.figure(figsize=(7,7))
colors=('red','blue','green','orange','yellow')
wp={'linewidth':1,"edgecolor":'black'}

tags=data['rating'].value_counts()/data.shape[0]

explode=(0.1,0.1,0.1,0.1,0.1)
tags.plot(kind='pie',autopct="%1.1f%%",shadow=True,colors=colors,startangle=90,wedgeprops=wp,explode=explode,label='percentage of rating')


# # Analyzing feedback column

# In[30]:


print(f"Feedback value count: \n{data['feedback'].value_counts()}")


# # Extracting verified reviews

# In[31]:


# Negative reviews

review_0=data[data["feedback"]==0].iloc[1]["verified_reviews"]
print(review_0)


# In[32]:


# Positive reviews

review_1=data[data["feedback"]==1].iloc[1]["verified_reviews"]
print(review_1)


# In[33]:


# feedback 0 means it is negative feedback, and 1 means positive feedback


# In[34]:


#total negative feedback
a=(data["feedback"]==0).sum()

print(a)


# In[35]:


#total positive feedback
b=(data["feedback"]==1).sum()
print(b)


# In[36]:


#% -ve feedback
print(a*100/(a+b))


# In[37]:


#% +ve feedback
print(b*100/(a+b))


# In[38]:


data['feedback'].value_counts().plot.bar(color='yellow')


# In[39]:


fig=plt.figure(figsize=(7,7))
colors=('blue','orange')
wp={'linewidth':1,"edgecolor":'black'}

tags=data['feedback'].value_counts()/data.shape[0]

explode=(0.1,0.1)
tags.plot(kind='pie',autopct="%1.1f%%",shadow=True,colors=colors,startangle=90,wedgeprops=wp,explode=explode,label='percentage of rating')


# # understanding ratings of posive and negative feedback

# In[40]:


data[data['feedback']==1]['rating'].value_counts()


# In[41]:


data[data['feedback']==0]['rating'].value_counts()

#rating 1,2 is for 0 (negative feedback)


# # Analyzing variation column

# In[42]:


data['variation'].value_counts()


# In[43]:


data['variation'].value_counts().plot.bar(color='grey')

plt.title('Variations in Alexa')
plt.xlabel('Variation')
plt.ylabel('count')
plt.show()


# In[44]:


data.groupby('variation')['rating'].mean()


# In[45]:


data.groupby('variation')['rating'].mean().sort_values().plot.bar(color= 'pink')


# In[46]:


data['verified_reviews'].value_counts() 

# give error 'Series' object has no attribute 'value_count'


# In[47]:


data['length'].value_counts() 


# In[48]:


sns.histplot(data[data['feedback']==1]['length'],color='green').set(title="Distribution of length of review if feedback is positive(1)")


# In[49]:


sns.histplot(data[data['feedback']==0]['length'],color='green').set(title="Distribution of length of review if feedback is negative(0)")


# In[50]:


sns.histplot(data[data['feedback']==1]['length'],color='green').set(title="Distribution of length of review if feedback is positive(1)")


# In[51]:


data.groupby('length')['rating'].mean().plot.hist(color='blue',figsize=(7,6),bins=20)
plt.title("Review length wise mean ratings")
plt.xlabel('ratings')
plt.ylabel('length')
plt.show()


# In[52]:


cv=CountVectorizer(stop_words='english')
words=cv.fit_transform(data.verified_reviews)


# In[53]:


#combine all reviews
reviews=" ".join([review for review in data['verified_reviews']])

#initialize wordcloud object
wc=WordCloud(background_color='white',max_words=50)

#generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(reviews))
plt.title('Wordcloud for all reviews', fontsize=10)
plt.axis('off')
plt.show()


# In[54]:


#combine all reviews
reviews=" ".join([reviews for reviews  in data['verified_reviews']])

#initialize wordcloud object
wc=WordCloud(background_color='white',max_words=50)

#generate and plot wordcloud
plt.figure(figsize=(10,10))
plt.imshow(wc.generate(reviews))
plt.title('Wordcloud for all reviews', fontsize=10)
plt.axis('off')
plt.show()


# In[55]:


#combineing all negative feed back and spliting them
neg_reviews=" ".join([review for review in data[data['feedback']==0]['verified_reviews']])
neg_reviews=neg_reviews.lower().split()

#combineing all positive feed back and spliting them

pos_reviews=" ".join([review for review in data[data['feedback']==1]['verified_reviews']])
pos_reviews=pos_reviews.lower().split()

#Finding words from reviews which are present in that feedback category only
unique_negative=[x for x in neg_reviews if x not in pos_reviews]
unique_negaive=" ".join(unique_negative)

unique_positive=[x for x in neg_reviews if x not in neg_reviews]
unique_positive=" ".join(unique_positive)



# In[56]:


wc=WordCloud(background_color="white",max_words=50)

plt.figure(figsize=(10,10))
plt.imshow(wc.generate(unique_negative))
plt.title("Wordcloud for negative reviews",fontsize=10)
plt.axis("off")
plt.show()


# In[ ]:





# In[ ]:




