#!/usr/bin/env python
# coding: utf-8

# # **Exploratory Data Analysis**
# 

# In[81]:


import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# In[82]:


import warnings
warnings.filterwarnings("ignore")


# Since we were provided with 2 datasets, with train and test datasets, so we are importing both the datasets together in 1 cell
# 

# In[83]:


fraud_train=pd.read_csv("D:/fraudTrain.csv")
fraud_test=pd.read_csv("D:/fraudTest.csv")
df = pd.concat([fraud_train,fraud_test], ignore_index = True)


# In[ ]:





# In[84]:


df.drop(['Unnamed: 0'],axis=1,inplace=True)  


# In[12]:


df.duplicated().any()   #Finding any duplicate values


# In[85]:


#Calculating age from Date of Birth
df['age'] = pd.DatetimeIndex(df["trans_date_trans_time"]).year-pd.DatetimeIndex(df["dob"]).year


# In[86]:


df["year"] = pd.DatetimeIndex(df["trans_date_trans_time"]).year.astype(int).astype(str)
df["hour"] = pd.DatetimeIndex(df["trans_date_trans_time"]).hour
df["month"] = pd.DatetimeIndex(df["trans_date_trans_time"]).month
d = {1:'JAN', 2:'FEB', 3:'MAR', 4:'APR', 5:'MAY', 6:'JUN', 7:'JUL', 8:'AUG', 9:'SEP', 10:'OCT', 11:'NOV', 12:'DEC'}
df.month = df.month.map(d)
df['dayofweek'] = pd.DatetimeIndex(df["trans_date_trans_time"]).dayofweek+1
d1 = {1:'MON',2:'TUE',3:'WED',4:'THU', 5:'FRI',6:'SAT',7:'SUN'}
df.dayofweek = df.dayofweek.map(d1)
df["day"] = pd.DatetimeIndex(df["trans_date_trans_time"]).day


# In[87]:


df.loc[(df['city_pop'] < 10000), ['pop_dense']] = "Less Dense"
df.loc[((df['city_pop'] > 10000) & (df['city_pop'] < 50000)), 
       ["pop_dense"]] = "Moderate Dense"
df.loc[(df['city_pop'] > 50000), ['pop_dense']] = "More Dense"

df = df.drop("city_pop", axis = 1)
df.pop_dense.value_counts(normalize = True)          
'''we can work with cities through their population parameter, as names of citiescannot implement whether a fraud will be done or not, while
population of a city can.'''


# In[88]:



''' sometimes distance from the customer's home location to the merchant's 
location can prove out to be main reason for fraud, so taking the 
#difference of longitude and lattitude of respective columns'''

df["lat_diff"] = abs(df["lat"] - df["merch_lat"])                           #1 degree = 110 kms
df["long_diff"] = abs(df["long"] - df["merch_long"]) 

''' it will be difficult to calculate distance between merchant's location
 or customer's location so applying pythogoras theorem'''   

df['displacement'] = np.sqrt(pow((df['lat_diff']*110),2) + pow((df['long_diff']*110),2))
df.head()


# In[89]:


df.loc[(df['displacement'] < 50), ['location']] = "Closeby"
df.loc[((df['displacement'] >= 50) & (df['displacement'] <= 100)),['location']] = "Far"
df.loc[(df['displacement'] > 100), ['location']] = "Very Far"
df.location.value_counts(normalize = True)


# In[90]:


df["recency"] = df.groupby(by="cc_num")["unix_time"].diff()


# In[91]:


df.loc[df.recency.isnull(),["recency"]] = -1


# In[92]:


# dividing recency to segments based on number of hours passed
df.recency = df.recency.apply(lambda x: float((x/60)/60))
df.loc[(df["recency"]<1),["recency_segment"]] = "Recent_Transaction"
df.loc[((df["recency"]>1) & (df["recency"]<6)),["recency_segment"]] = "Within 6 hours"
df.loc[((df["recency"]>6) & (df["recency"]<12)),["recency_segment"]] = "After 6 hours"
df.loc[((df["recency"]>12) & (df["recency"]<24)),["recency_segment"]] = "After Half-Day"
df.loc[(df["recency"]>24),["recency_segment"]] = "After 24 hours"
df.loc[(df["recency"]<0),["recency_segment"]] = "First Transaction"
df.recency_segment.value_counts(normalize = True)


# In[93]:


df1 = df.copy()


# In[94]:


df1.drop(['trans_date_trans_time','first','last','street','city','zip','lat','long','dob','trans_num','location','unix_time','merch_lat','merch_long','lat_diff','long_diff'],axis=1,inplace=True)


# In[96]:


df1.drop(['state','job','year','day','recency','merchant'],axis=1,inplace=True)


# In[113]:


#converting categorical variables to numerical
#ordinalEncoding


# In[97]:


df1['gender'] = df1['gender'].replace(('M', 'F'), (0,1))


# In[98]:


from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(dtype = np.float64)
enc.fit(df1.loc[:, ['category', 'dayofweek', 'pop_dense','month','recency_segment']])
df1.loc[:, ['category', 'dayofweek', 'pop_dense','month','recency_segment']] = enc.transform(df1[['category', 'dayofweek','pop_dense','month','recency_segment']])


# In[99]:


df1['recency_segment'].value_counts()


# In[100]:


df1.loc[df1.recency_segment.isnull(),["recency_segment"]] = -1


# In[ ]:





# In[102]:


x = df1.drop(['is_fraud'],axis=1) 
y = df1['is_fraud'] 


# In[103]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4, random_state=42)


# Xgboost
# In[104]:


# from sklearn.tree import DecisionTreeClassifier
# dtr=DecisionTreeClassifier()
# model2=dtr.fit(x_train,y_train)
# pickle.dump(model2, open('model.pkl','wb'))

# model = pickle.load(open('model.pkl','rb'))
# print(model.predict(x_train.iloc[0,:].values.reshape(1,-1)))


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
model3=rf.fit(x_train,y_train)
pickle.dump(model3, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict(x_train.iloc[0,:].values.reshape(1,-1)))
