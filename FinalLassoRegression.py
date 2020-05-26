#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


# In[93]:


plt.style.use('fivethirtyeight')
cols=["forex_rate","cpi","ppi","bank_rate","current_account","inflation_rate","gdp","per1","per2","per3"]
data=pd.read_table('Workbook3.csv',sep=',',names=cols,parse_dates=[0], index_col=0,header=0)
data.forex_rate=data.forex_rate.astype(float)
data.head()


# In[94]:


cols2=["cpi","ppi","bank_rate","current_account","inflation_rate","gdp","per1","per2","per3"]
y=data["forex_rate"]
x=data[cols2]


# In[95]:


##Here we have three regression models - SVM, Ridge, Lasso


# In[96]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,test_size=0.25,random_state=0)


# In[113]:


svr=SVR(epsilon=2)
svr.fit(x_train, y_train)
y_pred=svr.predict(x_test)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.plot(y_pred, color='blue',label='prediction')
plt.plot(np.array(y_test), color='red', label='original')
plt.show()


# In[114]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[115]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[116]:


from sklearn.linear_model import Ridge
model=Ridge()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.plot(y_pred, color='blue',label='prediction')
plt.plot(np.array(y_test), color='red', label='original')
plt.show()


# In[117]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[122]:


from sklearn.linear_model import Lasso
model=Lasso()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.plot(y_pred, color='blue',label='prediction')
plt.plot(np.array(y_test), color='red', label='original')
plt.show()


# In[123]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[124]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)


# In[125]:


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 10
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size


# In[126]:


#Correlation Matrix
#features show multicollinearity 

import matplotlib.pyplot as plt


# In[127]:


def plot_correlation_matrix(df):
    figure, axis = plt.subplots(nrows=1, ncols=1)
    caxis = axis.matshow(df.corr())
    ticks = list(range(len(df.columns)))
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)

    axis.set_xticklabels(df.columns, rotation=90, horizontalalignment='right')
    axis.set_yticklabels(df.columns)
    plt.show()


# In[128]:


y_train,y_test=np.split(data["forex_rate"], indices_or_sections=[8812], axis=0)

X=data[["cpi","ppi","bank_rate","current_account","inflation_rate","gdp","per1","per2","per3"]]
x_train = X[0:8812]
x_test = X[8812:11812]
plot_correlation_matrix(data[["cpi","ppi","bank_rate","current_account","inflation_rate","gdp"]])


# In[130]:


plt.plot(y_pred, color='red',label='prediction')
plt.plot(np.array(y_test), color='blue', label='original')

plt.show()


# In[133]:


model.fit(x,y)


# In[135]:


#April 4, 2018
#Average: 1 USD = 65.1393 INR
x_test =np.array(["137.1","116.8","6","-2.87E+11","4.58","2.80E+12","65.11","65.12","65.15"])
x_test=pd.DataFrame(x_test)
x_test=x_test.values.reshape(1,9)
y_pred=model.predict(x_test)

y_pred


# In[136]:


#June 30,2018
#Average: 1 USD = 68.4403 INR
x_test =np.array(["138.6","119.2","6.250","-2.87E+11","5","2.80E+12","68.81","68.63","68.31"])
x_test=pd.DataFrame(x_test)
x_test=x_test.values.reshape(1,9)
y_pred=model.predict(x_test)
y_pred


# In[137]:


#May 1, 2018
#Average: 1 USD = 66.6031 INR
x_test =np.array(["137.8","117.9","6","-2.87E+11","4.87","2.80E+12","66.51","66.68","66.8"])
x_test=pd.DataFrame(x_test)
x_test=x_test.values.reshape(1,9)
y_pred=model.predict(x_test)

y_pred


# In[138]:


#June 1, 2018
#Average: 1 USD = 67.1451 INR
x_test =np.array(["138.6","119.2","6.250","-2.87E+11","5","2.80E+12","67.4","67.46","67.78"])
x_test=pd.DataFrame(x_test)
x_test=x_test.values.reshape(1,9)
y_pred=model.predict(x_test)

y_pred


# In[ ]:




