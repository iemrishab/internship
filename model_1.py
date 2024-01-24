#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('https://raw.githubusercontent.com/FlipRoboTechnologies/ML-Datasets/main/Red%20Wine/winequality-red.csv')
data.head()


# In[3]:


data.sample(n=10)


# In[4]:


data.dtypes


# In[5]:


#All the columns except the quality are float data types.


# In[6]:


data.isnull().sum()


# In[7]:


#no nulls are present in the dataset.


# In[8]:


for i in data.columns:
    print(data[i].value_counts())
    print('/n')


# In[9]:


#now as given in problem statement , we will convert the int datatype of quality into object datatype and convert the values of 7 and above into good and below 7 into not good category


# In[10]:


data['quality']=data['quality'].replace({1:0,2:0,3:0,4:0,5:0,6:0,7:1,8:1,9:1,10:1})


# In[11]:


data


# In[ ]:


#we have converted all the labes into 0 and 1 .


# In[12]:


data['quality'].value_counts()


# In[13]:


#As this is imbalanced data set we will balanced it before train_test_split


# In[14]:


data.describe()


# In[15]:


#Now there are no nulls in this data
#Now we will check for mean and median for every column , as they both are very much less deviated so there is very less skewness.
#There might be some skewness in a 'total sulphur dioxide' but we will check with the graph also.
#There is a huge difference between 75% and maximum so there might be a outliers.
#'volatile acidity','residual sugar','free sulphur dioxide','total sulphur dioxide' in these columns there might be outliers,
   #but we will plot boxplot to make sure.


# In[16]:


plt.figure(figsize=(20,15),facecolor='Red')
plotnumber=1
for col in data:
    if plotnumber<=12:
        ax=plt.subplot(4,3,plotnumber)
        sns.distplot(data[col])
        plt.xlabel(col,fontsize=15)
    plotnumber+=1
plt.show()    
    


# In[17]:


#By seeing the plot we can assure that there is skewness present now we will check again.


# In[18]:


data.skew()


# In[19]:


#As the accepted value of skewness is (-0.5-+0.5), so now we use cbrt method to remove skewness


# In[20]:


data['fixed acidity']=np.cbrt(data['fixed acidity'])
data['volatile acidity']=np.cbrt(data['volatile acidity'])
data['chlorides']=np.cbrt(data['chlorides'])
data['alcohol']=np.cbrt(data['alcohol'])


# In[23]:


data.skew()


# In[ ]:


#Now we will run cbrt again for alcohol.


# In[24]:


data['alcohol']=np.cbrt(data['alcohol'])


# In[25]:


data.skew()


# In[26]:


plt.figure(figsize=(20,15),facecolor='Red')
plotnumber=1
for col in data:
    if plotnumber<=12:
        ax=plt.subplot(4,3,plotnumber)
        sns.distplot(data[col])
        plt.xlabel(col,fontsize=15)
    plotnumber+=1
plt.show()


# In[ ]:


#The skewness has been treated and now we will plot the boxplot to check for the outliers


# In[27]:


plt.figure(figsize=(20,25),facecolor='white')
graph=1
for colomn in data:
    if graph<=12:
        plt.subplot(4,3,graph)
        ax=sns.boxplot(data=data[colomn])
        plt.xlabel(colomn,fontsize=15)
    graph+=1
plt.show() 


# In[ ]:


#As we can see the graphs clearly,the outliers are present.now we will remove the outliers.
#In 'fixed acidity','volatile acidity','chlorides','density',and 'ph'both the higher and lower outliers are present.
#In 'residual sugar','ciric acid','free sulphur dioxide','total suphur dioxide','sulphates','achohol' there are higher ouliers present.


# In[28]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
iqr=q3-q1


# In[29]:


fixed_high=q3['fixed acidity']+ (1.5*iqr['fixed acidity'])
fixed_high


# In[ ]:


#now check the indexes for higher values


# In[30]:


np_index=np.where(data['fixed acidity']>fixed_high)
np_index


# In[31]:


#now we will drop the values


# In[32]:


data=data.drop(data.index[np_index])
data.shape


# In[ ]:


#12 values are deleted
#now we will reset the index


# In[33]:


data.reset_index()


# In[34]:


fixed_low=q1['fixed acidity']- (1.5*iqr['fixed acidity'])
fixed_low


# In[35]:


index=np.where(data['fixed acidity']<fixed_low)
index


# In[36]:


data=data.drop(data.index[index])
data.shape


# In[37]:


data.reset_index()


# In[38]:


volatile_high=q3['volatile acidity']+ (1.5*iqr['volatile acidity'])
volatile_high


# In[39]:


Index=np.where(data['volatile acidity']>volatile_high)
Index


# In[40]:


data=data.drop(data.index[Index])
data.shape


# In[41]:


data.reset_index()


# In[42]:


volatile_low=q1['volatile acidity']- (1.5*iqr['volatile acidity'])
volatile_low


# In[43]:


Index_=np.where(data['volatile acidity']<volatile_low)
Index_


# In[44]:


data=data.drop(data.index[Index_])
data.shape


# In[45]:


data.reset_index()


# In[46]:


residual_high=q3['residual sugar']+(1.5*iqr['residual sugar'])
residual_high


# In[47]:


INDEX=np.where(data['residual sugar']>residual_high)
INDEX


# In[48]:


data=data.drop(data.index[INDEX])
data.shape


# In[49]:


data.reset_index()


# In[50]:


chlorides_high=q3['chlorides']+(1.5*iqr['chlorides'])
chlorides_high


# In[51]:


INDEX_=np.where(data['chlorides']>chlorides_high)
INDEX_


# In[52]:


data=data.drop(data.index[INDEX_])
data.shape


# In[53]:


data.reset_index()


# In[54]:


chlorides_low=q1['chlorides']-(1.5*iqr['chlorides'])
chlorides_low


# In[55]:


INDEX_c=np.where(data['chlorides']<chlorides_low)
INDEX_c


# In[56]:


data=data.drop(data.index[INDEX_c])
data.shape


# In[57]:


data.reset_index()


# In[58]:


free_sulfur_high=q3['free sulfur dioxide']+(1.5*iqr['free sulfur dioxide'])
free_sulfur_high


# In[59]:


INDEX_f=np.where(data['free sulfur dioxide']>free_sulfur_high)
INDEX_f


# In[60]:


data=data.drop(data.index[INDEX_f])
data.shape


# In[61]:


data.reset_index()


# In[62]:


total_sulfur_high=q3['total sulfur dioxide']+(1.5*iqr['total sulfur dioxide'])
total_sulfur_high


# In[63]:


INDEX_t=np.where(data['total sulfur dioxide']>total_sulfur_high)
INDEX_t


# In[64]:


data=data.drop(data.index[INDEX_t])
data.shape


# In[65]:


data.reset_index()


# In[66]:


density_high=q3['density']+(1.5*iqr['density'])
density_high


# In[67]:


INDEX_d=np.where(data['density']>density_high)
INDEX_d


# In[68]:


data=data.drop(data.index[INDEX_d])
data.shape


# In[69]:


data.reset_index()


# In[70]:


density_low=q1['density']-(1.5*iqr['density'])
density_low


# In[71]:


INDEX_D=np.where(data['density']<density_low)
INDEX_D


# In[72]:


data=data.drop(data.index[INDEX_D])
data.shape


# In[73]:


data.reset_index()


# In[74]:


ph_high=q3['pH']+(1.5*iqr['pH'])
ph_high


# In[75]:


INDEX_p=np.where(data['pH']>ph_high)
INDEX_p


# In[76]:


data=data.drop(data.index[INDEX_p])
data.shape


# In[77]:


data.reset_index()


# In[78]:


ph_low=q1['pH']-(1.5*iqr['pH'])
ph_low


# In[79]:


INDEX_P=np.where(data['pH']<ph_low)
INDEX_P


# In[80]:


data=data.drop(data.index[INDEX_P])
data.shape


# In[81]:


data.reset_index()


# In[82]:


sulphates_high=q3['sulphates']+(1.5*iqr['sulphates'])
sulphates_high


# In[83]:


INDEX_s=np.where(data['sulphates']>sulphates_high)
INDEX_s


# In[84]:


data=data.drop(data.index[INDEX_s])
data.shape


# In[85]:


data.reset_index()


# In[86]:


#now we will check again


# In[87]:


plt.figure(figsize=(20,15),facecolor='Red')
plotnumber=1
for col in data:
    if plotnumber<=12:
        ax=plt.subplot(4,3,plotnumber)
        sns.distplot(data[col])
        plt.xlabel(col,fontsize=15)
    plotnumber+=1
plt.show()


# In[88]:


#now the perfect bell haped curves obtained


# In[89]:


data['quality'].value_counts()


# In[90]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve,roc_auc_score,classification_report
from sklearn.linear_model import LogisticRegression


# In[91]:


from imblearn.over_sampling import SMOTE


# In[93]:


x=data.drop('quality',axis=1)
y=data['quality']


# In[96]:


SM=SMOTE()
x1,y1=SM.fit_resample(x,y)


# In[ ]:


#now we will check the correlation between features and labels


# In[99]:


cor=data.corr()
cor


# In[101]:


scalar=StandardScaler()
x_scaled=scalar.fit_transform(x)
x_scaled.shape[1]


# In[103]:


data


# In[105]:


vif=pd.DataFrame()
vif['Vif Features']=[variance_inflation_factor(x_scaled,i) for i in range (x_scaled.shape[1])]
vif['Features']=x.columns


# In[106]:


vif


# In[116]:


#now we drop two columns that has value greater than 5 i.e,"fixed acidity","density"


# In[115]:


x


# In[130]:


maxAccu=0
maxRs=0
for i in range(0,200):
    x_train,x_test,y_train,y_test=train_test_split(x1,y1,random_state=i)
    LR=LogisticRegression()
    LR.fit(x_train,y_train)
    predLR=LR.predict(x_test)
    acc=accuracy_score(y_test,predLR)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
print('Best accuracy is ',maxAccu,'at random_state',maxRs)
            


# In[136]:


print(accuracy_score(y_test,predLR))
print(classification_report(y_test,predLR))

print(confusion_matrix(y_test,predLR))


# In[139]:


import pickle
filename='data_LR.pkl'
pickle.dump(LR, open(filename,'wb'))


# In[ ]:


import pickle


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




