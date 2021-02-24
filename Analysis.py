#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
sns.set()
from datetime import datetime
import math
from numpy import mean
from numpy import array
from sklearn.metrics import mean_squared_error
get_ipython().magic('matplotlib inline')


# In[6]:


dataset=pd.read_csv("crime.csv",header=0)
pd.options.display.max_columns = None


# In[124]:


dataset.info()


# In[125]:


dataset.describe()


# In[126]:



dataset.shape


# In[127]:


dataset.head(15)


# # Problem Statement
# 
# ### Predicting the number of solved and unsolved cases per district
# 
# # Key questions
# 
# ### Calculate the ratio to see the change in solved and unsolved cases?
# 
# ### Which district has improved the most and which have gotten worse
# 
# ### What time of the day the most cases are reported and vice versa##
# 
# ### What type of cases have been solved the most and least by each district
# 

# # Part 1: Fixing the data

# In[7]:


##Filtering out rows with NaN
def check_null_col(col):
    null_columns=col.columns[col.isnull().any()]
    return col[null_columns].isnull().sum()

dataset_check_NaN=check_null_col(dataset)
print("Name of column and number of Null values in 2015 dataset:    ")
print(dataset_check_NaN)
print("Size of dataset =",dataset.shape)


# In[8]:


#Deleting the rows with NaN values to avoid discrepancy
dataset_temp=dataset
dataset_temp=dataset_temp.dropna()
dataset_temp=dataset_temp.reset_index(drop=True)
print(dataset_temp.shape)
dataset_temp


# In[130]:


#Here we will seperate date and time into seperate cells for easier operations
datetime = pd.to_datetime(dataset_temp["Date"]) 
datetime_updatedon=pd.to_datetime(dataset_temp["Updated On"])


# In[131]:


dataset_temp['timestamp']=datetime
dataset_temp['Date'] = dataset_temp['timestamp'].dt.date
dataset_temp['Year'] = dataset_temp['timestamp'].dt.year
dataset_temp['Month_num'] = dataset_temp['timestamp'].dt.month
dataset_temp['Time'] = dataset_temp['timestamp'].dt.time

dataset_temp['timestamp_Updated On']=datetime_updatedon
dataset_temp['Date_Updated On'] = dataset_temp['timestamp_Updated On'].dt.date
dataset_temp['Year_Updated On'] = dataset_temp['timestamp_Updated On'].dt.year
dataset_temp['Month_num_Updated On'] = dataset_temp['timestamp_Updated On'].dt.month
dataset_temp['Time_Updated On'] = dataset_temp['timestamp_Updated On'].dt.time


# In[132]:


#Converting months from numbers to names
cvdow = pd.Series(["Extra","January", "February", "March", "April", "May", "June", "July","August","September","October","November","December"], name="Months")
dataset_temp['Month']=dataset_temp.join(cvdow,on='Month_num')["Months"]
dataset_temp.drop("Month_num",axis=1,inplace=True,errors='ignore')


# In[133]:


#Dropping the months_num col
dataset_temp['Month_Updated On']=dataset_temp.join(cvdow,on='Month_num_Updated On')["Months"]
dataset_temp.drop("Month_num_Updated On",axis=1,inplace=True,errors='ignore')


# In[134]:


dataset_temp


# # Part 2: Answering Key Questions

# ## Calculate the ratio to see the change in solved and unsolved cases per year?

# In[135]:


dataset_cases=dataset_temp.groupby(['Year','Arrest']).count()
dataset_cases=dataset_cases.loc[:,['ID']]
dataset_cases.rename(columns={'ID':'Number of Cases Solved'},inplace=True)

x=dataset_cases.loc[(dataset_cases.index.get_level_values("Arrest")==False)] 
x=x.reset_index()
x.drop(x.columns[1],axis=1,inplace=True)
x=x.set_index("Year")

y=dataset_cases.loc[(dataset_cases.index.get_level_values("Arrest")==True)] 
y=y.reset_index()
y.drop(y.columns[1],axis=1,inplace=True)
y=y.set_index("Year")

z=y/x
z.rename(columns={'Number of Cases Solved':'Ratio'},inplace=True)

dataset_cases=dataset_cases.join(z)
dataset_cases=dataset_cases.reset_index()
dataset_cases=dataset_cases.set_index(['Year','Ratio','Arrest'])
dataset_cases


# In[136]:


plt.figure(figsize=(30,15))
heatmap1_data=pd.pivot_table(dataset_cases,values="Number of Cases Solved",index=['Arrest'],columns='Year')
plt.title("Solved and Unsolved Arrets per Year")
cases_heatmap=sns.heatmap(heatmap1_data,annot=True,cmap="coolwarm",center=2000)
cases_heatmap
plt.show()


# In[137]:


plt.figure(figsize=(20,15))
sns.lineplot(data=dataset_cases,x=dataset_cases.index.get_level_values('Year'),y=dataset_cases.index.get_level_values('Ratio'))
plt.title("Ratio Of Arrest Completed vs Not Completed")
plt.show()


# In[138]:


#Ratio of crime solved per district



# In[139]:


#Overall Ratio

pie_arrest_ratio=dataset_temp.groupby('Arrest').count()
pie_arrest_ratio=pie_arrest_ratio.loc[:,(['ID'])]
pie_arrest_ratio.rename(columns={"ID":"Count"},inplace=True)
pie_arrest_ratio=pie_arrest_ratio.reset_index()

plt.figure(figsize=(5,5))

plt.pie(data=pie_arrest_ratio,x='Count',autopct="%.1f%%",labels='Arrest')
plt.title("Percentage of True Arrest vs False Arrest Over 20 Years")
plt.show()


# ## Which district has improved the most and which have gotten worse
#  

# In[140]:


dataset_areas=dataset_temp.groupby(['Year','District']).count()
# dataset_areas=dataset_areas.drop(dataset_areas.columns.difference('ID'),1,inplace=True)
dataset_areas=dataset_areas.loc[:,['ID']]
dataset_areas.rename(columns={'ID':'Number of Cases'},inplace=True)
dataset_areas


# In[143]:


plt.figure(figsize=(30,30))
sns.lineplot(data=dataset_areas,x=dataset_areas.index.get_level_values('Year'),y="Number of Cases",hue=dataset_areas.index.get_level_values('District'),legend="full",)
plt.title("Trend Of Cases Reported In Different Districts Over 20 Years")
plt.show()


# In[144]:


distribution=dataset_temp[(dataset_temp['X Coordinate']>0) & (dataset_temp['Y Coordinate']>0)]
sns.lmplot(x='X Coordinate', 
           y='Y Coordinate',
           data=distribution,
           fit_reg=False, 
           hue="District",
           palette='gist_ncar_r',
           height=10,
           ci=None,
           scatter_kws={"marker": "D", 
                        "s": 10})
ax = plt.gca()
ax.set_title("All Crime Distribution per District")


# ## What time of the day the most cases are reported and vice versa 

# In[145]:


day = dataset_temp["timestamp"].dt.day_name()
dataset_temp["Day"] = day
hour = dataset_temp["timestamp"].dt.hour
dataset_temp["Hour"] = hour.astype(int)
day_of_the_month = dataset_temp["timestamp"].dt.day.astype(int)
dataset_temp["Day of Month"] = day_of_the_month
dataset_temp


# In[146]:


df = dataset_temp.set_index("timestamp")
df.head()


# In[147]:


#Hours of the day
plt.figure(figsize=(12,6))
Hour_crime = dataset_temp[["Hour"]].groupby("Hour").size()
Hour_crime.plot(kind = "bar")
plt.xlabel("Hours of the Day")
plt.ylabel("Number of Cases")


# In[148]:


plt.figure(figsize=(11,5))
df.resample('D').size().plot(legend=False)
plt.title('Number of crimes per month')
plt.xlabel('Months')
plt.ylabel('Number of crimes')
plt.show()


# In[149]:


#Days of the week
plt.figure(figsize=(6,6))
day_crime = dataset_temp[["Day"]].groupby("Day").size()
day_crime.plot(kind = "bar")
plt.xlabel("Days of the Week")
plt.ylabel("Number of Cases")


# In[150]:


plt.figure(figsize=(11,5))
df.resample('W').size().plot(legend=False)
plt.title('Number of crimes per month')
plt.xlabel('Months')
plt.ylabel('Number of crimes')
plt.show()


# In[151]:


#Days of the Month
plt.figure(figsize=(6,6))
day_crime = dataset_temp[["Day of Month"]].groupby("Day of Month").size()
day_crime.plot(kind = "bar")
plt.xlabel("Days of the Month")
plt.ylabel("Number of Cases")


# In[152]:


plt.figure(figsize=(11,5))
df.resample('M').size().plot(legend=False)
plt.title('Number of crimes per month')
plt.xlabel('Months')
plt.ylabel('Number of crimes')
plt.show()


# In[153]:


#Months of the year
plt.figure(figsize=(12,6))
month_crime = dataset_temp[["Month"]].groupby("Month").size()
month_crime.plot(kind = "bar")
plt.xlabel("Month of the Year")
plt.ylabel("Number of Cases")


# In[154]:


plt.figure(figsize=(11,5))
df.resample('M').size().plot(legend=False)
plt.title('Number of crimes per month')
plt.xlabel('Months')
plt.ylabel('Number of crimes')
plt.show()


# In[155]:


#Crimes committed each year
plt.figure(figsize=(12,6))
year_crime = dataset_temp[["Year"]].groupby("Year").size()
year_crime.plot(kind = "bar")
plt.xlabel("Year")
plt.ylabel("Number of Cases")


# In[156]:


plt.figure(figsize=(11,5))
df.resample('Y').size().plot(legend=False)
plt.title('Number of crimes per month')
plt.xlabel('Months')
plt.ylabel('Number of crimes')
plt.show()


# ## What type of cases have been solved the most and least by each district
#  

# In[157]:


highest_crime=dataset_temp.groupby(['District','Primary Type']).count()
highest_crime=highest_crime.loc[:,['ID']]
highest_crime.rename(columns={'ID':'Total Cases'},inplace=True)
highest_crime=highest_crime.sort_values('Total Cases',ascending=False)
highest_crime=highest_crime.reset_index()

plt.figure(figsize=(45,25))
sns.swarmplot(data=highest_crime,x='District',y='Total Cases',hue='Primary Type',size=12)
plt.title("Total Cases Reported By Type Over 20 Years")
plt.show()


# In[158]:


# Most Common Crime in each district

most_common=dataset_temp.groupby(['District','Primary Type']).count()
most_common=most_common.loc[:,['ID']]
most_common.rename(columns={'ID':'Total Cases'},inplace=True)
most_common=most_common.sort_values(['Total Cases','District'],ascending=False)
most_common=most_common.reset_index()
most_common=most_common.drop_duplicates('District').sort_values('District',ascending=True).reset_index(drop=bool)

most_common

plt.figure(figsize=(25,15))
sns.scatterplot(data=most_common,x='District',y='Total Cases',hue='Primary Type',size='Primary Type',sizes=(400,400),style="Primary Type")
plt.title("Most Common Crimes For Each District")
plt.show()


# In[159]:


least_solved=dataset_temp[['Primary Type','Arrest']]
least_solved=least_solved[least_solved.Arrest.eq(False)].groupby('Primary Type').count()
least_solved.rename(columns={'Arrest':'Total Cases'},inplace=True)
least_solved=least_solved.sort_values('Total Cases',ascending=False)
least_solved

plt.figure(figsize=(25,15))
sns.barplot(data=least_solved,y=least_solved.index.get_level_values('Primary Type'),x='Total Cases',orient="h",palette="Reds_d",ci=None)
plt.title("Least Cases Solved Over The Last 20 Years")
plt.show()


# In[160]:


most_solved=dataset_temp[['Primary Type','Arrest']]
most_solved=most_solved[most_solved.Arrest.eq(True)].groupby('Primary Type').count()
most_solved.rename(columns={'Arrest':'Total Cases'},inplace=True)
most_solved=most_solved.sort_values('Total Cases',ascending=False)
most_solved

plt.figure(figsize=(25,15))
sns.barplot(data=most_solved,y=most_solved.index.get_level_values('Primary Type'),x='Total Cases',orient="h",palette="Greens_d",ci=None)
plt.title("Most Cases Solved Over The Last 20 Years")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Part 3: Machine Learning 

# In[9]:


dataset_temp


# In[10]:


#getting a sample from original dataset as the dataset is tooo large to compute over. It leads to ram issues.
sampled_df=dataset_temp.sample(n=2000000)
sampled_df=sampled_df.reset_index(drop=True)


# In[11]:


#Machine LearningT
#Primary Type, Location, Arrest
from sklearn import preprocessing
x=sampled_df[['Primary Type','Domestic','District']]
y=sampled_df['Arrest']
x.head()


# In[12]:


le=preprocessing.LabelEncoder()
le.fit(x.loc[:,['Domestic']])
x['Domestic']=le.transform(x.loc[:,['Domestic']])
y=le.fit_transform(y)


# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[14]:


district_classes=[]
for i in sorted(x['District'].unique()):
  a="District "+str(i)
  district_classes.append(a)


# In[15]:


ohe=OneHotEncoder()
feature_arr_d=ohe.fit_transform(x[['District']]).toarray()
df_temp1=pd.DataFrame(feature_arr_d,columns=district_classes)


# In[16]:


feature_arr_pt=ohe.fit_transform(x[['Primary Type']]).toarray()
df_temp=pd.DataFrame(feature_arr_pt,columns=sorted(x['Primary Type'].unique()))


# In[17]:


df_temp1=df_temp1.reset_index(drop=True)
df_temp=df_temp.reset_index(drop=True)


# In[18]:


df_x=pd.DataFrame(x.iloc[:,[1]])
for i in df_temp.columns:
  df_x[i]=df_temp[i]
for i in df_temp1.columns:
  df_x[i]=df_temp1[i]


# In[19]:


#Final Feature Matrix with one-hot-encoding
df_x.head()


# In[20]:


x_train, x_test, y_train, y_test = train_test_split(df_x, y, random_state = 0)
clf=LogisticRegression().fit(x_train,y_train)
prediction=clf.predict(x_test)
print("Accuracy:",accuracy_score(y_test,prediction))
print("Confusion matrix:")
print(confusion_matrix(y_test,prediction))


# In[21]:


p = clf.predict_proba(x_test)
df_p=pd.DataFrame(p,columns=['False Arrest Prob.','True Arrest Prob.'])


# In[22]:


df_test=pd.DataFrame()
df_test=pd.concat([x['Primary Type'],x['District'],x_test.iloc[:,[0]]], axis=1, join='inner')
for i in df_p.columns:
  df_test[i]=df_p[i]

df_test.head()


# # Forecast

# In[161]:


day = dataset_temp["timestamp"].dt.day_name()
dataset_temp["Day"] = day
hour = dataset_temp["timestamp"].dt.hour
dataset_temp["Hour"] = hour.astype(int)
day_of_the_month = dataset_temp["timestamp"].dt.day.astype(int)
dataset_temp["Day of Month"] = day_of_the_month
dataset_temp.head()


# In[162]:


df = dataset_temp.set_index("timestamp")
df.head()


# ## True Arrest Forecast

# In[163]:


df1 = df[df["Arrest"]==True].resample('D').size().to_frame()
df1.columns = ["Number of Crimes"]


# In[164]:


fig, ax = plt.subplots() 
df1.plot(ax=ax, figsize = (12,6), legend = False)
df1.rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)
plt.ylabel('Number of Arrests:True')
plt.xlabel('Years') 
plt.legend(['Actual',"Moving Average"]) 


# In[165]:


train_df = df1.loc['2004-01-01':'2016-12-31']
test_df = df1.loc['2017-01-01':'2020-10-10']

test_df.tail(20)


# In[166]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

fit_model = ExponentialSmoothing(train_df['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 680).fit()

prediction = fit_model.forecast(1379)
prediction

train_df['Number of Crimes'].plot(figsize=(12,6))
test_df['Number of Crimes'].plot()
prediction.plot(xlim=['2015-01-01','2020-10-10'])
plt.ylabel('Number of Arrests:True')
plt.xlabel('Years') 
plt.legend(['Training',"Testing","Predicted"]) 


# In[167]:


from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(test_df, prediction)


# In[168]:


mean_squared_error(test_df, prediction)


# In[169]:


np.sqrt(mean_squared_error(test_df, prediction))


# In[170]:


model_df = df1.loc['2004-01-01':'2020-10-10']


# In[171]:


crime_model = ExponentialSmoothing(model_df['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=680).fit()

crime_forecast = crime_model.forecast(365)

df1.plot(figsize=(12,6))
crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:True')
plt.xlabel('Years') 
plt.legend(['Actual',"Forecasted"]) 


# ## False Arrest Forecast

# In[172]:


df2 = df[df["Arrest"]==False].resample('D').size().to_frame()
df2.columns = ["Number of Crimes"]


# In[173]:


fig, ax = plt.subplots() 
df2.plot(ax=ax, figsize = (12,6), legend = False)
df2.rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)
plt.ylabel('Number of Arrests:False')
plt.xlabel('Years') 
plt.legend(['Actual',"Moving Average"])


# In[174]:


train_df = df2.loc['2004-01-01':'2016-12-31']
test_df = df2.loc['2017-01-01':'2020-10-10']

test_df.tail(20)


# In[175]:


fit_model = ExponentialSmoothing(train_df['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=680).fit()

prediction = fit_model.forecast(1379)
prediction

train_df['Number of Crimes'].plot(figsize=(12,6))
test_df['Number of Crimes'].plot()
prediction.plot(xlim=['2015-01-01','2020-10-10'])
plt.ylabel('Number of Arrests:False')
plt.xlabel('Years') 
plt.legend(['Training',"Testing","Predicted"]) 


# In[176]:


mean_absolute_error(test_df, prediction)


# In[177]:


mean_squared_error(test_df, prediction)


# In[178]:


np.sqrt(mean_squared_error(test_df, prediction))


# In[179]:


model_df = df2.loc['2004-01-01':'2020-10-10']


# In[180]:


crime_model = ExponentialSmoothing(model_df['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=680).fit()
# Let's predict 365 days into the future
crime_forecast = crime_model.forecast(365)
# Print original and then our prediction
df2.plot(figsize=(12,6))
crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:False')
plt.xlabel('Years') 
plt.legend(['Actual',"Forecasted"]) 


# In[ ]:





# In[181]:


df_True_district = df[df["Arrest"]==True].groupby([pd.Grouper('District')]+ [pd.Grouper(level = 'timestamp', freq = 'M')]).size().to_frame()
df_True_district.columns = ["Number of Crimes"]
#df4['timestamp']= pd.to_datetime(df4['timestamp'])
df_True_district


# In[182]:


df_True_plot = df_True_district.query("District <= 5.0")
df_True_plot

plt.figure(figsize=(17,10))
sns.lineplot(data = df_True_plot,x = df_True_plot.index.get_level_values('timestamp'), 
             y = "Number of Crimes",
             hue = df_True_plot.index.get_level_values('District'),
             style = df_True_plot.index.get_level_values('District'),
             palette = "flag_r",
             size_order = "list",
             legend = "full",
             dashes=False)


# In[183]:


data_tr = df_True_district.query("timestamp >= '2004-01-01'  & timestamp <= '2016-12-31'")
data_test = df_True_district.query("timestamp >= '2017-01-01'  & timestamp <= '2020-10-10'")


# In[184]:


listOfDist = list(df_True_district.index.get_level_values('District').unique())
listOfDist


# In[185]:


train_df = []
for district in listOfDist:
    result = data_tr[data_tr.index.get_level_values('District') == district]
    train_df.append(result)

train_df = pd.concat(train_df)
train_df


# In[186]:


test_df = []
for district in listOfDist:
    result = data_test[data_test.index.get_level_values('District') == district]
    test_df.append(result)

test_df = pd.concat(test_df)
test_df


# ## District 1

# In[187]:


fig, ax = plt.subplots() 
df_True_district.query('District == 1').groupby('timestamp').mean()['Number of Crimes'].plot(ax=ax, figsize = (12,6), legend = False)
df_True_district.query('District == 1').groupby('timestamp').mean()['Number of Crimes'].rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)


# In[188]:


fit_district_model = ExponentialSmoothing(train_df.query('District == 1').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 24).fit()

prediction = fit_district_model.forecast(45)
prediction.index = test_df.query('District == 1').groupby('timestamp').mean().index
prediction


fig, ax = plt.subplots() 
train_df.query('District == 1').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
prediction.plot()#xlim=['2015-01-01','2020-10-10'])
#test_df.query('District == 1')['Number of Crimes'].plot()
#prediction.plot(xlim=['2015-01-01','2020-10-10'])      


# In[189]:


model_df = df_True_district.query("timestamp >= '2004-01-01'  & timestamp <= '2020-10-10'")


# In[190]:


crime_model = ExponentialSmoothing(model_df.query('District == 1').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=24).fit()

crime_forecast1 = crime_model.forecast(12)
# Print original and then our prediction
#df_True_district[df_True_district.index.get_level_values('District')==1].plot(figsize=(12,6))
#crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
 

#fig, ax = plt.subplots() 
df_True_district.query('District == 1').groupby('timestamp').mean().plot(figsize=(12,6))
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast1.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:True')
plt.xlabel('Years')


# In[ ]:





# In[ ]:





# ## District 2

# In[191]:


fig, ax = plt.subplots() 
df_True_district.query('District == 2').groupby('timestamp').mean()['Number of Crimes'].plot(ax=ax, figsize = (12,6), legend = False)
df_True_district.query('District == 2').groupby('timestamp').mean()['Number of Crimes'].rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)


# In[192]:


fit_district_model = ExponentialSmoothing(train_df.query('District == 2').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 18).fit()

prediction = fit_district_model.forecast(45)
prediction.index = test_df.query('District == 2').groupby('timestamp').mean().index
prediction


fig, ax = plt.subplots() 
train_df.query('District == 2').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
test_df.query('District == 2').groupby('timestamp').mean().plot(ax=ax)
prediction.plot()#xlim=['2015-01-01','2020-10-10'])
#test_df.query('District == 1')['Number of Crimes'].plot()
#prediction.plot(xlim=['2015-01-01','2020-10-10']) 


# In[193]:


model_df = df_True_district.query("timestamp >= '2004-01-01'  & timestamp <= '2020-10-10'")


# In[194]:


crime_model = ExponentialSmoothing(model_df.query('District == 2').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=18).fit()

crime_forecast2 = crime_model.forecast(12)
# Print original and then our prediction
#df_True_district[df_True_district.index.get_level_values('District')==1].plot(figsize=(12,6))
#crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
 

#fig, ax = plt.subplots() 
df_True_district.query('District == 2').groupby('timestamp').mean().plot(figsize=(12,6))
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast2.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:True')
plt.xlabel('Years')


# In[ ]:





# ## District 3

# In[ ]:





# In[195]:


fig, ax = plt.subplots() 
df_True_district.query('District == 3').groupby('timestamp').mean()['Number of Crimes'].plot(ax=ax, figsize = (12,6), legend = False)
df_True_district.query('District == 3').groupby('timestamp').mean()['Number of Crimes'].rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)


# In[196]:


fit_district_model = ExponentialSmoothing(train_df.query('District == 3').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 18).fit()

prediction = fit_district_model.forecast(45)
prediction.index = test_df.query('District == 3').groupby('timestamp').mean().index
prediction


fig, ax = plt.subplots() 
train_df.query('District == 3').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
test_df.query('District == 3').groupby('timestamp').mean().plot(ax=ax)
prediction.plot()#xlim=['2015-01-01','2020-10-10'])
#test_df.query('District == 1')['Number of Crimes'].plot()
#prediction.plot(xlim=['2015-01-01','2020-10-10']) 


# In[197]:


model_df = df_True_district.query("timestamp >= '2004-01-01'  & timestamp <= '2020-10-10'")


# In[198]:


crime_model = ExponentialSmoothing(model_df.query('District == 3').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=18).fit()

crime_forecast3 = crime_model.forecast(12)
# Print original and then our prediction
#df_True_district[df_True_district.index.get_level_values('District')==1].plot(figsize=(12,6))
#crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
 

#fig, ax = plt.subplots() 
df_True_district.query('District == 3').groupby('timestamp').mean().plot(figsize=(12,6))
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast3.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:True')
plt.xlabel('Years')


# ## District 4

# In[ ]:





# In[199]:


fig, ax = plt.subplots() 
df_True_district.query('District == 4').groupby('timestamp').mean()['Number of Crimes'].plot(ax=ax, figsize = (12,6), legend = False)
df_True_district.query('District == 4').groupby('timestamp').mean()['Number of Crimes'].rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)


# In[200]:


fit_district_model = ExponentialSmoothing(train_df.query('District == 4').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 24).fit()

prediction = fit_district_model.forecast(45)
prediction.index = test_df.query('District == 4').groupby('timestamp').mean().index
prediction


fig, ax = plt.subplots() 
train_df.query('District == 4').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
test_df.query('District == 4').groupby('timestamp').mean().plot(ax=ax)
prediction.plot()#xlim=['2015-01-01','2020-10-10'])
#test_df.query('District == 1')['Number of Crimes'].plot()
#prediction.plot(xlim=['2015-01-01','2020-10-10']) 


# In[201]:


model_df = df_True_district.query("timestamp >= '2004-01-01'  & timestamp <= '2020-10-10'")


# In[202]:


crime_model = ExponentialSmoothing(model_df.query('District == 4').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=18).fit()

crime_forecast4 = crime_model.forecast(12)
# Print original and then our prediction
#df_True_district[df_True_district.index.get_level_values('District')==1].plot(figsize=(12,6))
#crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
 

#fig, ax = plt.subplots() 
df_True_district.query('District == 4').groupby('timestamp').mean().plot(figsize=(12,6))
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast4.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:True')
plt.xlabel('Years')


# ## District 5

# 

# In[203]:


fig, ax = plt.subplots() 
df_True_district.query('District == 5').groupby('timestamp').mean()['Number of Crimes'].plot(ax=ax, figsize = (12,6), legend = False)
df_True_district.query('District == 5').groupby('timestamp').mean()['Number of Crimes'].rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)


# In[204]:


fit_district_model = ExponentialSmoothing(train_df.query('District == 5').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 18).fit()

prediction = fit_district_model.forecast(45)
prediction.index = test_df.query('District == 5').groupby('timestamp').mean().index
prediction


fig, ax = plt.subplots() 
train_df.query('District == 5').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
test_df.query('District == 5').groupby('timestamp').mean().plot(ax=ax)
prediction.plot()#xlim=['2015-01-01','2020-10-10'])
#test_df.query('District == 1')['Number of Crimes'].plot()
#prediction.plot(xlim=['2015-01-01','2020-10-10']) 


# In[205]:


model_df = df_True_district.query("timestamp >= '2004-01-01'  & timestamp <= '2020-10-10'")


# In[206]:


crime_model = ExponentialSmoothing(model_df.query('District == 5').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=18).fit()

crime_forecast5 = crime_model.forecast(12)
# Print original and then our prediction
#df_True_district[df_True_district.index.get_level_values('District')==1].plot(figsize=(12,6))
#crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
 

#fig, ax = plt.subplots() 
df_True_district.query('District == 5').groupby('timestamp').mean().plot(figsize=(12,6))
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast5.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:True')
plt.xlabel('Years')


# In[ ]:





# ## Combined Districts : True

# In[207]:


fig, ax = plt.subplots() 
df_True_district.query('District == 1').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast1.plot(ax=ax)#xlim=['2015-01-01','2021-10-10'])


df_True_district.query('District == 2').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast2.plot(ax=ax)#xlim=['2015-01-01','2021-10-10'])


df_True_district.query('District == 3').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast3.plot(ax=ax)#xlim=['2015-01-01','2021-10-10'])


df_True_district.query('District == 4').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast4.plot(ax=ax)#xlim=['2015-01-01','2021-10-10'])


df_True_district.query('District == 5').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast5.plot(ax=ax)#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:True')
plt.xlabel('Years')

plt.legend(["District 1","District 2","District 3","District 4","District 5"])


# In[ ]:





# ## False Arrest Districts

# In[ ]:





# In[208]:


df_False_district = df[df["Arrest"]==False].groupby([pd.Grouper('District')]+ [pd.Grouper(level = 'timestamp', freq = 'M')]).size().to_frame()
df_False_district.columns = ["Number of Crimes"]
#df4['timestamp']= pd.to_datetime(df4['timestamp'])
df_False_district


# In[209]:


df_False_plot = df_False_district.query("District <= 5.0")
df_False_plot

plt.figure(figsize=(17,10))
sns.lineplot(data = df_False_plot,x = df_False_plot.index.get_level_values('timestamp'), 
             y = "Number of Crimes",
             hue = df_False_plot.index.get_level_values('District'),
             style = df_False_plot.index.get_level_values('District'),
             palette = "flag_r",
             size_order = "list",
             legend = "full",
             dashes=False)


# In[210]:


data_tr = df_False_district.query("timestamp >= '2004-01-01'  & timestamp <= '2016-12-31'")
data_test = df_False_district.query("timestamp >= '2017-01-01'  & timestamp <= '2020-10-10'")


# In[211]:


listOfDist = list(df_False_district.index.get_level_values('District').unique())
listOfDist


# In[ ]:





# In[212]:


train_df = []
for district in listOfDist:
    result = data_tr[data_tr.index.get_level_values('District') == district]
    train_df.append(result)

train_df = pd.concat(train_df)
train_df


# In[213]:


test_df = []
for district in listOfDist:
    result = data_test[data_test.index.get_level_values('District') == district]
    test_df.append(result)

test_df = pd.concat(test_df)
test_df


# In[ ]:





# ## District 1

# In[214]:


fig, ax = plt.subplots() 
df_False_district.query('District == 1').groupby('timestamp').mean()['Number of Crimes'].plot(ax=ax, figsize = (12,6), legend = False)
df_False_district.query('District == 1').groupby('timestamp').mean()['Number of Crimes'].rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)


# In[215]:


fit_district_model = ExponentialSmoothing(train_df.query('District == 1').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 12).fit()

prediction = fit_district_model.forecast(45)
prediction.index = test_df.query('District == 1').groupby('timestamp').mean().index
prediction


fig, ax = plt.subplots() 
train_df.query('District == 1').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
prediction.plot()#xlim=['2015-01-01','2020-10-10'])
#test_df.query('District == 1')['Number of Crimes'].plot()
#prediction.plot(xlim=['2015-01-01','2020-10-10'])      


# In[216]:


model_df = df_False_district.query("timestamp >= '2004-01-01'  & timestamp <= '2020-10-10'")


# In[217]:


crime_model = ExponentialSmoothing(model_df.query('District == 1').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=12).fit()

crime_forecast1 = crime_model.forecast(12)
# Print original and then our prediction
#df_True_district[df_True_district.index.get_level_values('District')==1].plot(figsize=(12,6))
#crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
 

#fig, ax = plt.subplots() 
df_False_district.query('District == 1').groupby('timestamp').mean().plot(figsize=(12,6))
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast1.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:False')
plt.xlabel('Years')


# In[ ]:





# In[ ]:





# ## District 2

# In[218]:


fig, ax = plt.subplots() 
df_False_district.query('District == 2').groupby('timestamp').mean()['Number of Crimes'].plot(ax=ax, figsize = (12,6), legend = False)
df_False_district.query('District == 2').groupby('timestamp').mean()['Number of Crimes'].rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)


# In[219]:


fit_district_model = ExponentialSmoothing(train_df.query('District == 2').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 12).fit()

prediction = fit_district_model.forecast(45)
prediction.index = test_df.query('District == 2').groupby('timestamp').mean().index
prediction


fig, ax = plt.subplots() 
train_df.query('District == 2').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
test_df.query('District == 2').groupby('timestamp').mean().plot(ax=ax)
prediction.plot()#xlim=['2015-01-01','2020-10-10'])
#test_df.query('District == 1')['Number of Crimes'].plot()
#prediction.plot(xlim=['2015-01-01','2020-10-10']) 


# In[220]:


model_df = df_False_district.query("timestamp >= '2004-01-01'  & timestamp <= '2020-10-10'")


# In[221]:


crime_model = ExponentialSmoothing(model_df.query('District == 2').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=12).fit()

crime_forecast2 = crime_model.forecast(12)
# Print original and then our prediction
#df_True_district[df_True_district.index.get_level_values('District')==1].plot(figsize=(12,6))
#crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
 

#fig, ax = plt.subplots() 
df_False_district.query('District == 2').groupby('timestamp').mean().plot(figsize=(12,6))
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast2.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:False')
plt.xlabel('Years')


# In[ ]:





# ## District 3

# In[ ]:





# In[222]:


fig, ax = plt.subplots() 
df_False_district.query('District == 3').groupby('timestamp').mean()['Number of Crimes'].plot(ax=ax, figsize = (12,6), legend = False)
df_False_district.query('District == 3').groupby('timestamp').mean()['Number of Crimes'].rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)


# In[223]:


fit_district_model = ExponentialSmoothing(train_df.query('District == 3').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 24).fit()

prediction = fit_district_model.forecast(45)
prediction.index = test_df.query('District == 3').groupby('timestamp').mean().index
prediction


fig, ax = plt.subplots() 
train_df.query('District == 3').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
test_df.query('District == 3').groupby('timestamp').mean().plot(ax=ax)
prediction.plot()#xlim=['2015-01-01','2020-10-10'])
#test_df.query('District == 1')['Number of Crimes'].plot()
#prediction.plot(xlim=['2015-01-01','2020-10-10']) 


# In[224]:


model_df = df_False_district.query("timestamp >= '2004-01-01'  & timestamp <= '2020-10-10'")


# In[225]:


crime_model = ExponentialSmoothing(model_df.query('District == 3').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=24).fit()

crime_forecast3 = crime_model.forecast(12)
# Print original and then our prediction
#df_True_district[df_True_district.index.get_level_values('District')==1].plot(figsize=(12,6))
#crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
 

#fig, ax = plt.subplots() 
df_False_district.query('District == 3').groupby('timestamp').mean().plot(figsize=(12,6))
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast3.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:False')
plt.xlabel('Years')


# ## District 4

# In[ ]:





# In[226]:


fig, ax = plt.subplots() 
df_False_district.query('District == 4').groupby('timestamp').mean()['Number of Crimes'].plot(ax=ax, figsize = (12,6), legend = False)
df_False_district.query('District == 4').groupby('timestamp').mean()['Number of Crimes'].rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)


# In[227]:


fit_district_model = ExponentialSmoothing(train_df.query('District == 4').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 24).fit()

prediction = fit_district_model.forecast(45)
prediction.index = test_df.query('District == 4').groupby('timestamp').mean().index
prediction


fig, ax = plt.subplots() 
train_df.query('District == 4').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
test_df.query('District == 4').groupby('timestamp').mean().plot(ax=ax)
prediction.plot()#xlim=['2015-01-01','2020-10-10'])
#test_df.query('District == 1')['Number of Crimes'].plot()
#prediction.plot(xlim=['2015-01-01','2020-10-10']) 


# In[228]:


model_df = df_False_district.query("timestamp >= '2004-01-01'  & timestamp <= '2020-10-10'")


# In[229]:


crime_model = ExponentialSmoothing(model_df.query('District == 4').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=24).fit()

crime_forecast4 = crime_model.forecast(12)
# Print original and then our prediction
#df_True_district[df_True_district.index.get_level_values('District')==1].plot(figsize=(12,6))
#crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
 

#fig, ax = plt.subplots() 
df_False_district.query('District == 4').groupby('timestamp').mean().plot(figsize=(12,6))
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast4.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:False')
plt.xlabel('Years')


# ## District 5

# In[ ]:





# In[230]:


fig, ax = plt.subplots() 
df_False_district.query('District == 5').groupby('timestamp').mean()['Number of Crimes'].plot(ax=ax, figsize = (12,6), legend = False)
df_False_district.query('District == 5').groupby('timestamp').mean()['Number of Crimes'].rolling(window=30).mean().plot(ax=ax,figsize = (12,6) , legend = False)


# In[231]:


fit_district_model = ExponentialSmoothing(train_df.query('District == 5').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods= 12).fit()

prediction = fit_district_model.forecast(45)
prediction.index = test_df.query('District == 5').groupby('timestamp').mean().index
prediction


fig, ax = plt.subplots() 
train_df.query('District == 5').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
test_df.query('District == 5').groupby('timestamp').mean().plot(ax=ax)
prediction.plot()#xlim=['2015-01-01','2020-10-10'])
#test_df.query('District == 1')['Number of Crimes'].plot()
#prediction.plot(xlim=['2015-01-01','2020-10-10']) 


# In[232]:


model_df = df_False_district.query("timestamp >= '2004-01-01'  & timestamp <= '2020-10-10'")


# In[233]:


crime_model = ExponentialSmoothing(model_df.query('District == 5').groupby('timestamp').mean()['Number of Crimes'],
                                  trend='add',
                                  seasonal='add',
                                  seasonal_periods=24).fit()

crime_forecast5 = crime_model.forecast(12)
# Print original and then our prediction
#df_True_district[df_True_district.index.get_level_values('District')==1].plot(figsize=(12,6))
#crime_forecast.plot()#xlim=['2015-01-01','2021-10-10'])
 

#fig, ax = plt.subplots() 
df_False_district.query('District == 5').groupby('timestamp').mean().plot(figsize=(12,6))
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast5.plot()#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:False')
plt.xlabel('Years')


# ## Combined District : False

# In[234]:


fig, ax = plt.subplots() 
df_False_district.query('District == 1').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast1.plot(ax=ax)#xlim=['2015-01-01','2021-10-10'])


df_False_district.query('District == 2').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast2.plot(ax=ax)#xlim=['2015-01-01','2021-10-10'])


df_False_district.query('District == 3').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast3.plot(ax=ax)#xlim=['2015-01-01','2021-10-10'])


df_False_district.query('District == 4').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast4.plot(ax=ax)#xlim=['2015-01-01','2021-10-10'])


df_False_district.query('District == 5').groupby('timestamp').mean().plot(figsize=(12,6),ax=ax)
#test_df.query('District == 1').groupby('timestamp').mean().plot(ax=ax)
crime_forecast5.plot(ax=ax)#xlim=['2015-01-01','2021-10-10'])
plt.ylabel('Number of Arrests:False')
plt.xlabel('Years')

plt.legend(["District 1","District 2","District 3","District 4","District 5"])


# In[ ]:





# In[ ]:




