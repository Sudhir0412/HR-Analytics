
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import pycaret


# In[3]:


#!pip show numpy


# In[4]:


#import train and test data
train_hr=pd.read_csv("E://Analytics_Vidhya_Hackathon/HR_Analytics/train_hr.csv",sep=",")
test_hr=pd.read_csv("E://Analytics_Vidhya_Hackathon/HR_Analytics/test_hr.csv",sep=",")


# In[5]:


#from pycaret.classification import *


# In[6]:


#!pip install --upgrade pip=20.1.1 --user


# In[7]:


#!pip install --upgrade numpy==1.16.1 --user


# In[8]:


#df = setup(train_hr, target = 'is_promoted') #Setup dataset to check model performance using pycaret


# In[9]:


#compare_models() #Check model performance using pycaret so we can choos best fit model accordingly.


# In[10]:


train_hr.keys()


# In[11]:


train_hr.head() #Train data


# In[12]:


test_hr.head() #Test data


# In[13]:


train_hr.info()


# In[14]:


train_hr.isna().sum()


# In[15]:


train_hr.shape


# In[16]:


test_hr.isna().sum()


# In[17]:


#!pip install missingno==0.4.2 --user


# In[18]:


#missingno.matrix(train_hr,figsize=(30,10))


# In[19]:


train_hr.head()


# In[20]:


train_hr.describe()


# In[21]:


train_hr.info()


# In[22]:


train_hr.columns


# # Check unique values for categorical data

# In[23]:



print(train_hr.department.unique())


# In[24]:


print(train_hr.region.unique())


# In[25]:


train_hr['education'].unique()


# In[26]:


train_hr.recruitment_channel.unique()


# In[27]:


train_hr.age.unique() #Check the unique values for age


# In[28]:


train_hr.length_of_service.unique()


# In[29]:


train_hr.avg_training_score.unique()


# In[30]:


train_hr.no_of_trainings.unique()


# # Check data with gropuby

# In[31]:


train_hr[(train_hr['region']=='region_2')] #Check the details about region_2 because it's higly promoted region.


# In[32]:


p=train_hr.groupby(['region'])['is_promoted'].count().sort_values(ascending=False) #Check average value promoted by region
p/54808


# In[33]:


train_hr.groupby(['region'])['is_promoted'].count().plot(kind='barh',figsize=(20,30))


# In[34]:


#Check the promotion region wise
train_hr.groupby(['is_promoted'])['region'].value_counts()[1].sort_values(ascending=True).plot(kind='barh',figsize=(20,30))


# In[35]:


#With the help of this code we can take top 10 regions those are promoted more employee & 
# we make new region as 'Other' to store remaining regions
Top10_rgn_promoted=train_hr.groupby(['is_promoted'])['region'].value_counts()[1].head(10)


# In[36]:


p = pd.DataFrame(Top10_rgn_promoted)
p/54508


# In[37]:


p.plot.pie(y='region',autopct='%1.0f%%',figsize=(10,10)) #Visualize Top 10 regions for promoted employee


# # Create a new category 'Other' for remaining regions except above top 10 regions.

# In[38]:


other_region=['region_24','region_12','region_9','region_21','region_3','region_34','region_33','region_18']
train_hr['region'] = train_hr['region'].replace(other_region,'Others')


# In[39]:


train_hr['region'].value_counts()


# In[40]:


#Store Top10_rgn_promoted in rgn
rgn=pd.DataFrame({'region':['region_2','region_22','region_7','region_4','region_13','region_15','region_28','region_26','region_23','region_27']})
rgn


# In[41]:


train_hr[(train_hr['previous_year_rating']>=5.0)] #Check the previous year rating 


# In[42]:


#Check promoted employee count previous_year_rating wise
#train_hr.groupby(['previous_year_rating'])['is_promoted'].value_counts().sort_values(ascending=False) 
train_hr.groupby(['is_promoted'])['previous_year_rating'].value_counts()


# In[43]:


#Check promoted employee length of service wise
train_hr.groupby(['is_promoted'])['length_of_service'].value_counts()[1].sort_values(ascending=True).plot(kind='barh',figsize=(20,30)) 


# In[44]:


train_hr.groupby(['is_promoted'])['length_of_service'].value_counts()[1].sort_values(ascending=False)


# # Working for missing data

# In[45]:


numeric_data = train_hr.select_dtypes(include=[np.number])
categorical_data = train_hr.select_dtypes(exclude=[np.number])


# In[46]:


numeric_data.shape


# In[47]:


categorical_data.shape


# In[48]:


#train_hr=train_hr.interpolate()
#train_hr['previous_year_rating'].fillna((train_hr['previous_year_rating'].mean()), inplace=True)


# In[49]:


#Missing values replacement for categorical data
# for each column, get value counts in decreasing order and take the index (value) of most common class
df_most_common_imputed = categorical_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
df_most_common_imputed.isnull().sum()


# In[50]:


#Replace missing values for numerical variable
numeric_data['previous_year_rating'].fillna((numeric_data['previous_year_rating'].mean()), inplace=True)
df_numeric_imputed=numeric_data


# In[51]:


#Create new df as final_df
final_df=pd.concat([df_numeric_imputed,df_most_common_imputed],axis=1)


# In[52]:


final_df.head()


# In[53]:


final_df.shape


# In[54]:


final_df.isnull().sum()


# # Outlier detection

# In[55]:


final_df.boxplot('length_of_service')


# In[56]:


##Check promoted employee count less than 13 years of experience
final_df[(final_df['length_of_service']>=13)&(final_df['is_promoted']==1)].count()


# In[57]:


##Check promoted employee count less than 13 years of experience
final_df[(final_df['length_of_service']<=13)&(final_df['is_promoted']==1)].count()


# In[58]:


Q1 = final_df['length_of_service'].quantile(0.25);Q1
Q3 = final_df['length_of_service'].quantile(0.75);Q3
IQR = Q3 - Q1    #IQR is interquartile range. 
Lower_q= (Q1 - 1.5 * IQR) 
Upper_q= (Q3 + 1.5 *IQR)

final_df['length_of_service'] = np.where(final_df['length_of_service'] >= Upper_q,Q3,final_df['length_of_service'])
final_df['length_of_service'] = np.where(final_df['length_of_service'] <= Lower_q,Q1,final_df['length_of_service'])


# In[59]:


final_df.boxplot('previous_year_rating')


# In[60]:


Q1 = final_df['previous_year_rating'].quantile(0.25);Q1
Q3 = final_df['previous_year_rating'].quantile(0.75);Q3
IQR = Q3 - Q1    #IQR is interquartile range. 
Lower_q= (Q1 - 1.5 * IQR) 
Upper_q= (Q3 + 1.5 *IQR)

final_df['previous_year_rating'] = np.where(final_df['previous_year_rating'] >= Upper_q,Q3,final_df['previous_year_rating'])
final_df['previous_year_rating'] = np.where(final_df['previous_year_rating'] <= Lower_q,Q1,final_df['previous_year_rating'])


# # Feature Engineering

# In[61]:


final_df['age'].unique()


# In[62]:


final_df['age_groups'] = pd.cut(x=final_df['age'], bins=[18,30,50,60,70])


# In[63]:


final_df['length_of_service_group'] = pd.cut(x=final_df['length_of_service'], bins=[0,3,5,7,10,15])


# In[64]:


final_df['avg_training_score_group']=pd.cut(x=final_df['avg_training_score'], bins=[20,40,60,80,100])


# In[65]:


final_df.isna().sum()


# # Create new dataframe 

# In[66]:


new_train=pd.DataFrame(final_df)


# In[67]:


new_train.head()


# In[68]:


#Remove unnecessory & duplicate features
new_train.drop(['gender','recruitment_channel','age','length_of_service','avg_training_score'],axis=1,inplace=True)


# In[69]:


new_train.shape


# In[70]:


new_train.groupby(['length_of_service_group'])['is_promoted'].value_counts().sort_values(ascending=False)


# # Target variable

# In[71]:


#Seperate target variable from dataset
y=new_train.pop('is_promoted') 


# In[72]:


y.shape


# # Dummy variables

# In[73]:


new_train.info()


# In[74]:


new_train=pd.get_dummies(new_train)


# In[75]:


new_train.shape


# # Checking correlation

# In[76]:


corr_matrix=new_train.corr().abs()
    


# In[77]:


# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))


# In[78]:


# Find index of feature columns with correlation greater than 50%
to_drop = [column for column in upper.columns if any(upper[column] > 0.60)]


# In[79]:


len(to_drop)


# In[80]:


to_drop


# In[81]:


#get correlations of each features in dataset
#corrmat = new_train.corr()
#top_corr_features = corrmat.index
#plt.figure(figsize=(25,25))
#plot heat map
#g=sns.heatmap(new_train[top_corr_features].corr(),annot=True,cmap="YlGnBu")

mtrx=new_train.corr()
mtrx = pd.DataFrame(mtrx)
pd.options.display.max_columns = None
mtrx.head()


# In[82]:


# Drop features 
new_train=new_train.drop(new_train[to_drop], axis=1)


# In[83]:


new_train.shape


# In[84]:


#Remove column 'age_groups_(60, 70] which is only contating null values
new_train.drop(['age_groups_(60, 70]'],inplace=True,axis=1) 


# In[85]:


new_train.head()


# In[86]:


#new_train store to X
X=new_train


# # Train-Test split

# In[93]:


X = new_train
y = y


# In[94]:


print(X.shape)
print(y.shape)


# # Feature selection

# # 1. Find Top 10 best features with SelectKBest function of sklearn

# In[232]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[238]:


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[239]:


#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Columns','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


# # 2.Find best features with ExtraTreesClassifier

# In[240]:


from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()


# In[241]:


model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()


# In[242]:


feat_importances.nlargest(10)


# # 3. Correlation matrix with Heat_map

# In[243]:


import seaborn as sns


# In[244]:


data=X
data['is_promoted']=y


# In[246]:


#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(25,30))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="YlGnBu")


# # Final Train Test data for model building

# In[247]:


Df_train=new_train[["employee_id","awards_won?","KPIs_met >80%","previous_year_rating","region_region_4",
"region_region_22","department_Technology","department_Sales & Marketing","region_region_7",
"region_region_17","no_of_trainings","age_groups_(18, 30]","education_Bachelor's","length_of_service_group_(0, 3]",
"length_of_service_group_(3, 5]","length_of_service_group_(5, 7]"]]


# In[248]:


X1=Df_train
y1=y


# In[249]:


X1.shape


# # Data cleaning for test data

# In[250]:


test_hr.head()


# In[251]:


test_hr.isnull().sum()


# In[252]:


#Seperate continuous & ctegorial data
num_var = test_hr.select_dtypes(include=[np.number])
cat_var = test_hr.select_dtypes(exclude=[np.number])


# In[253]:


num_var.shape


# In[254]:


cat_var.shape


# In[255]:


#Missing values replacement for categorical data
# for each column, get value counts in decreasing order and take the index (value) of most common class
test_cat = cat_var.apply(lambda x: x.fillna(x.value_counts().index[0]))
test_cat.isnull().sum()


# In[256]:


#Replace missing values for numerical variable
num_var['previous_year_rating'].fillna((num_var['previous_year_rating'].mean()), inplace=True)
test_num=num_var


# In[257]:


test_num.isnull().sum()


# In[258]:


#Create new df as final_df
final_test=pd.concat([test_num,test_cat],axis=1)


# In[259]:


final_test.head()


# In[260]:


final_test.boxplot('length_of_service')


# In[261]:


Q1 = final_test['length_of_service'].quantile(0.25);Q1
Q3 = final_test['length_of_service'].quantile(0.75);Q3
IQR = Q3 - Q1    #IQR is interquartile range. 
Lower_q= (Q1 - 1.5 * IQR) 
Upper_q= (Q3 + 1.5 *IQR)

final_test['length_of_service'] = np.where(final_test['length_of_service'] >= Upper_q,Q3,final_test['length_of_service'])
final_test['length_of_service'] = np.where(final_test['length_of_service'] <= Lower_q,Q1,final_test['length_of_service'])


# In[262]:


final_test.boxplot('previous_year_rating')


# In[263]:


Q1 = final_test['previous_year_rating'].quantile(0.25);Q1
Q3 = final_test['previous_year_rating'].quantile(0.75);Q3
IQR = Q3 - Q1    #IQR is interquartile range. 
Lower_q= (Q1 - 1.5 * IQR) 
Upper_q= (Q3 + 1.5 *IQR)

final_test['previous_year_rating'] = np.where(final_test['previous_year_rating'] >= Upper_q,Q3,final_test['previous_year_rating'])
final_test['previous_year_rating'] = np.where(final_test['previous_year_rating'] <= Lower_q,Q1,final_test['previous_year_rating'])


# In[264]:


final_test['age_groups'] = pd.cut(x=final_test['age'], bins=[18,30,50,60,70])
final_test['length_of_service_group'] = pd.cut(x=final_test['length_of_service'], bins=[0,3,5,7,10,15])
final_test['avg_training_score_group']=pd.cut(x=final_test['avg_training_score'], bins=[20,40,60,80,100])


# In[265]:


final_test.head()


# In[266]:


#New cleaned test dataframe
new_test=pd.DataFrame(final_test)
new_test.drop(['gender','recruitment_channel','age','length_of_service','avg_training_score'],axis=1,inplace=True)


# In[267]:


new_test.head()


# In[268]:


#Dummy variables for test data
new_test=pd.get_dummies(new_test)


# In[269]:


Df_test=new_test[["employee_id","awards_won?","KPIs_met >80%","previous_year_rating","region_region_4",
"region_region_22","department_Technology","department_Sales & Marketing","region_region_7",
"region_region_17","no_of_trainings","age_groups_(18, 30]","education_Bachelor's","length_of_service_group_(0, 3]",
"length_of_service_group_(3, 5]","length_of_service_group_(5, 7]"]]


# In[270]:


X_test1=Df_test


# # Model building

# # Logistic regression

# In[271]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[272]:


X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.3,random_state=0)


# In[273]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[274]:


model=LogisticRegression()


# In[275]:


model_train=model.fit(X_train,y_train)


# In[279]:


model_train.score(X_train,y_train) #Train score


# In[280]:


model_train.score(X_test,y_test) #Test score


# In[317]:


pred=model_train.predict(X_test)


# In[318]:


cm=confusion_matrix(y_test,pred)
cm


# In[287]:


#sample = pd.DataFrame({'Predicted': pred, 'Actual_data': y_test})
# you could use any filename. We choose submission here
#sample.to_csv("E://Analytics_Vidhya_Hackathon/HR_Analytics/confusion_mtrx.csv", index=False)


# In[288]:


model_train.coef_


# In[289]:


model.intercept_


# In[319]:


cm=confusion_matrix(y_test,pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[293]:


y_test.value_counts() #Check actual 0 & 1


# # Submission test data

# In[294]:


submition=model_train.predict(X_test1)


# In[296]:


my_submission = pd.DataFrame({'Id': X_test1['employee_id'], 'is_promoted': submition})
# you could use any filename. We choose submission here
my_submission.to_csv("E://Analytics_Vidhya_Hackathon/HR_Analytics/Logistic.csv", index=False)


# # Random forest

# In[297]:


from sklearn.ensemble import RandomForestClassifier


# In[298]:


rf=RandomForestClassifier(n_estimators=100)


# In[299]:


rf_model=rf.fit(X_train,y_train)


# In[300]:


rf_model.score(X_train,y_train) #Train model


# In[301]:


rf_model.score(X_test,y_test) #Check accuracy for testing data.. find model overfitted


# In[302]:


pred_rf=rf_model.predict(X_test)


# In[309]:


cm_rf=confusion_matrix(y_test,pred_rf)
conf_matrix=pd.DataFrame(data=cm_rf,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[303]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, pred_rf))


# In[306]:


RF_submission=rf_model.predict(X_test1)


# In[307]:


my_submission = pd.DataFrame({'employee_id': X_test1['employee_id'], 'is_promoted': RF_submission})
# you could use any filename. We choose submission here
my_submission.to_csv("E://Analytics_Vidhya_Hackathon/HR_Analytics/RF_submission.csv", index=False)


# In[135]:


feature_imp = pd.Series(rf_model.feature_importances_,index=X_test1.columns).sort_values(ascending=False)
feature_imp


# In[136]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# # Naive Bayes model

# In[137]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()


# In[138]:


nb_model=nb.fit(X_train,y_train)


# In[139]:


nb_pred=nb_model.predict(X_test)


# In[140]:


nb_model.score(X_train,y_train)


# In[141]:


print("Accuracy:",metrics.accuracy_score(y_test, nb_pred))


# In[142]:


confusion_matrix(y_test,nb_pred)


# # Boosting model

# In[321]:


from sklearn.ensemble import GradientBoostingClassifier


# In[322]:


gb=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)


# In[323]:


gb_model=gb.fit(X_train,y_train)


# In[324]:


gb_model.score(X_train,y_train) #Train model


# In[325]:


gb_model.score(X_test,y_test) #Test model


# In[328]:


gb_pred=gb_model.predict(X_test)


# In[330]:


gb_cm=confusion_matrix(y_test,gb_pred)
conf_matrix=pd.DataFrame(data=gb_cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[331]:


(14984+67)/(14984+67+1319+73) #Manual score checking


# In[332]:


gb_pred_test=gb_model.predict(X_test1)


# In[333]:


my_submission = pd.DataFrame({'employee_id': X_test1['employee_id'], 'is_promoted': gb_pred_test})
# you could use any filename. We choose submission here
my_submission.to_csv("E://Analytics_Vidhya_Hackathon/HR_Analytics/GB_submission.csv", index=False)


# # KNN-model

# In[347]:


from sklearn.neighbors import KNeighborsClassifier


# In[348]:


kn=KNeighborsClassifier(n_neighbors=3)


# In[349]:


kn_model=kn.fit(X_train,y_train) 


# In[350]:


kn_model.score(X_train,y_train) #Train model


# In[351]:


kn_model.score(X_test,y_test) #Test model


# In[352]:


kn_pred=kn_model.predict(X_test)


# In[353]:


knn_cm=confusion_matrix(y_test,kn_pred)
conf_matrix=pd.DataFrame(data=knn_cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")


# In[354]:


kn_pred=kn_model.predict(X_test1)


# In[355]:


my_submission = pd.DataFrame({'employee_id': X_test1['employee_id'], 'is_promoted': kn_pred})
# you could use any filename. We choose submission here
my_submission.to_csv("E://Analytics_Vidhya_Hackathon/HR_Analytics/KNN_submission.csv", index=False)

