## IMPORTING ALL THE LIBRARIES

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score 
import seaborn as sns


## IMPORTING THE DATA SET

spam = pd.read_csv('path of the dataset file')
spam .head()



## DATA PREPROCESSING ---CHECKING FOR NULL VALUES IN THE DATASET

spam .isnull()


# In[6]:


## DATA PREPROCESSING ---REMOVING THE COLUMNS WITH NULL VALUES

spam .drop("Unnamed: 2",axis=1,inplace=True)
spam .drop("Unnamed: 3",axis=1,inplace=True)
spam .drop("Unnamed: 4",axis=1,inplace=True)
spam .head()


# In[7]:


## FINDING THE DIMENSIONS OF THE DATASET AFTER REMOVING THE NULL COLUMNS

spam .shape 


# In[8]:


## CLASSIFYING AND ASSIGNING 0 TO SPAM MAIL AND 1 TO NON-SPAM MAIL

spam.loc[spam['category']=='spam','category']=0 
spam.loc[spam['category']=='ham','category']=1
spam 


# In[10]:


## LOCATING THE DEPENDANT AND INDEPENDANT VARIABLES

X=spam ['messaage']
Y=spam ['category']
print("X data is : \n ",X)
print("Y data is : \n ",Y)


# In[11]:


## TRAINING AND TESTING WITH RATIO=8:2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)  


# In[12]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[13]:


# xtrain and x test data are in strings and need to be converted to numerical data so we need to convert using tfidf vectorisation

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers because some values are in strings 

Y_train = Y_train.astype('float')
Y_test = Y_test.astype('float')
print("The X data is :\n",X_train)
print()
print("The converted X data is :\n",X_train_features)


# In[14]:


s= StandardScaler(with_mean=False)    
x_train= s.fit_transform(X_train_features)    
x_test= s.transform(X_test_features) 


# In[15]:


## KNN 
knn = KNeighborsClassifier(n_neighbors=9, metric='euclidean')  
knn.fit(x_train, Y_train)
pred_knn= knn.predict(x_test)


# In[16]:


accuracy_knn= accuracy_score(Y_test, pred_knn)
print('Accuracy of KNN : ', accuracy_knn)

pre_knn=precision_score(Y_test, pred_knn)
print("Precison Score of KNN :",pre_knn)

rs_knn=recall_score(Y_test, pred_knn)
print("Recall Score of KNN :",rs_knn)

cm_knn = confusion_matrix(Y_test, pred_knn)
print("Confusion Matrix of KNN : \n",cm_knn)


# In[17]:


ax = sns.heatmap(cm_knn, annot=True, cmap='coolwarm')
ax.set_title('Confusion Matrix for Knn\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
plt.show()


# In[18]:


## SVM
svm_model=svm.SVC(kernel="linear")
svm_model.fit(X_train_features,Y_train)
pred_svm=svm_model.predict(X_test_features)


# In[19]:


accuracy_svm=accuracy_score(Y_test,pred_svm)
print("accuracy of svm is  : ",accuracy_svm)

pre_svm=precision_score(Y_test, pred_svm)
print("Precison Score of svm :",pre_svm)

rs_svm=recall_score(Y_test, pred_svm)
print("Recall Score of svm :",rs_svm)

cm_svm = confusion_matrix(Y_test, pred_svm)
print("Confusion Matrix of svm : \n",cm_svm)


# In[20]:


ax = sns.heatmap(cm_svm, annot=True, cmap='coolwarm')
ax.set_title('Confusion Matrix for SVM\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
plt.show()


# In[21]:


## DECISION TREE
classifier = DecisionTreeClassifier(criterion='entropy',random_state = 0)
classifier.fit(x_train, Y_train)
pred_dt = classifier.predict(X_test_features)


# In[22]:


accuracy_dt=accuracy_score(Y_test,pred_dt)
print("accuracy of Decision tree is  : ",accuracy_dt)

pre_dt=precision_score(Y_test, pred_dt)
print("Precision Score of Decision tree :",pre_dt)

rs_dt=recall_score(Y_test, pred_dt)
print("Recall Score of Decision tree :",rs_dt)

cm_dt = confusion_matrix(Y_test, pred_dt)
print("Confusion Matrix of Decision tree : \n",cm_dt)


# In[23]:


ax = sns.heatmap(cm_dt, annot=True, cmap='coolwarm')
ax.set_title('Confusion Matrix for Decision tree\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
plt.show()


# In[24]:


mail_recieved_knn = ["FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, 螢1.50 to rcv"]
print("The mail message is :",mail_recieved_knn)
conv_data_knn = feature_extraction.transform(mail_recieved_knn)
prediction_knn = knn.predict(conv_data_knn)
print("The predicted value is :",prediction_knn)
if (prediction_knn[0]==1):
  print('The mail recieved is not a spam mail')
else:
  print('The mail recieved is Spam mail')


# In[25]:


a=spam['category'].value_counts()
a.plot(kind='bar',color=['red','blue'])
plt.title("Comparing the number of spam and non spam mails")
plt.xlabel("non spam                                  spam")
plt.ylabel("number of mails")
plt.show()


# In[26]:


figure,(axis) = plt.subplots()
models= np.array( ['KNN','SVM','DECISION TREE'])
accuracy=[accuracy_knn,accuracy_svm,accuracy_dt]
sns.set_theme(style="darkgrid")
sns.barplot(x=models, y=accuracy, palette="rocket",ax=axis)

