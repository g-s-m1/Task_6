# Import the libraries 

#Dataframe/Numerical libraries
import pandas as pd 
import numpy as np

#Data visualization 
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#Import the libraries to train the model 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 


#Reading the data
path='boston.csv'
housing_df=pd.read_csv(path,header=None,delim_whitespace=True)

'''columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

housing_df.columns=columns
print(housing_df)'''

# Check if there is any missing values.
housing_df.isna().sum()
#To check the data type of each columns
housing_df.info()
housing_df.describe()
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] =14
matplotlib.rcParams['figure.figsize']= (10,6)
matplotlib.rcParams['figure.facecolor'] ='#00000000'
fig,ax=plt.subplots(2,7,figsize=(100,30))
j=0
d=0
part1=housing_df.columns[0:7]
part2=housing_df.columns[7:14]

for k in part2:
    for i in part1:
        if j<7:
            ax[0,j].set_title(i)
            sns.histplot(data=housing_df,x=i,ax=ax[0,j])
            j+=1
    if d<7:
        ax[1,d].set_title(k)
        sns.histplot(data=housing_df,x=k,ax=ax[1,d])
        d+=1

new_df=housing_df[housing_df['TAX']<600]
new_df.shape
fig,ax=plt.subplots(2,7,figsize=(100,30))
j=0
d=0
part1=new_df.columns[0:7]
part2=new_df.columns[7:14]

for k in part2:
    for i in part1:
        if j<7:
            ax[0,j].set_title(i)
            sns.histplot(data=new_df,x=i,ax=ax[0,j])
            j+=1
    if d<7:
        ax[1,d].set_title(k)
        sns.histplot(data=new_df,x=k,ax=ax[1,d])
        d+=1
        
# correlation 
plt.figure(figsize=(18,18))
plt.title('Correlation Matrix')
sns.heatmap(new_df.corr(),annot=True)
high_corr=new_df.corr().MEDV.sort_values(ascending=False) #Create a correlation matrix
high_corr_var=[] # create a variable to store the name

# loop through to the name
for name in high_corr.index:
    high_corr_var.append(name)# store the name into the variable 

high_corr_var.pop(0) # remove the 'MEDV' from the list
# high_corr_var
sns.pairplot(data=housing_df,x_vars=['RM','ZN','B','CHAS','RAD'],y_vars='MEDV',height=7,aspect=.7);
#Create a list of features
cols_inputs=list(housing_df.columns)
cols_inputs.pop()
# df.drop(columns='MEDV', inplace=True)
X=new_df[cols_inputs]
y=new_df['MEDV']

def standard(X):
    '''Standard makes the feature 'X' have a zero mean'''
    mu=np.mean(X) #mean
    std=np.std(X) #standard deviation
    sta=(X-mu)/std # mean normalization
    return mu,std,sta 
    
mu,std,sta=standard(X)
X=sta
print(X)

X=new_df[['RM']]
mu,std,sta=standard(X)
print(X)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)

#Create and Train the model
model=LinearRegression().fit(X_train,y_train)

#Generate prediction
predictions_test=model.predict(X_test)
#Compute loss to evaluate the model
coefficient= model.coef_
intercept=model.intercept_
print(coefficient,intercept)

print(high_corr_var)
