# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import time

print('program is running')
print()
start_time = time.time()
# Load dataset
# There are two datasets. First sheet is the dataset of Bank products and other
# sheet  is the CIBIL dataset.
a1= pd.read_excel(r"C:\Users\Admin\Desktop\Bindu\Projects All\Credit Risk Modelling Using Machine Learning\case_study1.xlsx")
a2= pd.read_excel(r"C:\Users\Admin\Desktop\Bindu\Projects All\Credit Risk Modelling Using Machine Learning\case_study2.xlsx")

# creating the copies of the dataset for proceeding further.
df1=a1.copy()
df2=a2.copy()

df1.describe()
df1.isnull().sum()

# Though it is not showing null values, the data has multiple values under 'Age_Oldest_TL'
# and 'Age_Newest_TL' as '-99999' which we can see as min using describe function. These are 
# null values which are not there in the system. As the null values are not big in number,
# we have decided to remove these from dataset.


# counting the number of nulls
count = (df1['Age_Oldest_TL'] == -99999).sum()
count1= (df1['Age_Newest_TL'] == -99999).sum()

# removing from one column and it makes changes in the complete sheet
df1=df1.loc[df1['Age_Oldest_TL'] != -99999]

# Alternatively, same could be done by 
# df1 = df1[df1['Age_Oldest_TL'] != -99999]

# dealing with null values in df2. 
# for df2, we are going to check how many null values do we have.
# if null values in a column is more than 10000, then we are deciding to remove the 
# column as it is any way not serving anything. If the values is less than 10000,
# then we are removing the rows. 
cols_for_removal = []

for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000 :
        cols_for_removal.append(i)

# dropping the columns collected in cols_for_removal
df2 = df2.drop(cols_for_removal, axis=1)

# checking number of null left in df2 now.

#for i in df2.columns:
#   df2 = df2[df2[i] != -99999]

for i in df2.columns:
    df2 = df2.loc[ df2[i] != -99999 ]


# Check null values in df2, df1

df2.isnull().sum()
df1.isnull().sum()

# Finding the common column in df1 and df2 to go ahead with merger of the files.
for i in list(df1.columns):
    if i in list(df2.columns):
        print(i)
    

# Merging the two files. i.e. df1 and df2. inner join to avoid any null values.

df = df1.merge(df2, how = 'inner', on = 'PROSPECTID')

df.info()
df.isnull().sum().sum()

# check the count of categorical columns
# cat_col= {}
# for i in df.columns:
#    if df[i].dtype == 'object':
#        j = df[i].value_counts()
#        cat_col[i]=j

# print(cat_col)
# len(cat_col)
# cat_col.keys()
# cat_col.values()
# cat_col.items()
        
for i in df.columns:
    if df[i].dtype == 'object':
        print(i)  
        
df.columns
# Find which feature is important. so that we can keep only relevant ones.


# chi- square test to find out the p -value for feature selection. It will see the
# relation b/w the 'Approved Flag' which is the target variable with each category.

# h0 - null hypothesis = not associated
# h1 - Alternate hypothesis = associated
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2,pval,_,_ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '-->', pval)
    
# The p value for all the categories is <=0.05. 
# It means we reject null hypothesis. and these are associated. 

# checking the numberical columns
num_col= []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID', 'Approved_Flag']:
        num_col.append(i)
    
print(num_col)   
len(num_col)

# Need to check Multicollinearity. 
# Multicollinearity is the concept wherein we try to find out if the set of 
# variables are associated with each other or if these can be predicted via other
# variables. To find out this we find the VIF value which is 
# Variance Inflation Factor
# VIF(i) = 1/(1-R square(i)) Rsquare varies from -infinity to 1.
# The closer it is to 1, closer is the relationship.

# VIF (1 to infinity)
# VIF = 1 (No Multicollinearity)
# VIF (b/w 1 to 5) (Low Multicollinearity)
# VIF (b/w 5 to 10) (Moderate Multicollinearity)
# VIF (above 10) (High Multicollinearity)

# VIF check Sequentially
vif_data = df[num_col]
vif_col = vif_data.shape[1]
col_tobe_kept = []
column_index = 0

for i in range(0,vif_col):
    vif_val = variance_inflation_factor(vif_data, column_index)
    print(column_index, '-->', vif_val)
    
    if vif_val <= 6:
        col_tobe_kept.append(num_col[i])
        column_index += 1
        
    else:
        vif_data = vif_data.drop(num_col[i], axis = 1)

    
vif_data.shape
# check anova for col_tob_kept

from scipy.stats import f_oneway

col_to_be_kept_num = []

for i in col_tobe_kept:
    a = list(df[i])   
    b= list(df['Approved_Flag'])
    
    group_p1 = [value for value, group in zip(a,b) if group == 'P1']
    group_p2 = [value for value, group in zip(a, b) if group == 'P2']
    group_p3 = [value for value, group in zip(a,b) if group == 'P3']
    group_p4 = [value for value, group in zip(a,b) if group == 'P4']    
    
    f_statistic, p_value = f_oneway(group_p1, group_p2, group_p3, group_p4)
    
    if p_value<=0.05:
        col_to_be_kept_num.append(i)
    
# concatenating numerical col and categorical col name to features.
features = col_to_be_kept_num + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]
    
# unique values in the columns
df['MARITALSTATUS'].unique()
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()
df['Approved_Flag'].value_counts()

# setting order for 'Education' unique values for label encoding.

# 12TH           : 2
# GRADUATE       : 3
# SSC            : 1
# POST-GRADUATE  : 4
# UNDER GRADUATE : 3
# OTHERS         : 1
# PROFESSIONAL   : 3
    
# others need to be verified by end user.

df.loc[df['EDUCATION'] == '12TH', 'EDUCATION'] = 2
df.loc[df['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
df.loc[df['EDUCATION'] == 'GRADUATE', 'EDUCATION'] = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE', 'EDUCATION'] = 3
df.loc[df['EDUCATION'] == 'PROFESSIONAL', 'EDUCATION'] = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE', 'EDUCATION'] = 4
df.loc[df['EDUCATION'] == 'OTHERS', 'EDUCATION'] = 1

df['EDUCATION'].unique()
df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype(int)



# One hot encoding for the categorical columns other than 'Education'
df_encoded = pd.get_dummies(df, columns = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])
df_encoded.info()                                           

# Dependent and Independent Variable
# x = df_encoded.drop(['Approved_Flag'], axis = 1)
x = df_encoded.drop(columns = ['Approved_Flag'])
y = df_encoded['Approved_Flag']
y.value_counts()

# RANDOM FOREST
x_train, x_test, y_train, y_test  = train_test_split(x,y, test_size = 0.2, random_state = 42)

rf =  RandomForestClassifier(n_estimators= 200, random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f'Class: {v}:')
    print(f'Precision: {precision[i]}')
    print(f'Recall: {recall[i]}')
    print(f'F1 Score: {f1_score[i]}')


# xgboost
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Access the mapping of categories to codes
category_to_code_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
for category, code in category_to_code_mapping.items():
    print(f"{category}: {code}")
    
# P1: 0
# P2: 1
# P3: 2
# P4: 3


xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class = 4)


x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size = 0.2, random_state = 42)
xgb_classifier.fit(x_train, y_train)   
y_pred = xgb_classifier.predict(x_test)

accuracy_new = accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

print(f'Overall Accuracy: {accuracy_new:.2f}')
print()
for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f'Class : {v}')
    print(f'Precision: {precision[i]}')
    print(f'Recall: {recall[i]}')
    print(f'f1 score: {f1_score[i]}')
    print()
    
# Decision Tree

from sklearn.tree import DecisionTreeClassifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)

DT = DecisionTreeClassifier(max_depth = 20, min_samples_split=10)

DT.fit(x_train, y_train)
y_pred = DT.predict(x_test)

accuracy_DT = accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

print(f'DT Accuracy: {accuracy_DT:.2f}')
print()
for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f'Class: {v}')
    print(f'precision: {precision[i]}')
    print(f'Recall: {recall[i]}')
    print(f'f1 score : {f1_score[i]}')
    print()
    

# Till now, we have observed the xgboost which is ensemble learning is working best amongst
# the three. Hence proceeding with this and will hypertune the same.

df_encoded['Approved_Flag'].value_counts()
# The data seems balance hence we can go with accuracy as the metric.
# We judge the metric for accuracy by looking at the nature of target variable whether it is
# balanced or imbalanced.

# xgboost is giving highest accuracy, hence we will pick it.

# Hyperparameter Tuning - Defining the hyperparameter grid

param_grid = {
    'colsample_bytree' : [0.1, 0.3, 0.5, 0.7, 0.9],
    'learning_rate'    : [0.001, 0.01, 0.1, 1],
    'max_depth'        : [3,5,8,10],
    'alpha'            : [1,10,100],
    'n_estimators'     : [10, 50, 100]
    }

index = 0

answers_grid = { 
    'combination'       : [],
    'train_accuracy'    : [],
    'test_accuracy'     : [],
    'colsample_bytree'  : [],
    'learning_rate'     : [],
    'max_depth'         : [],
    'alpha'             : [],
    'n_estimators'      : []
    }


# for colsample_bytree in param_grid['colsample_bytree']:
#   for learning_rate in param_grid['learning_rate']:
#     for max_depth in param_grid['max_depth']:
#       for alpha in param_grid['alpha']:
#           for n_estimators in param_grid['n_estimators']:
             
#               index = index + 1
             
#               # Define and train the XGBoost model
#               model = xgb.XGBClassifier(objective='multi:softmax',  
#                                        num_class=4,
#                                        colsample_bytree = colsample_bytree,
#                                        learning_rate = learning_rate,
#                                        max_depth = max_depth,
#                                        alpha = alpha,
#                                        n_estimators = n_estimators)
               
       
                     
#               y = df_encoded['Approved_Flag']
#               x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

#               label_encoder = LabelEncoder()
#               y_encoded = label_encoder.fit_transform(y)


#               x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


#               model.fit(x_train, y_train)
  

       
#               # Predict on training and testing sets
#               y_pred_train = model.predict(x_train)
#               y_pred_test = model.predict(x_test)
       
       
#               # Calculate train and test results
              
#               train_accuracy =  accuracy_score (y_train, y_pred_train)
#               test_accuracy  =  accuracy_score (y_test , y_pred_test)
              
              
       
#               # Include into the lists
#               answers_grid ['combination']   .append(index)
#               answers_grid ['train_Accuracy']    .append(train_accuracy)
#               answers_grid ['test_Accuracy']     .append(test_accuracy)
#               answers_grid ['colsample_bytree']   .append(colsample_bytree)
#               answers_grid ['learning_rate']      .append(learning_rate)
#               answers_grid ['max_depth']          .append(max_depth)
#               answers_grid ['alpha']              .append(alpha)
#               answers_grid ['n_estimators']       .append(n_estimators)
       
       
#               # Print results for this combination
#               print(f"Combination {index}")
#               print(f"colsample_bytree: {colsample_bytree}, learning_rate: {learning_rate}, max_depth: {max_depth}, alpha: {alpha}, n_estimators: {n_estimators}")
#               print(f"Train Accuracy: {train_accuracy:.2f}")
#               print(f"Test Accuracy : {test_accuracy :.2f}")
#               print("-" * 30)

# Define the parameter grid for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# Define the XGBClassifier with the initial set of hyperparameters
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model with the best hyperparameters on the test set
best_model = grid_search.best_estimator_
accuracy = best_model.score(x_test, y_test)
print("Test Accuracy:", accuracy)

# Best Hyperparameters: {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200}
# Based on risk appetite of the bank, you will suggest P1,P2,P3,P4 to the business end user


# Predicting on the unseen data
a3 = pd.read_excel(r"C:\Users\Admin\Desktop\Bindu\Projects All\Credit Risk Modelling Using Machine Learning\Unseen_Dataset.xlsx")
cols_df = list(df.columns) # list of columns in df 
cols_df.pop(42)  # removing 'Approved_Flag' for which the index number is 42.

# selecting columns in a3 which are there in cols_df
df_unseen = a3[cols_df]  

df_unseen['EDUCATION'].unique() # identifying unique categories in Education column

# Locating and mapping the categories manually as there is an ordinal relationship 
df_unseen.loc[df_unseen['EDUCATION'] == '12TH', 'EDUCATION'] = 2
df_unseen.loc[df_unseen['EDUCATION'] == 'SSC', 'EDUCATION'] = 1
df_unseen.loc[df_unseen['EDUCATION'] == 'GRADUATE', 'EDUCATION'] = 3
df_unseen.loc[df_unseen['EDUCATION'] == 'UNDER GRADUATE', 'EDUCATION'] = 3
df_unseen.loc[df_unseen['EDUCATION'] == 'PROFESSIONAL', 'EDUCATION'] = 3
df_unseen.loc[df_unseen['EDUCATION'] == 'POST-GRADUATE', 'EDUCATION'] = 4
df_unseen.loc[df_unseen['EDUCATION'] == 'OTHERS', 'EDUCATION'] = 1

# confirmation of mapping
df_unseen['EDUCATION'].value_counts()

# checking data type of the Education column and then converting to integer.
df_unseen['EDUCATION'].dtype
df_unseen['EDUCATION'] = df_unseen['EDUCATION'].astype(int)
df_unseen.info()

# One hot encoding of df_unseen
df_unseen_encoded = pd.get_dummies(df_unseen, columns = ['MARITALSTATUS', 'GENDER', 'last_prod_enq2', 'first_prod_enq2'])
df_unseen_encoded.info() 

# creating model with best parameters identified earlier
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4,
                              learning_rate = 0.2,  max_depth= 3, n_estimators= 200)
xgb_model.fit(x_train, y_train)   # fitting the data to the model

# Prediction on unseen data
y_pred_unseen = xgb_model.predict(df_unseen_encoded)
a3['Target_variable'] = y_pred_unseen

# Downloading to excel file.
a3.to_excel(r'C:\Users\Admin\Desktop\Bindu\Projects All\Credit Risk Modelling Using Machine Learning\Predictions_unseen_data.xlsx', index = False)
