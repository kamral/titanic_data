# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
#
# train_data=pd.read_csv('train.csv', sep=',')
#
# print(train_data.head())
# '''
#  PassengerId  Survived  Pclass                                               Name  ...            Ticket     Fare  Cabin  Embarked
# 0            1         0       3                            Braund, Mr. Owen Harris  ...         A/5 21171   7.2500    NaN         S
# 1            2         1       1  Cumings, Mrs. John Bradley (Florence Briggs Th...  ...          PC 17599  71.2833    C85         C
# 2            3         1       3                             Heikkinen, Miss. Laina  ...  STON/O2. 3101282   7.9250    NaN         S
# 3            4         1       1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  ...            113803  53.1000   C123         S
# 4            5         0       3                           Allen, Mr. William Henry  ...            373450   8.0500    NaN         S
#
# '''
#
# test_data=pd.read_csv('test.csv',sep=',')
# print(test_data.head())
# '''
# PassengerId  Pclass                                          Name     Sex   Age  SibSp  Parch   Ticket     Fare Cabin Embarked
# 0          892       3                              Kelly, Mr. James    male  34.5      0      0   330911   7.8292   NaN        Q
# 1          893       3              Wilkes, Mrs. James (Ellen Needs)  female  47.0      1      0   363272   7.0000   NaN        S
# 2          894       2                     Myles, Mr. Thomas Francis    male  62.0      0      0   240276   9.6875   NaN        Q
# 3          895       3                              Wirz, Mr. Albert    male  27.0      0      0   315154   8.6625   NaN        S
# 4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female  22.0      1      1  3101298  12.2875   NaN        S
#
# '''
#
# women=train_data.loc[train_data.Sex=='female']['Survived']
# print(women)
#
# '''
# 1      1
# 2      1
# 3      1
# 8      1
# 9      1
#       ..
# 880    1
# 882    0
# 885    0
# 887    1
# 888    0
#
# '''
# rate_women=sum(women)/len(women)
# print(rate_women)
# '''
# 0.7420382165605095
# '''
# men=train_data.loc[train_data['Sex']=='male']['Survived']
# print(men)
# '''
# 0      0
# 4      0
# 5      0
# 6      0
# 7      0
#       ..
# 883    0
# 884    0
# 886    0
# 889    1
# 890    0
#
# '''
# rate_men=sum(men)/len(men)
# print(rate_men)
# '''
# 0.18890814558058924
#
# '''
# features=["Pclass", "Sex", "SibSp",  "Parch"]
# X=train_data[features]
# X=pd.get_dummies(X)
# y=train_data.Survived
# X_test=test_data[features]
# X_test=pd.get_dummies(X_test)
#
#
#
# model_1=DecisionTreeClassifier(random_state=0)
# model_2=DecisionTreeRegressor(random_state=1)
# model_3=RandomForestClassifier(n_estimators=100, random_state=0)
# model_4=RandomForestClassifier
#
#
#
# clf=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# print(clf.fit(X,y))
# predictions=clf.predict(X_test)
# print(predictions)
# out_put=pd.DataFrame({
#     'PassengerId':test_data.PassengerId,
#     'Survived':predictions,
# })
# out_put.to_csv('my_submission.csv', index=False)
# print('YOur submisisons was successfuly saved')
#
# model_1=DecisionTreeRegressor(random_state=0)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
train_data=pd.read_csv('train .csv', sep=',')
test_data=pd.read_csv('test.csv', sep=',')
print(train_data.columns)
features=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X_train=train_data[features]
print(X_train)
X_test=test_data[features]
print(X_test)
y=train_data.Survived
print(y)

X_isnull_sum=X_train.isnull().sum()
print(X_isnull_sum)
X_train=train_data.fillna(X_train.Age.mean())
print(X_train)
X_train=pd.get_dummies(X_train)


X_test=train_data.fillna(X_train.Age.mean())
print(X_test)
X_test=pd.get_dummies(X_test)
print(X_test)



train_X, val_X, train_y, val_y=train_test_split(X_train,y,train_size=0.8,
                                                test_size=0.2,random_state=0)

clf=DecisionTreeClassifier(random_state=0)
clf.fit(train_X, train_y)
preds=clf.predict(val_X)
mae=mean_absolute_error(val_y, preds)
print(preds)
print(mae)








# model_1=RandomForestRegressor(n_estimators=50, random_state=0)
# model_2=RandomForestRegressor(n_estimators=100, random_state=0)
# model_3=RandomForestRegressor(n_estimators=150, criterion='mae', random_state=0)
# model_4=RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
# model_4=RandomForestRegressor(n_estimators=200, min_samples_split=7, random_state=0)
# model_5=DecisionTreeClassifier(random_state=0)
# model_6=DecisionTreeRegressor(random_state=0)
#
# models=[model_1,model_2,model_3,model_4,model_5,model_6]
#
# def best_model(model, X_t=train_X, X_v=val_X, y_t=train_y,y_v=val_y ):
#     model.fit(X_t,y_t)
#     preds=model.predict(X_v)
#     return mean_absolute_error(y_v, preds)
#
# for i in range(0,len(models)):
#     mae=best_model(models[i])
#     print('Model %d MAE: %d' %(i+1, mae))













