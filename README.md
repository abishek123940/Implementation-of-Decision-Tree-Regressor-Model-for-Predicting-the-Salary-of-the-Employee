# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ABISHEK S
RegisterNumber:  212224240003
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

y_pred

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test, y_pred)
mse

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_test, y_pred)
mae

from sklearn.metrics import r2_score
r1=r2_score(y_test,y_pred)
r1

dt.predict([[5,6]])

```

## Output:
head:

<img width="872" height="225" alt="image" src="https://github.com/user-attachments/assets/12e6a143-2c9f-4a11-aab4-0c671462848e" />

info:

<img width="704" height="204" alt="image" src="https://github.com/user-attachments/assets/ae113598-2600-4624-b980-1cdab0213272" />

isnull:

<img width="541" height="128" alt="image" src="https://github.com/user-attachments/assets/85636479-0a9c-4854-979c-977db13bec9b" />

head:

<img width="673" height="218" alt="image" src="https://github.com/user-attachments/assets/25b5b9d8-947f-4a39-a187-a620d0e94ce1" />

x.head:

<img width="637" height="223" alt="image" src="https://github.com/user-attachments/assets/c5d055c1-dd99-4812-92d5-6e2bfa08d735" />

y.head:

<img width="685" height="234" alt="image" src="https://github.com/user-attachments/assets/0a257c37-64a4-432c-9825-8b4649d34901" />

y_pred:

<img width="476" height="68" alt="image" src="https://github.com/user-attachments/assets/e36c240f-4639-4526-9d3b-9971afc0f86e" />

mse:

<img width="559" height="56" alt="image" src="https://github.com/user-attachments/assets/df3a3129-3da2-4750-ac37-c95984b898e3" />

mae:

<img width="488" height="98" alt="image" src="https://github.com/user-attachments/assets/071bce7d-1057-46d8-b7bb-c4c9511a37e7" />

r2:

<img width="543" height="75" alt="image" src="https://github.com/user-attachments/assets/c9f51355-4804-44c1-86bb-5fe2a4f1550c" />

predict:

<img width="1243" height="130" alt="image" src="https://github.com/user-attachments/assets/aa30b139-6066-4309-8d2b-7e339a6dd789" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
