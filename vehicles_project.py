import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


data=pd.read_csv("cars 2 copy.csv",nrows=1000)#extract only the first 1000 rows only
data=pd.DataFrame(data)
print(data.head())
print(data.info())
print(data.columns)#lets check column name spelling


data=data.drop(["url","id","region_url"],axis=1)#these three features are not necessary for price pridiction
data["cylinder"]=data["cylinder"].str.extract("(\d+)") #we need only number 
data["cylinder"]=data["cylinder"].astype(float)#iconvert the number to float
data['cylinder'].head()

print(data.isnull().sum())#it is checking if there is missed data and sum up them

data["cylinder"]=data["cylinder"].fillna(data['cylinder'].median())
data["year"]=data["year"].fillna(data["year"].median())#it fills null columns with the median of the column
data["manufacturer"]=data["manufacturer"].fillna("unknown")
data["condition"]=data["condition"].fillna("unknown")
data["model"]=data['model'].fillna("unknown")

print(data.isnull().sum())#check for missing data again
