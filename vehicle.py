import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv("cars.csv",nrows=1000)#the first 1000 rows only
print(df.columns)
df=df.drop([
    "seller_name",
    "exterior_color",
    "interior_color",
    "driver_reviews_num"
    ],axis=1)#droping less necessary columns
df["price"]=df["price"].replace(r'[\$,]',"",regex=True).astype(float)# find $ and , and replace with nothing ,regex means it tells python to detect pattern not text
print(df["price"].head())
print(df.isnull().sum())
df = df.drop([
    'seller_rating',
    'driver_rating',
    'price_drop',
    'mpg'
], axis=1)#drop higher missing data
element=["engine",
           "drivetrain",
           "fuel_type",
           "accidents_or_damage"]
for i in element:
    df[i]=df[i].fillna("unknown")


df["accidents_or_damage"]=df["accidents_or_damage"].astype(str).str.lower().map({"yes":1 ,"no":0})#convert yes/no and fill missing data
df["one_owner"]=df["one_owner"].astype(str).str.lower().map({"yes":1 ,"no":0})
df["accidents_or_damage"]=df["accidents_or_damage"].fillna(0)
df["one_owner"]=df["one_owner"].fillna(0)

def clean_transmission(x):
    if "auto"in x :
      return "automatic" 
    elif "manual"  in x:
        return "manual"
     
    else :
        return "other"
df['transmission']=df['transmission'].astype(str).str.lower()
df['transmission']=df['transmission'].apply(clean_transmission)

df["drivetrain"]=df["drivetrain"].str.lower()
df["drivetrain"]=df["drivetrain"].apply(
    lambda x: "fwd" if "front_wheel drive" in x
    else "awd" if "all_wheel drive" in x
    else "rwd" if "rear_wheel drive" in x
    else "other"
)
df=pd.get_dummies(df,columns=[
    'manufacturer',
    'fuel_type',
    'transmission',
    'drivetrain',
    'accidents_or_damage'],drop_first=True)#encoding


df["engine_size"]=df["model"].str.extract(r'(\d+\.\d+)')
df["engine_size"]=df["engine_size"].astype(float)
df=df.drop(["model","engine"],axis=1)

bool_col=df.select_dtypes(include=bool).columns
df[bool_col]=df[bool_col].astype(int)
print(df.head())

#lets train our model
x=df.drop(['price'],axis=1)
y=df['price']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2 ,random_state=42)

model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
score=model.score(X_test,y_test)#evaluate the model 
print(f"Random forest score:{score}")

feature_importance=pd.Series(model.feature_importances_,index=x.columns).sort_values(ascending=False)
print(feature_importance)

predicted=model.predict(X_test)
error=mean_absolute_error(predicted,y_test)
print(f"the mean absolute value error is {error}")