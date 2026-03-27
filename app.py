import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn .model_selection import train_test_split

df=pd.read_csv("cars.csv",nrows=1000)
df['price']=df['price'].replace(r'[\$,]','',regex=True).astype(float)
df = df.drop([
    "seller_name","exterior_color","interior_color",
    "driver_reviews_num","seller_rating","driver_rating",
    "price_drop","mpg"
], axis=1)
df["drivetrain"]=df['drivetrain'].astype(str).str.lower()
df['drivetrain']=df["drivetrain"].apply(
    lambda x: "fwd" if "front_wheel drive" in x
    else "awd" if "all_wheel drive" in x
    else "rwd" if "rear_wheel drive" in x
    else "other"
)

def clean_transmission(x):
    if "auto"in x :
      return "automatic" 
    elif "manual"  in x:
        return "manual"
     
    else :
        return "other"
df['transmission']=df['transmission'].astype(str).str.lower()
df['transmission']=df['transmission'].apply(clean_transmission)

df['engine_size']=df['model'].str.extract(r'(\d+\.\d+)')
df["engine_size"]=df['engine_size'].astype(float)
df.fillna(0,inplace=True)
df=df.drop(['model','engine'],axis=1)
df=pd.get_dummies(df,columns=['manufacturer',
                             'fuel_type',
                             'transmission',
                             'drivetrain',
                             'accidents_or_damage'],drop_first=True)
bool_cols=df.select_dtypes(include='bool').columns
df[bool_cols]=df[bool_cols].astype(int)


#split
x=df.drop(['price'],axis=1)
y=df['price']
model=RandomForestRegressor()
model.fit(x,y)


#---UI---
st.title('car price predictor')
st.sidebar.title("this predicts based on machinlearning model")
st.sidebar.markdown("""
    <style>
                    body{background-color:f7f5fa;}
            <style>""",unsafe_allow_html=True
)
st.write("enter details of the car you want")
year=st.number_input('year',2000,2025,2015)
mileage =st.number_input('mileage',0,200000,50000)
engine_size=st.number_input('engine size',1.0,6.0,2.0)
fuel_type=st.selectbox("fuel type",["electric","gas","diessel","hybrid"])
transmission=st.selectbox("transmission",["automatic","manual","other"])



if st.button('predict price'):

    inputdata=pd.DataFrame({
        'year':[year],
        'mileage':[mileage],
        'engine_size':[engine_size]
    })
    inputdata=inputdata.reindex(columns=x.columns,fill_value=0)
    prediction=model.predict(inputdata)[0]
    st.success(f'Estimated price: ${int(prediction):,}')