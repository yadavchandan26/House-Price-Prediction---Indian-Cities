#in this particular file , we will mainly work on data loading , preprocessing & data splitting

import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(data_path):
    data=pd.read_csv(data_path)
    data=data.dropna()
    return data

def preprocessing(data):
    data['Age']=2026-data['Build_Year']
    data['location_hack']=data['Location'].map({
    'Delhi':4,
    'Gurugram':3,
    'Noida':2,
    'Lucknow':2,
    'Jaipur':1,
    'Indore':1,
    'Kanpur':1,
    'Prayagraj':1
    
    })

    return data

def feature_selection(data):
    
    x=data[['Area_SqFt','Rooms','Age','Has_Pool','location_hack']]
    x=pd.get_dummies(x,columns=['Has_Pool'],drop_first=True)
    y=data['Price']

    return x,y


def final_process(data_path):
    data=load_dataset(data_path)
    data=preprocessing(data)

    x,y=feature_selection(data)

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    return x_train,x_test,y_train,y_test
