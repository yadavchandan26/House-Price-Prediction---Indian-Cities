#in this file , our main focused will be on model training 
#this file we return model at the end

from sklearn.linear_model import LinearRegression

def train_model(x_train,y_train):
    model=LinearRegression()
    model.fit(x_train,y_train)

    return model