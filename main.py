from prediction.data_preprocessing import final_process
from prediction.model_training import train_model
from prediction.model_evaluation import evaluate_model,final_submission

data="data/dataset.csv"

def main():
    x_train,x_test,y_train,y_test=final_process(data)

    model=train_model(x_train,y_train)
    evaluate_model(model,x_train,y_train)
    y_pred=model.predict(x_test)
    final_submission(x_test,y_test,y_pred,output_path="output.csv")

if __name__=='__main__':
    main()