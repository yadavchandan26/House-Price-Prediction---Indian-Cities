import pandas as pd

def evaluate_model(model,x_train,y_train):
    score=model.score(x_train,y_train)
    print(f"model score is :{score: }")

    return score

def final_submission(x_test,y_test,y_pred,output_path="submission.csv"):
    submission=pd.DataFrame({
        "Price":y_pred
        })

    submission.to_csv(output_path,index=True)
    print(f"submission completed and saved to {output_path}")
    return submission