import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
import sys


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Argumen:
    # sys.argv[1] = n_estimators
    # sys.argv[2] = max_depth
    # sys.argv[3] = path ke file CSV
    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "train_pca.csv"
        )
    )

    data = pd.read_csv(file_path)  

   
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2
    )


    input_example = X_train.iloc[0:5]    
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    # Mulai MLflow run
    with mlflow.start_run():      
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=40
        )
        model.fit(X_train, y_train)
        predicted_qualities = model.predict(X_test)
      
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )  
     
        accuracy = model.score(X_test, y_test)

        # Log parameter
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

    

     