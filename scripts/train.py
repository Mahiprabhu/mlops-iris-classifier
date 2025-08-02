import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

# Configure logging to display informative messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# All runs started from this script will be grouped under this experiment name in MLFlow.
mlflow.set_experiment("Iris Classifier Experiment")

def train_and_log_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Trains a given model and logs its parameters, metrics, and the model artifact
    to MLflow.
    """
    # Everything we do inside this block will be logged to this single run.
    with mlflow.start_run():
        logging.info(f"Starting MLflow run for {model_name}...")

        # 1. Train the model
        model.fit(X_train, y_train)

        # 2. Make predictions on the test set
        predictions = model.predict(X_test)
        
        # 3. Calculate an evaluation metric (accuracy in this case)
        accuracy = accuracy_score(y_test, predictions)

        # 4. Log parameters and metrics to MLflow
     
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        
        # 5. Log the trained model as an artifact
        mlflow.sklearn.log_model(model, "model")

        logging.info(f"Finished run for {model_name}. Accuracy: {accuracy:.4f}")
        
        # Get the run ID for reference
        run_id = mlflow.active_run().info.run_id
        logging.info(f"MLflow Run ID: {run_id}")


if __name__ == "__main__":
    # Load the Iris dataset from scikit-learn
    logging.info("Loading and splitting the Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train and track the Logistic Regression model ---
    logging.info("Training Logistic Regression model...")
    # Instantiate the model with some hyperparameters
    lr_model = LogisticRegression(max_iter=200, solver='liblinear')
    # Call our function to train and log the model
    train_and_log_model(lr_model, X_train, X_test, y_train, y_test, "LogisticRegression")
    
    # --- Train and track the Random Forest Classifier model ---
    logging.info("Training Random Forest Classifier model...")
    # Instantiate a second model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Call the same function to train and log this model as a new run
    train_and_log_model(rf_model, X_train, X_test, y_train, y_test, "RandomForestClassifier")
    
    logging.info("Training process complete. To view results, run 'mlflow ui' in this directory.")