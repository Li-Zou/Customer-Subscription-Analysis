import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import logging
import yaml

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise
             
def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../'))  

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace = True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
        
def all_eval_metrics(y_test: np.ndarray, y_pred_prob: np.ndarray) -> list:
    acc = []
    acc.append(roc_auc_score(y_test, y_pred_prob))
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
    acc.append(auc(recall,precision))
    return acc

def KNN_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, n_neighbor: list) -> KNeighborsClassifier:
    """Train a KNN model."""
    try:
        param_grid = {'n_neighbors': n_neighbor}
        knn = KNeighborsClassifier()
        grid = GridSearchCV(knn, param_grid, refit = True, verbose = 3)##the default 5-fold cross validation
        grid.fit(X_train, y_train) ## fitting the model for grid search 
        y_pred_prob = grid.predict_proba(X_test)[:, 1]
        ac = all_eval_metrics(y_test,y_pred_prob)
        logger.debug('KNN model training completed')
        return ac,grid
    except Exception as e:
        logger.error('Error during KNN model training: %s', e)
        raise
def LogisticRegression_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, LR_C: list) -> LogisticRegression:
    """Train a LogisticRegression model."""
    try:
        param_grid = { 'C': LR_C}   
        grid = GridSearchCV(LogisticRegression(random_state = 0), param_grid, refit = True, verbose = 3)##the default 5-fold cross validation
        grid.fit(X_train, y_train) ## fitting the model for grid search 
        y_pred_prob = grid.predict_proba(X_test)[:, 1]
        ac = all_eval_metrics(y_test,y_pred_prob)
        logger.debug('LogisticRegression model training completed')
        return ac,grid
    except Exception as e:
        logger.error('Error during LogisticRegression model training: %s', e)
        raise
def RandomForest_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, RF_n_estimators: list) -> RandomForestClassifier:
    """Train a RandomForest model."""
    try:
        param_grid = { 'n_estimators': RF_n_estimators} #many parameters have been tested
        grid = GridSearchCV(RandomForestClassifier(random_state = 1), param_grid,refit = True, verbose = 3)
        grid.fit(X_train, y_train)
        y_pred_prob = grid.predict_proba(X_test)[:, 1]
        ac = all_eval_metrics(y_test,y_pred_prob)
        logger.debug('RandomForest model training completed')
        return ac,grid
    except Exception as e:
        logger.error('Error during RandomForest model training: %s', e)
        raise
        
def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise      
              
def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        KNN_C = params['model_building']['knn']
        LR_C = params['model_building']['LR_C']
        RF_n_estimators = params['model_building']['RF_n_estimators']

        # Load the preprocessed training data from the interim directory
        df_train_data = load_data(os.path.join(root_dir, 'data/raw/train.csv'))

        ##Remove features that are not useful to train ML models
        df_train_data.drop(columns = ['customer_id','subscription_id','campaign_cohort','treatment_sent_flag','revenue_next_12m_observed','marketing_channel','country_code'],inplace=True)

        #label encodling for ordinal data
        df_train_data['baseline_churn_risk_band'] = df_train_data['baseline_churn_risk_band'].map({'low':0,'medium':1,'high':2,'very high':3})
        X_train = df_train_data.drop('churned',axis = 'columns').values
        y_train = df_train_data['churned'].values#.astype(np.float32)
        
        # Load the preprocessed training data from the interim directory
        df_test_data = load_data(os.path.join(root_dir, 'data/raw/test.csv'))

        ##Remove features that are not useful to train ML models
        df_test_data.drop(columns = ['customer_id','subscription_id','campaign_cohort','treatment_sent_flag','revenue_next_12m_observed','marketing_channel','country_code'],inplace=True)

        #label encodling for ordinal data
        df_test_data['baseline_churn_risk_band'] = df_test_data['baseline_churn_risk_band'].map({'low':0,'medium':1,'high':2,'very high':3})
        X_test = df_test_data.drop('churned',axis = 'columns').values
        y_test = df_test_data['churned'].values#.astype(np.float32)
        
        ##handle imbalanced data by oversampling
        ros = RandomOverSampler(random_state = 0)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Train the models using hyperparameters from params.yaml
        ac1,knn_model = KNN_model(X_train, X_test, y_train, y_test,KNN_C)
        ac2,LR_model = LogisticRegression_model(X_train, X_test, y_train, y_test,LR_C)
        ac3,RF_model = RandomForest_model(X_train, X_test, y_train, y_test,RF_n_estimators)
        prediction_acc = pd.DataFrame({'K-Nearest Neighbors':ac1,'Logistic Regression':ac2,'Random Forest':ac3},index = ['ROC AUC','PR AUC'])
        prediction_acc.to_csv(os.path.join(root_dir,"outputs/prediction_acc.csv"))

        # Save the trained model in the root directory
        save_model(knn_model, os.path.join(root_dir, 'outputs/KNN_model.pkl'))
        save_model(LR_model, os.path.join(root_dir, 'outputs/LR_model.pkl'))
        save_model(RF_model, os.path.join(root_dir, 'outputs/RF_model.pkl'))

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()



