import os
import pandas as pd
import chardet
import numpy as np
import dateutil.parser
from datetime import datetime
from currency_converter import CurrencyConverter
from scipy import stats
from sklearn.model_selection import train_test_split
import logging
import yaml


# Logging configuration
logger = logging.getLogger('preprocessing')
logger.setLevel(logging.DEBUG)

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

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        # Check the encodings for csv file
        with open(data_url, 'rb') as f:
            data = f.read(200000)
        result = chardet.detect(data)
        # load data
        df = pd.read_csv(data_url, encoding=result['encoding'])
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise             
        
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling missing values, duplicates, mixed decimal separators, mixed date formats, and empty strings."""
    try:
        # Some columns contain a mix of uppercase and lowercase letters; convert them all to lowercase
        columns=df.columns
        for i in columns:
            try:
                df[i]=df[i].str.lower()
            except:
                pass     
        # Handling missing value
        df.replace(['unknown','missing'], np.nan, inplace=True)
        # Check the missing value for each feature
        missing_count = df.isna().sum()
        missing_count = round(missing_count[missing_count != 0]/df.shape[0],2)
        
        # Drop any features whose proportion of missing values exceeds a specified threshold (e.g., 0.2)
        missing_count.sort_values(inplace = True,ascending = False)
        missing_count = missing_count[missing_count >= 0.2]
        feature_remove = list(missing_count.index)
        df1=df.drop(columns = feature_remove)
        df1 = df1.dropna()# Remove samples that contain at least one missing value
        
        # Handle date formats: Convert different date formats to a given date format
        df2 = df1.copy()
        df2['subscription_date'] = df2['subscription_date'].apply(lambda x: dateutil.parser.parse(x).strftime("%Y-%m-%d"))
        df2['observation_end_date'] = df2['observation_end_date'].apply(lambda x: dateutil.parser.parse(x).strftime("%Y-%m-%d"))
        # These two date formats may not be usable directly, so a new feature (subscription_month_length) is created to make use of this information
        o1 = df2['subscription_date'] + ',' + df2['observation_end_date']
        def month_difference(date_str):
            a = date_str.split(',')
            d1 = datetime.strptime(a[1], '%Y-%m-%d')
            d2 = datetime.strptime(a[0], '%Y-%m-%d')
            return d1.month - d2.month + 12*(d1.year - d2.year)
        df2['subscription_month_length'] = o1.apply(lambda x: month_difference(x))
            
        # Replace "," with "."
        df2['add_ons'] = df2['add_ons'].apply(lambda x: int(float((x.replace(',','.')))))
        
        # Convert all currency values to euros
        c = CurrencyConverter()
        def Currency_Conversion(x:str) ->float:
            x = x.replace(',','.')
            if x.startswith("$"):  
                return round(c.convert(float(x[1:]), 'EUR', 'USD'),2)
            elif x.startswith("€"):
                return float(x[1:])
            else:
                return float(x)
        for variable in ['monthly_spend_estimated','offer_cost_eur','historic_revenue_12m','revenue_next_12m_observed']:
            df2[variable] = df2[variable].apply(lambda x: Currency_Conversion(x))
         
        # Transform the two formats into a single one for the donation_share_charity feature
        def handel_donation_share_charity(x:str) ->float:
            x = x.replace(',','.')
            if float(x)<1:  
                return round(float(x)*100,2)
            else:
                return round(float(x),2)
        df2['donation_share_charity'] = df2['donation_share_charity'].apply(lambda x: handel_donation_share_charity(x))
        
        # Remove the '+' from the web_sessions_90d_raw feature; to be honest, I don't fully understand what the '+' indicates
        df2['web_sessions_90d_raw'] = df2['web_sessions_90d_raw'].apply(lambda x: x.strip('+'))
        
        # 'service_contacts_12m' column contain 366 outliers, these samples corresponding to these outliers needed to be removed
        df2 = df2[(np.abs(stats.zscore(df2['service_contacts_12m'])) < 3)]
        df2['treatment_sent_flag'] = df2['treatment_sent_flag'].map({'true': 1, 'y': 1,'yes':1,'1':1, 'false': 0, 'n': 0,'no':0,'0':0})
        df2['baseline_churn_risk_band'] = df2['baseline_churn_risk_band'].str.replace("_",' ')
        df2['churned'] = df2['churned'].map({'true': 1, 'y': 1,'yes':1,'1':1, 'false': 0, 'n': 0,'no':0,'0':0})
          
        columns = list(df2.columns)
        di = {}
        for i in columns:
            di[i] = df2[i].dtype
        
        col1 = ['participant_age','web_sessions_90d_raw']
        for variable in col1:
            df2[variable] = df2[variable].apply(lambda x: int(x)) 
        
        '''only consider 3 situations: 
            'campaign_cohort'='Control' and 'treatment_sent_flag'=0;
            'campaign_cohort'='Variant_A' and 'treatment_sent_flag'=1;
            'campaign_cohort'='Variant_B' and 'treatment_sent_flag'=1;
        '''
        df2=df2.loc[((df2['campaign_cohort'] == 'control') & (df2['treatment_sent_flag'] == 0))|
                   ((df2['campaign_cohort'] == 'variant_a') & (df2['treatment_sent_flag'] == 1))|
                   ((df2['campaign_cohort'] == 'variant_b') & (df2['treatment_sent_flag'] == 1))]
        df2.drop(columns = ['legacy_system_id','subscription_date','postcode_area','observation_end_date'],inplace = True)
        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed, and mixed  are handled.')
        return df2
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it doesn't exist."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        
        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save the train and test data
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # Load parameters from the params.yaml in the root directory
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../params.yaml'))
        test_size = params['data_ingestion']['test_size']
        
        # Load data from the specified URL
        data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..\data\sample_data.csv')
        df = load_data(data_url=data_path)
        
        # Preprocess the data
        final_df = preprocess_data(df)
        
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        # Save the split datasets and create the raw folder if it doesn't exist
        save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data'))
        
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
    




