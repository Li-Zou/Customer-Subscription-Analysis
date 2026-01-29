import pandas as pd
import numpy as np
from scipy.stats import norm
import os
import logging

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../'))  

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
        
def two_proportion_z_test(df1, variable, alpha):
    df1_A = df1.loc[(df1['campaign_cohort'] == 'control') | (df1['campaign_cohort'] == variable)]

    N_con = (df1_A["campaign_cohort"] == "control").sum().item()
    N_exp = (df1_A["campaign_cohort"] == variable).sum().item()
    
    # calculating the total number of churn per group by summing 1's
    X_con = df1_A.groupby("campaign_cohort")["churned"].sum().loc["control"].item()
    X_exp = df1_A.groupby("campaign_cohort")["churned"].sum().loc[variable].item()
    
    # computing the estimate of churn probability per group
    p_con_hat = X_con/N_con
    p_exp_hat = X_exp/N_exp
    
    # computing the estimate of pooled churned probability
    p_pooled_hat = (X_con+X_exp)/(N_con + N_exp)
    # computing the estimate of pooled variance
    pooled_variance = p_pooled_hat * (1-p_pooled_hat) * (1/N_con + 1/N_exp)
    # computing the standard error of the test
    SE = np.sqrt(pooled_variance)
    
    # computing the test statistics of Z-test
    Test_stat = round((p_exp_hat - p_con_hat)/SE,3)
    #ATE estimation (difference in means)
    ATE = round((p_exp_hat - p_con_hat) ,3)
    # critical value of the Z-test
    Z_crit = round(norm.ppf(1-alpha/2),3)
    #calculating p value
    p_value = round(2 * norm.sf(abs(Test_stat)).item(),3)#sf--survival function 
    
    # Calculate the Confidence Interval (CI) for a 2-sample Z-test
    ## Calculate the lower and upper bounds of the confidence interval
    CI = [round((p_exp_hat - p_con_hat) - SE*Z_crit, 3).item(),  
        round((p_exp_hat - p_con_hat) + SE*Z_crit, 3) .item()  ]
    
    return [ATE, CI, p_value, Test_stat]

def AB_test(df):
    # alpha: significance level
    ATE_a = two_proportion_z_test(df, variable = 'variant_a', alpha = 0.05)
    ATE_b = two_proportion_z_test(df, variable = 'variant_b', alpha = 0.05)
    
    df_age_low = df.loc[df['participant_age']<=50]
    df_age_high = df.loc[df['participant_age']>50]
    ATE_a_age_low = two_proportion_z_test(df_age_low, variable = 'variant_a', alpha = 0.05)
    ATE_a_age_high = two_proportion_z_test(df_age_high, variable = 'variant_a', alpha = 0.05)
    
    ATE_b_age_low = two_proportion_z_test(df_age_low, variable = 'variant_b', alpha = 0.05)
    ATE_b_age_high = two_proportion_z_test(df_age_high, variable = 'variant_b', alpha = 0.05)
    
    d1 = pd.DataFrame({'variant_a': ATE_a,'variant_b': ATE_b,
                                'Age<=50 (variant_a)': ATE_a_age_low, 'Age>50 (variant_a)': ATE_a_age_high,
                                'Age<=50 (variant_b)': ATE_b_age_low, 'Age>50 (variant_b)': ATE_b_age_high},
                                index = ['ATE', 'confidence_intervals', 'p_value','Test_stat'])
    d2 = {}
    col1 = list(df['baseline_churn_risk_band'].unique())
    for i in col1:
        df_seg = df.loc[df['baseline_churn_risk_band'] == i]
        ATE_seg_a = two_proportion_z_test(df_seg, variable = 'variant_a', alpha = 0.05)
        ATE_seg_b = two_proportion_z_test(df_age_high, variable='variant_b', alpha = 0.05)
        d2[i+' (variant_a)'] = ATE_seg_a
        d2[i+' (variant_b)'] = ATE_seg_b
    d2 = pd.DataFrame(d2,index = ['ATE', 'confidence_intervals', 'p_value', 'Test_stat'])
    AB_test_result = pd.merge(d1, d2, left_index=True, right_index=True)
    
    root_dir = get_root_directory()
    AB_test_result.to_csv(os.path.join(root_dir, "outputs/AB_test_result.csv")) 
    
def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load the preprocessed training data from the interim directory
        df = load_data(os.path.join(root_dir, 'data/raw/train.csv'))
        
        # Conduct the AB test
        AB_test(df)
        
    except Exception as e:
        logger.error('Failed to complete the AB test: %s', e)
        print(f"Error: {e}")   
    
if __name__ == '__main__':
    main()  
    









