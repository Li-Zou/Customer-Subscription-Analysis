# Retention & Uplift Optimiser

## 1. Business Scenario

A company runs a monthly subscription business and wants to take a proactive, 
test-and-learn approach to predicting churn and understanding which actions 
actually help retain customers.

### The Experiment
Recently, the company ran a randomized controlled experiment with three groups:
* **Control:** No special treatment.
* **Variant A:** A standard discount offer.
* **Variant B:** A value-add offer (e.g., bonus tickets).

*Note: Each active subscription at the start of the experiment was randomly assigned 
to a cohort, but due to operational constraints, not everyone in a treatment cohort 
actually received the communication.*

### The Goal of this project
Build a production-grade data science pipeline that:
1.  **Cleans and standardizes** a messy CRM export.
2.  **Builds robust models** to predict churn probability.
3.  **Quantifies incremental impact** of Variant A and Variant B vs Control.


## 2. The Data

**File:** `data/sample_data.csv`

Data comes from 2 different main sources, given below as legacy_system_id. Each row 
represents one subscription at the start of the experiment. 
This file is a raw legacy export and is **not in an ingest-ready format**. We 
should assume:
* Mixed decimal separators (`.` and `,`).
* Mixed thousand separators.
* Mixed date formats (US vs EU, text months).
* Mixed encodings for booleans and categories.
* Embedded JSON-like strings in some columns.
* And other anomalies that can appear in a data export.

### Data Schema

|    | Column Name                 | Description                                           | Example Values                      |
|----|:----------------------------|:------------------------------------------------------|:------------------------------------|
| 1  | `customer_id`               | Unique customer identifier                            | `b7a782741f667201b5`                |
| 2  | `subscription_id`           | Unique subscription identifier                        | `7db88cdd3c295d2276`                |
| 3  | `legacy_system_id`          | Data Source system identifier                         | `SYS_OLD`, `SYS_NEW`                |
| 4  | `subscription_date`         | Original subscription datetime                        | `2022-03-04 00:00:00`, `July 2018`  |
| 5  | `participant_age`           | Age at experiment start                               | `69`, `32`, `Unknown`               |
| 6  | `marketing_channel`         | Acquisition channel                                   | `DM`, `Direct mail`, `Online paid`  |
| 7  | `country_code`              | Country                                               | `NL`, `DE`, `GB`, `N/A`             |
| 8  | `postcode_area`             | Coarse location of subscription owner                 | `1012`, `SE-123 45`                 |
| 9  | `extra_info_per_year`       | Opt-in for extra "13th/14th" month                    | `0` (None), `1`, `2` (Both)         |
| 10 | `Add-ons`                   | Number of add-ons                                     | `0`, `1`, `3`, `0,0`                |
| 11 | `payment_method`            | Primary payment method                                | `Direct debit`, `Credit card`       |
| 12 | `failed_payments_12m`       | Count of failed payments in last 12 months            | `0`, `1`, `NaN`                     |
| 13 | `monthly_spend_estimated`   | Estimated monthly net spend (pre-tax)                 | `€15,50`, `EUR 15.50`, `1,250`      |
| 14 | `donation_share_charity`    | Share of monthly spend                                | `0.40`, `40`                        |
| 15 | `engagement_history`        | Recent engagement in semi-structured format           | `{"email_opens": "5"}`, `opens=3`   |
| 16 | `web_sessions_90d_raw`      | Web sessions in last 90 days                          | `0`, `5`, `10+`, `missing`          |
| 17 | `c_service_contacts_12m`    | Number of customer service contacts in last 12 months | `0`, `1`, `99`                      |
| 18 | `complaints_12m`            | Number of complaints in last 12 months                | `0`, `2`, `5`                       |
| 19 | `lifetime_wins`             | Total number of git card                              | `0`, `1`, `15`                      |
| 20 | `win_dates`                 | Dates of git card                                     | `2022-03-04 00:00:00`, `July 2018`  |
| 21 | `campaign_cohort`           | Randomized experiment group                           | `Control`, `Variant_A`, `Variant_B` |
| 22 | `treatment_sent_flag`       | Whether the campaign communication was actually sent  | `1`, `0`, `Y`, `N`                  |
| 23 | `offer_cost_eur`            | Marketing + incentive cost                            | `0`, `€20,00`, `20.0`               |
| 24 | `baseline_churn_risk_band`  | Legacy rule-based churn band at experiment start      | `Low`, `Medium`, `High`             |
| 25 | `historic_revenue_12m`      | Observed revenue last 12 months (before experiment)   | `€180,00`, `1.200,00`               |
| 26 | `churned`                   | Whether subscription churned in follow-up window      | `1`, `0`, `Y`, `N`                  |
| 27 | `churn_date`                | Date of churn (if any)                                | `2023-01-15`, `NA`                  |
| 28 | `observation_end_date`      | End date of follow-up observation                     | `2023-03-31`                        |
| 29 | `revenue_next_12m_observed` | Observed revenue (after exp) over 12-month follow-up  | `€600,00`, `0`, `NA`                |
### Assumptions
* The experiment assignment (`campaign_cohort`) is randomized at the subscription level.
* Some participants in Variant A/B did not actually receive the treatment due to 
  operational issues (`treatment_sent_flag`)


---	

# Process of data analysis

## Part A — Data Engineering & Pipeline

	(`src/data_ingestion.py`)
	* 1. Exploratory Data Analysis (EDA)
	* 2. Cleaning & Standardization
	* 3. Feature Engineering
	* 4. Reproducibility
	
1.1.  Check the raw file’s encoding and load it using the corresponding encoding.

1.2.  Some columns contain a mix of uppercase and lowercase letters; convert them all to lowercase

1.3.  Remove missing values
	  Replace ['unknown', 'missing'] with np.nan
	  First, check the proportion of missing values of each feature,
	  and drop any features whose proportion of missing values exceeds a specified threshold (e.g., 0.2).
	  Second, drop samples that contain at least one missing value.
	  
1.4.  Convert different date formats to a given date format, e.g,. '%Y-%m-%d'.
	  Create a new feature (subscription_month_length)to make use of this date information.
	  
1.5.  Replace "," with "." for feature 'add_ons'

1.6.  Convert all currency values to euros for:
	  'monthly_spend_estimated','offer_cost_eur','historic_revenue_12m','revenue_next_12m_observed'.
	  
1.7.  Transform the two formats into a single one for the donation_share_charity feature.

1.8.  Remove the '+' from the web_sessions_90d_raw feature; 
	  
1.9.  Remove outliers for 'service_contacts_12m'.

1.10.  Map {'true', 'y', '1', 'yes'} to 1, and {'false', 'f', '0', 'no'} to 0 for:
	  'treatment_sent_flag', 'churned'
	  
1.11. Replace "_" with " " for feature 'baseline_churn_risk_band'

1.12. Change str datatype to int for:
	  'participant_age','web_sessions_90d_raw'.
	  
1.13. Further feature engineering	
	    '''Only consider 3 situations: 
        'campaign_cohort'='Control' and 'treatment_sent_flag'=0;
        'campaign_cohort'='Variant_A' and 'treatment_sent_flag'=1;
        'campaign_cohort'='Variant_B' and 'treatment_sent_flag'=1;'''	
		Remove features: 'legacy_system_id','subscription_date','postcode_area','observation_end_date'	 
		
1.14. Examine the distribution of features with numeric values

1.15. Save the cleaned data to train.csv and test.csv

---	  

## Part B — Predictive Modeling 
(`src/models.py`)
	### 1. Churn Propensity Model
	
1.1. The models used to predict `churned`:
	 KNN, LogisticRegression, RandomForestClassifier
	 
1.2. The features not considered:
	 'customer_id','subscription_id','campaign_cohort','treatment_sent_flag','revenue_next_12m_observed'
	 
1.3. Perform encoding on categorical variables
	 Label encoding for ordinal data: 'baseline_churn_risk_band'
	 One hot encoding for nominal data: 'marketing_channel', 'country_code'
	 
1.4. Handle imbalanced data by oversampling (in the training set)

1.5. Standardize features

1.6. Grid search for hyperparameter tuning, (5-folds cross-validation is used)

1.7. Save the trained models, prediction accuracy (ROC AUC, PR AUC)

1.8. Explore the feature importance

---

## Part C — Causal & Uplift Analysis 
(`src/causal.py`)
	### 1. A/B Test Analysis
	
1.1. Check the number of customers in each group: Variant A, Variant B, and Control.

1.2. Check the distribution of 'churned'.

1.3. A two-proportions z-test will be used. 
	 It's a statistical hypothesis test that determines if there's a significant difference 
	 between the proportions (percentages) of a binary outcome (yes/no, success/fail) in two independent groups.
	 
1.4. Find: 
	For Variant A vs Control, p-value is 0.0. So the difference between Control and Variant A is significant.
	For Variant B vs Control, p-value is 0.893. So the difference between Control and Variant B is not significant.

### 2. Treatment Effect Heterogeneity (Segments)

2.1. Participant_age and baseline_churn_risk_band are used separately to group customers, 
	 and a two-proportions z-test is then applied within each group.
	 
2.2. Find:
	 Overall, the results show that experiment variant_a is effective in reducing churn. 
	 Customers aged above 50 appear to benefit more from variant_a than younger customers. 
	 However, when the legacy rule-based churn risk (baseline_churn_risk_band) is very high, 
	 variant_a no longer shows a meaningful effect.

---
## Part D — Engineering & Reproducibility

**Tech Stack:** 
* Follow good MLOps practices.
* Make the work easy to rerun from scratch.

**Final Structure:**
* `src/` for Python modules.
* `notebooks/` for exploratory work.
* `data/` for input data.
* `outputs/` for output data.

```text
├── README.md
├── design_doc.md
├── data/
│   ├── raw/
│   │   └── train.csv
│   │   └── test.csv
│   └── sample_data.csv
├── notebooks/
│   └── eda_and_experiments.ipynb
├── src/
│   ├── data_ingestion.py
│   ├── models.py
│   ├── ABtest.py
```

# More learning source
https://fastercapital.com/content/Price-modeling--Subscription-Pricing--Retaining-Customers-and-Maximizing-Revenue.html
