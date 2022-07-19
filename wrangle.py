# importing needed libraries/modules
import os
import pandas as pd
import numpy as np

# importing visualization libraries 
import seaborn as sns
import matplotlib.pyplot as plt

# importing sql 
import env
from env import user, password, host, get_connection

# sklearn train, test, and split function
from sklearn.model_selection import train_test_split



def get_zillow_dataset():
    # importing "cached" dataset
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=[0])

    # if not in local folder, let's import from MySQL and create a .csv file
    else:
        # query necessary to pull the 2017 properties table from MySQL
        query = ''' 
        SELECT
                bedroomcnt as bedroom_count,
                bathroomcnt as bath_count,
                calculatedfinishedsquarefeet as finished_sq_feet,
                taxvaluedollarcnt as tax_dollar_value,
                yearbuilt as year_built,
                taxamount,
                fips
        FROM properties_2017
        JOIN propertylandusetype using (propertylandusetypeid)
        WHERE propertylandusedesc = "Single Family Residential"
        '''
        db_url = f'mysql+pymysql://{user}:{password}@{host}/telco_churn'
        # creating the zillow dataframe using Pandas' read_sql() function
        df = pd.read_sql(query, db_url)
        df.to_csv(filename)
        return df



# Preparing/cleaning zillow dataset
# focus is dropping Null values and changing column types 

def clean_zillow_dataset(df):
    # dropping null values in dataset (where <=1% makeup nulls in ea. feature/column)
    df = df.dropna()

    # converting "bedroom_count" "year_built", and "fips" columns to int type
    df["bedroom_count"] = df["bedroom_count"].astype("int").round()
    df["year_built"] = df["year_built"].astype("int")
    df["fips"] = df["fips"].astype("int")

    # lastly, return the cleaned dataset
    return df

# function for handling outliers in the dataset
def zillow_outliers(df):
    df = df[df["bath_count"] <= 6]
    df = df[df["bedroom_count"] <= 6]
    df = df[df["finished_sq_feet"] <= 8000]
    df = df[df["tax_dollar_value"] <= 1_500_000]

    return df

def train_validate_test_split(df):
    train_and_validate, test = train_test_split(
    df, test_size=0.2, random_state=123)
    
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=123)

    return train, validate, test