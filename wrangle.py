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
    # dropping null values in dataset (where <=2% makeup nulls in ea. feature/column)
    df = df.dropna()

    # converting "bedroom_count" "year_built", and "fips" columns to int type
    df[["bedroom_count", "year_built", "fips"]] = df[["bedroom_count", "year_built", "fips"]].astype("int")
    # lastly, return the cleaned dataset
    return df
