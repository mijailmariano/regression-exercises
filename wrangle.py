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


'''function that will either 
1. import the zillow dataset from MySQL or 
2. import from cached .csv file'''
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
                taxvaluedollarcnt as home_value,
                yearbuilt as year_built,
                taxamount as tax_amount,
                fips
        FROM properties_2017
        JOIN propertylandusetype using (propertylandusetypeid)
        WHERE propertylandusedesc = "Single Family Residential"
        '''
        db_url = f'mysql+pymysql://{user}:{password}@{host}/zillow'
        # creating the zillow dataframe using Pandas' read_sql() function
        df = pd.read_sql(query, db_url)
        df.to_csv(filename)
        return df


'''Preparing/cleaning zillow dataset
focus is dropping Null values and changing column types'''
def clean_zillow_dataset(df):
    # dropping null values in dataset (where <=1% makeup nulls in ea. feature/column)
    df = df.dropna()

    # converting "bedroom_count" "year_built", and "fips" columns to int type
    df["bedroom_count"] = df["bedroom_count"].astype("int").round()
    df["year_built"] = df["year_built"].astype("int")
    df["fips"] = df["fips"].astype("int")
    
    # rearranging columns for easier readibility
    df = df[[
        'bedroom_count',
        'bath_count',
        'finished_sq_feet',
        'year_built',
        'fips',
        'tax_amount',
        'home_value']]

    # lastly, return the cleaned dataset
    return df


# function for handling outliers in the dataset
def zillow_outliers(df):
    df = df[df["bath_count"] <= 6]
    df = df[df["bedroom_count"] <= 6]
    df = df[df["finished_sq_feet"] <= 8000]
    df = df[df["home_value"] <= 1_500_000]

    return df

'''Function created to split the initial dataset into train, validate, and test sub-datsets'''
def train_validate_test_split(df):
    train_and_validate, test = train_test_split(
    df, test_size=0.2, random_state=123)
    
    train, validate = train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=123)

    return train, validate, test


'''Function takes in a dataframe and plots all variables against one another using sns.pairplot function. 
This function also shows the line-of-best-fit for ea. plotted variables'''
def plot_variable_pairs(df):
    g = sns.pairplot(data = df.sample(1000), corner = True, kind="reg", diag_kind = "kde", plot_kws={'line_kws':{'color':'red'}})
    plt.show()


'''function takes in a dataframe and list, and plots them against target variable w/'line-of-best-fit'''
def plot_variable_pairs(train_df, x_features_lst):
    for col in x_features_lst:
        plt.figure(figsize = (10, 4))
        sns.set(font_scale = 1)

        # plotting ea. feature against target variable with added "independent jitter" for easier visual
        ax = sns.regplot(train_df[[col]].sample(2000), \
        train_df[["home_value"]].sample(2000), \
        x_jitter = 1, # adding superficial noise to independent variables
        line_kws={
            "color": "red", 'linewidth': 1.5})
        
        ax.figure.set_size_inches(18.5, 8.5)
        sns.despine()
        # removing scientific notations
        ax.ticklabel_format(style = "plain")
        
        # removing x_axis label
        ax.set_xlabel(None)

        plt.title(col)
        plt.show()







'''function for plotting categorical or discrete/low feature option columns'''
def plot_discrete(df, feature_lst):
    for column in df[[feature_lst]]:
        plt.figure(figsize=(12, 6))
        sns.set(font_scale = 1)
        ax = sns.countplot(x = column, 
                        data = df,
                        palette = "crest_r",
                        order = df[column].value_counts().index)
        ax.bar_label(ax.containers[0])
        ax.set(xlabel = None)
        plt.title(column)
        plt.show()


'''function for plotting continuous/high feature option columns'''
def plot_continuous(df, feature_lst):
    for column in df[[feature_lst]]:
        plt.figure(figsize=(12, 6))
        ax = sns.distplot(x = df[feature_lst], 
                        bins = 50,
                        kde = True)
        ax.set(xlabel = None)
        plt.axvline(df[column].median(), linewidth = 2, color = 'purple', alpha = 0.4, label = "median")
        plt.title(column)
        plt.legend()
        plt.show()


'''plotting the target variable'''
def plot_target(df):
    plt.figure(figsize = (12, 5))
    sns.set(font_scale = .8)
    ax = sns.histplot(df, bins = 20, kde = True)

    ax.ticklabel_format(style = "plain") # removing axes scientific notation 
    ax.bar_label(ax.containers[0])

    plt.axvline(df.median(), linewidth = 2, color = 'purple', alpha = 0.4, label = "median")
    plt.legend()
    plt.show()


'''Plotting features against target variable w/line-of-best-fit'''
def features_and_target(df):
    cols = df.columns.to_list()
    for col in cols:
        plt.figure(figsize = (10, 4))
        sns.set(font_scale = 1)

        # plotting ea. feature against target variable with added "independent jitter" for easier visual
        ax = sns.regplot(df[[col]].sample(2000), \
        df["home_value"].sample(2000), \
        
        # adding superficial noise to independent variables to help visualize the individual plots
        x_jitter = 1, \
        line_kws={
            "color": "red", 'linewidth': 1.5})
        
        ax.figure.set_size_inches(18.5, 8.5)
        sns.despine()
        # removing scientific notations
        ax.ticklabel_format(style = "plain")
        
        # removing x_axis label
        ax.set_xlabel(None)

        plt.title(col)
        plt.show()


'''Function to compare Model vs. Baseline Sum-of-Squares'''
# note: the lower the SSE, the lower the predicted error from actual observations & the better the model represents the "actual" predictions
def compare_sum_of_squares(SSE_baseline, SSE_model):
    if SSE_model >= SSE_baseline:
        print("Model DOES NOT outperform baseline.")
    else:
        print("Model outperforms baseline!")