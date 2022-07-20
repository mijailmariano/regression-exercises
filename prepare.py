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



'''function for plotting categorical or discrete/low feature option columns'''
def plot_discrete(df, column_lst):
    for column in df[[column_lst]]]:
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
def plot_continuous(df, column_lst):
    for column in df[[column_lst]]:
        plt.figure(figsize=(12, 6))
        ax = sns.distplot(x = df[column_lst], 
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