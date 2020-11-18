#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(self):
        pass
    
    def data_prep_check(self, df):
        null_val = df.isnull().sum()
        null_val_percent = 100 * null_val / len(df)

        unique_vals_count = df.nunique(dropna=True)
        unique_vals = pd.Series({c: df[c].unique() for c in df})

        df_table = pd.concat([null_val, null_val_percent,unique_vals_count,unique_vals], axis=1)
        df_table = df_table.rename(columns = {0 : 'Null Values', 1 : '% of Total Values', 
                                              2 : 'Total Unique Values', 3 : 'Unique Values'})

        df_table['Data Type'] = df.dtypes
        df_table = df_table.sort_values('% of Total Values', ascending=False).round(1)

        print("The dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " rows.\n"      
              "There are " + str(df_table[df_table.iloc[:,1] != 0].shape[0]) + " columns that have missing values.")

        return df_table


    def convert_col_types(self, df):

        numeric_cols = ['household_children', 'household_adults']    # values of these columns are capped at 3
        convert_to_str = list(df.select_dtypes(include=['float64']).drop(numeric_cols, axis=1).columns)
        df[convert_to_str] = df[convert_to_str].astype('category')

        categorical_cols = list(df.select_dtypes(include=['object','category']).columns)
        opinion_cols = [col for col in df.columns if 'opinion' in col]

        return df, numeric_cols, categorical_cols, opinion_cols


    def impute_na(self, df, col_type, strategy = 'most_frequent'):
        """Takes a dataframe and list of columns (default set to columns of 'float64' type), 
           and replaces NaN values with 'most_frequent'(default) of the column. 
           Other available strategies: 'mean', 'median'
           """

        num_imputer = SimpleImputer(strategy = strategy)
        df[col_type] = num_imputer.fit_transform(df[col_type])
        return df


    def replace_unknown(self, df, col_type, fill_text = "unknown"):
        """Takes a dataframe and list of columns (default set to columns of 'object' type), 
           and replaces NaN values with 'Unknown' or user-defined text to a create a new category"""

        text_imputer = SimpleImputer(strategy='constant', fill_value = fill_text)
        df[col_type] = text_imputer.fit_transform(df[col_type])
        return df


    def del_high_corr_feat(self, df, threshold = 0.95):
        corr_matrix = df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        cols_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        df = df.drop(df[cols_to_drop], axis=1)
        return df, cols_to_drop, threshold
    
    
    def clean_colnames(self, df):
        df.columns = df.columns.str.strip().str.lower().str.replace(' - ','_to_').str.replace(' ','_').str.replace('+','_and_above').str.replace('.0','').str.replace('-','_').str.replace(',','').str.replace('<','less_than').str.replace('>','more_than')
        
        return df
    
    
    def preprocess_data(self, df):
        df, numeric_cols, categorical_cols, opinion_cols = self.convert_col_types(df)

        df = self.impute_na(df, numeric_cols, strategy = 'most_frequent')
        df = self.replace_unknown(df, categorical_cols, fill_text = "unknown")
        df = pd.get_dummies(df, columns = categorical_cols, dummy_na=False, drop_first=True,dtype=int)
        df = self.clean_colnames(df)
        return df
    

    def corr_plot(self, df):
        # Compute the correlation matrix
        corr = df.apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1)

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask = mask, cmap = cmap, 
                    center = 0, vmax = 1, vmin = -1,
                    square = True, linewidths = .5, 
                    cbar_kws = {"shrink": .5})
        sns.set_theme(style="white")


