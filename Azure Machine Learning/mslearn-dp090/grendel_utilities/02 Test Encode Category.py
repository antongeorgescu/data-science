import pandas as pd
import numpy as np
import os

from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

currdir = os.getcwd () 

def encode_onehot(df,cols):
    dfencodeonehot = df
    for c in cols:
        dfencodeonehot = pd.get_dummies(dfencodeonehot, columns=[c])
        mask = dfencodeonehot.columns.str.contains('make_*')
        x_vector = dfencodeonehot.loc[:,mask].to_numpy()
        col_name = f'{c}_classVector'
        
        empty_list = [[None]] * len(df)
        df_col = pd.DataFrame(empty_list)
                        
        for idx, row in dfencodeonehot.iterrows():
            df_col.loc[idx,col_name] = x_vector[idx]
            # dfencodeonehot.loc[idx,col_name] = x_vector[idx]
        print(dfencodeonehot.head(5))
    
    return dfencodeonehot

def encode_pipelines(df,feature_cols):
    column_trans = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                        ['fuel_type', 'make', 'drive_wheels']),
                        (OrdinalEncoder(), ['aspiration']),
                        remainder='passthrough')

    print(column_trans)

if __name__ == "__main__":
    file_path = f'{currdir}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities\\data\\imports-85.data'
    
    # Define the headers since the data does not have any
    headers = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
                "num_doors", "body_style", "drive_wheels", "engine_location",
                "wheel_base", "length", "width", "height", "curb_weight",
                "engine_type", "num_cylinders", "engine_size", "fuel_system",
                "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
                "city_mpg", "highway_mpg", "price"]

    # Read in the CSV file and convert "?" to NaN
    df = pd.read_csv(file_path,header=None, names=headers, na_values="?" )
    print(df.head(8))

    # as we want to encode only the categorical variables, we are going to include only the object columns in our dataframe.
    print(df.dtypes)
    obj_df = df.select_dtypes(include=['object']).copy()
    obj_df.head()

    # clean up the null values
    print(obj_df[obj_df.isnull().any(axis=1)])
    print(obj_df["num_doors"].value_counts())

    # fill the null values with the most common value (num_doors = 4)
    obj_df = obj_df.fillna({"num_doors": "four"})

    # encode the categories
    cols = ["make","fuel_type","aspiration","num_doors","body_style","drive_wheels","engine_location","engine_type","fuel_system"]
    dfonehot = encode_onehot(obj_df,cols)
    # print(dfonehot.head(7))

    cols = ['fuel_type', 'make', 'aspiration', 'highway_mpg', 'city_mpg','curb_weight', 'drive_wheels']
    encode_pipelines(obj_df,cols)

    cols = ['fuel_type', 'make', 'aspiration', 'highway_mpg', 'city_mpg','curb_weight', 'drive_wheels']
    enc = OneHotEncoder(handle_unknown='ignore')

