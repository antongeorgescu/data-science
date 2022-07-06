import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from utils import vector_assembler, sine_cosine_cyclic, encode_category, scale_numeric, impute_missing

currdir = os.getcwd () 

if __name__ == "__main__":
    file_path = f'{currdir}\\Azure Machine Learning\\mslearn-dp090\\nyc_taxi_before_impute.csv'
    
    # impute missing numerical values
    originalDF = pd.read_csv(file_path)
    print(originalDF.describe())

    imputedDF = impute_missing(originalDF,"passengerCount")

    # drop totalAmount column
    imputedDF.drop(["totalAmount"],axis=1,inplace=True)
    
    print(imputedDF.head(5))
    print(imputedDF.describe())

    # transform cyclical attributes (eg hour of day, day of month etc)
    cycle_col = "hour_of_day"
    dfsincos, col_sine, col_cosine = sine_cosine_cyclic(imputedDF,cycle_col,24)
    print(dfsincos.head(5))

    # scale numerical values
    numerical_cols = ["passengerCount", "tripDistance", "snowDepth", "precipTime", "precipDepth", "temperature", col_sine, col_cosine]
    dfscaled = scale_numeric(dfsincos,numerical_cols)
    print(dfscaled.head(5))

    # encode categorical features
    categorical_cols = ["day_of_week", "month_num", "normalizeHolidayName", "isPaidTimeOff"]
    dfencoded = encode_category(dfscaled,categorical_cols)
    print(dfencoded.head(5))
