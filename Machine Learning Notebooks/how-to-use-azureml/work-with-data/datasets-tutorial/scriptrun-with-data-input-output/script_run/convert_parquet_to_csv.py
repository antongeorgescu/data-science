
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import pandas
import os

curr_dir = os.getcwd()
parquet_file = f'{curr_dir}\\data_samples\\car_prices.parquet'
df = pandas.read_parquet(parquet_file,engine='fastparquet')
print(df.head())
df.to_csv(f'{curr_dir}\\data_samples\\car_prices.csv')
df.pop('price')
df1 = df.dropna()
print(f'before cleaning:{len(df)},after cleaning:{len(df1)}')
df2 = df1.sample(n=50)
print(df2.head())
df2.to_csv(f'{curr_dir}\\data_samples\\car_prices_to_predict.csv')
