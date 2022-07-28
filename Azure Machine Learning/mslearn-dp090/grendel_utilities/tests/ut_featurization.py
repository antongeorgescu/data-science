#!/usr/bin/env python3
#import packages
import os,sys
import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from pathlib import Path

import seaborn as sns

sys.path.insert(1, f'{os.getcwd()}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities')
from utils import vector_assembler, sine_cosine_cyclic

import warnings
warnings.filterwarnings("ignore")

# references
# https://datagy.io/sklearn-one-hot-encode/

class TestDataFeaturization(unittest.TestCase):
    def setUp(self):
        self.currdir = os.getcwd () 

    def test_get_and_fix_dataset(self):
        file_path = f'{self.currdir}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities\\data\\imports-85.data'
    
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
        
        # save the fixed dataset to data folder
        file_path = f'{self.currdir}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities\\data\\imputed-85.data'
        obj_df.to_csv(file_path)

        self.assertTrue(Path(file_path).stat().st_size > 0)

    # def encode_onehot(self,df):
    #     return

    def test_run_multivector_assembler(self):
        v1 = pd.DataFrame([[1,2,3,4,5],[11,12,13,14,15]],columns=["a1","a2","a3","a4","a5"])
        v2 = pd.DataFrame([[6,7,8],[16,17,18]],columns=["b1","b2","b3"])
        v3 = pd.DataFrame([[9,10],[19,20]],columns=["c1","c2"])

        vR = vector_assembler(v1,v2,v3)
        print(f"\n*** Results for {self._testMethodName}")
        print(vR.head())
        self.assertTrue(len(vR.columns) == 10)

    def test_normalize_label_encoder2(self):
        le = LabelEncoder()
        le.fit([1, 1, 2, 6])
        print(le.classes_)
        ar = le.fit_transform([1, 1, 2, 6])
        print(f"\n*** Results for {self._testMethodName}")
        print(ar)
        print(le.inverse_transform(ar))
        self.assertTrue(ar.size == 4)

    def test_normalize_label_encoder(self):
        file_path = f'{self.currdir}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities\\data\\encoding_test_data.csv'
    
        # Read in the CSV file and convert "?" to NaN
        df = pd.read_csv(file_path,header='infer' )
        print(df.head(8))

        enc = LabelEncoder()
        cols = ["Sex","Blood", "Study","Bday"]
        for col in cols:
            df[col] = enc.fit_transform(df[col])
        
        print(f"\n*** Results for {self._testMethodName}")
        print(df.head(8))
        self.assertTrue(df.columns.size == 7)

    def test_normalize_ordinal_encoder(self):
        file_path = f'{self.currdir}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities\\data\\encoding_test_data.csv'
    
        # Read in the CSV file and convert "?" to NaN
        df = pd.read_csv(file_path,header='infer' )
        print(df.head(8))

        cols = ["Sex","Blood", "Study","Bday"]

        enc = OrdinalEncoder()
        df[cols] = enc.fit_transform(df[cols])
        print(f"\n*** Results for {self._testMethodName}")
        print(df.head(8))
        self.assertTrue(df.columns.size == 7)

    def test_normalize_getdummies_encoder(self):
        file_path = f'{self.currdir}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities\\data\\encoding_test_data.csv'
    
        # Read in the CSV file and convert "?" to NaN
        df = pd.read_csv(file_path,header='infer' )
        print(df.head(8))

        cols = ["Sex","Study","Bday"]
        
        empty_list = [[None]] * len(df)
        
        df_encoded = pd.DataFrame()
        
        for c in cols:
            col_name = f'{c}_classVector'

            df_temp = pd.DataFrame()
            df_temp[[col_name]] = empty_list

            df_dummy = pd.get_dummies(df, columns=[c])
            mask = df_dummy.columns.str.contains(f'{c}_*')
            x_vector = df_dummy.loc[:,mask].to_numpy()
            for idx, row in df_dummy.iterrows():
                df_temp.loc[idx,col_name] = x_vector[idx]
            df_encoded[[col_name]] = df_temp[[col_name]]
        
        print(f"\n*** nResults for {self._testMethodName}")
        print(df_encoded.head(5))
        self.assertTrue(df_encoded.size == 21)

    def test_normalize_onehot_encoder_2(self):
        file_path = f'{self.currdir}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities\\data\\encoding_test_data.csv'
    
        # Read in the CSV file and convert "?" to NaN
        df = pd.read_csv(file_path,header='infer' )
        print(df.head(8))

        # df = sns.load_dataset('penguins')
        # print(df.head())

        cols = ["Study","Sex","Bday"]
                
        enc = OneHotEncoder(sparse=True)
        empty_list = [[None]] * len(df)
        
        df_encoded = pd.DataFrame()
        for c in cols:
            col_name = f'{c}_classVector'

            df_temp = pd.DataFrame()
            x_vector = enc.fit_transform(df[[c]]).toarray()  
            
            df_temp[[col_name]] = empty_list
            for idx, row in df.iterrows():
                df_temp.loc[idx,col_name] = x_vector[idx]

            print(enc.categories_)
        
            print(df_temp.head(5))

            df_encoded[[col_name]] = df_temp[[col_name]]
        print(f"\n*** Results for {self._testMethodName}")
        print(df_encoded)
        self.assertTrue(df_encoded.columns.size == 3)

    def test_normalize_onehot_encoder(self):
        df = sns.load_dataset('penguins')
        print(df.head())
        print(df.describe())

        cols = ["species","sex","island"]
                
        enc = OneHotEncoder()
        empty_list = [[None]] * len(df)
        
        df_encoded = pd.DataFrame()
        for c in cols:
            col_name = f'{c}_classVector'

            df_temp = pd.DataFrame()
            x_vector = enc.fit_transform(df[[c]]).toarray()  
            
            df_temp[[col_name]] = empty_list
            for idx, row in df.iterrows():
                df_temp.loc[idx,col_name] = x_vector[idx]

            print(enc.categories_)
        
            print(df_temp.head(5))

            df_encoded[[col_name]] = df_temp[[col_name]]
        
        print(f"\n*** Results for {self._testMethodName}")
        print(df_encoded)
        self.assertTrue(df_encoded.columns.size == 3)

if __name__ == '__main__':
    # unittest.main(TestDataFeaturization().get_and_fix_dataset())
    # unittest.main(TestDataFeaturization().run_multivector_assembler())
    # unittest.main(TestDataFeaturization().normalize_label_encoder())
    # unittest.main(TestDataFeaturization().normalize_ordinal_encoder())
    # unittest.main(TestDataFeaturization().normalize_onehot_encoder())
    # unittest.main(TestDataFeaturization().normalize_onehot_encoder_2())
    # unittest.main(TestDataFeaturization().normalize_getdummies_encoder())
    
    unittest.main()
