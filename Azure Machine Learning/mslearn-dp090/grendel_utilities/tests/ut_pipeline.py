from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import unittest
import os, sys
import pandas as pd

# importing pipes for making the Pipe flow
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

sys.path.insert(1, f'{os.getcwd()}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities')
from utils import vector_assembler, sine_cosine_cyclic, scale_numeric
from utils import impute_missing, category_string_encoder, category_onehot_encoder

import warnings
warnings.filterwarnings("ignore")

class TestPipelines(unittest.TestCase):
    def __init__(self):
        self.currdir = os.getcwd () 

    def run_iris_dataset(self):
        # import some data within sklearn for iris classification
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target

        # Splitting data into train and testing part
        # The 25 % of data is test size of the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
        
        # pipe flow is :
        # PCA(Dimension reduction to two) -> Scaling the data -> DecisionTreeClassification
        pipe = Pipeline([('pca', PCA(n_components = 2)), ('std', StandardScaler()), ('decision_tree', DecisionTreeClassifier())], verbose = True)

        # fitting the data in the pipe
        pipe.fit(X_train, y_train)

        # to see all the hyper parameters
        print(pipe.get_params())

        # scoring data
        
        print(accuracy_score(y_test, pipe.predict(X_test)))

    def run_nyctaxi_dataset_no_pipe(self):
        # 1. read the csv file in a dataframe
        file_path = f'{self.currdir}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities\\data\\nyc-taxi.csv'
    
        # Read in the CSV file and convert "?" to NaN
        originalDF = pd.read_csv(file_path,header='infer')
        res = input("Display original dataset (Y/n)")
        if res.upper() == 'Y':
            print(originalDF.head(8))
        print("Original columns:",originalDF.columns)

        # 2. impute missing data
        imputedDF = impute_missing(originalDF,"passengerCount")
        # drop totalAmount column
        imputedDF.drop(["totalAmount"],axis=1,inplace=True)
        res = input("Display imputed dataset (Y/n)")
        if res.upper() == 'Y':
            print(imputedDF.head(8))
            print(imputedDF.describe())
        print("Imputed columns:",imputedDF.columns)

        # 3. scale & normalize numeric features
        # transform cyclical attributes (eg hour of day, day of month etc)
        cycle_col = "hour_of_day"
        dfsincos, col_sine, col_cosine = sine_cosine_cyclic(imputedDF,cycle_col,24)
        res = input("Display sine-cosine dataset (Y/n)")
        if res.upper() == 'Y':
            print(dfsincos.head(5))
        print("Sine-cosine columns:",dfsincos.columns)

        # scale numerical values
        numerical_cols = ["passengerCount", "tripDistance", "snowDepth", "precipTime", "precipDepth", "temperature", col_sine, col_cosine]
        dfscaled = scale_numeric(dfsincos,numerical_cols)
        res = input("Display numeric scaled dataset (Y/n)")
        if res.upper() == 'Y':
            print(dfscaled.head(5))
        print("Numeric scaled columns:",dfscaled.columns)

        # 4. category encoding: apply ordinal encoder
        cols = ["day_of_week", "month_num", "normalizeHolidayName", "isPaidTimeOff"]
        dfencordinal = category_string_encoder(dfscaled,cols)
        res = input("Display category ordinal encoded dataset (Y/n)")
        if res.upper() == 'Y':
            print(dfencordinal.head(8))
        print("Category ordinal columns:",dfencordinal.columns)

        # 5. category encoding: apply onehot encoder
        dfenconehot = category_string_encoder(dfscaled,cols)
        res = input("Display category onehot encoded dataset (Y/n)")
        if res.upper() == 'Y':
            print(dfenconehot.head(8))
        print("Category onehot encoded columns:", dfenconehot.columns)

        # 6. apply vector assembly on categories
        dfassembled = vector_assembler(dfscaled,dfenconehot)
        res = input("Display vector assembled dataset (Y/n)")
        if res.upper() == 'Y':
            print(dfassembled.head(8))
        print("Vector assembled columns:", dfassembled.columns)

        # 7. run the algorithm

        # 8. calculate accuracy score

        return

if __name__ == '__main__':
    # unittest.main(TestPipelines().run())
    # unittest.main(TestPipelines().run_iris_dataset())
    unittest.main(TestPipelines().run_nyctaxi_dataset_no_pipe())
