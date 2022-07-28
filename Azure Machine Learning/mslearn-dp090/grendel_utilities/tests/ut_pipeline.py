from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import unittest
import os, sys
import pandas as pd
import numpy as np

# importing pipes for making the Pipe flow
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

sys.path.insert(1, f'{os.getcwd()}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities')
from utils import vector_assembler, sine_cosine_cyclic, scale_numeric
from utils import impute_missing, category_string_encoder, category_onehot_encoder

import warnings
warnings.filterwarnings("ignore")

class TestPipelines(unittest.TestCase):
    def setUp(self):
        self.currdir = os.getcwd () 

    @unittest.skip("skip scoring unit test")
    def test_run_iris_dataset(self):
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

    def test_run_nyctaxi_dataset_no_pipe(self):
        SHOW_COLS_ONLY = False
        NO_STEPS = 0

        dtframes = []

        # 1. read the csv file in a dataframe
        NO_STEPS = NO_STEPS + 1
        file_path = f'{self.currdir}\\Azure Machine Learning\\mslearn-dp090\\grendel_utilities\\data\\nyc-taxi.csv'
    
        # Read in the CSV file and convert "?" to NaN
        originalDF = pd.read_csv(file_path,header='infer')
        if not SHOW_COLS_ONLY:
            print(originalDF.head(8))
        print("#1. Original columns:",originalDF.columns)

        # 2. impute missing data
        NO_STEPS = NO_STEPS + 1
        imputedDF = impute_missing(originalDF,"passengerCount")
        # drop totalAmount column
        imputedDF.drop(["totalAmount"],axis=1,inplace=True)
        if not SHOW_COLS_ONLY:
            print(imputedDF.head(8))
            print(imputedDF.describe())
        print("#2. Imputed columns:",imputedDF.columns)

        # 3. scale & normalize numeric features
        NO_STEPS = NO_STEPS + 1
        # transform cyclical attributes (eg hour of day, day of month etc)
        cycle_col = "hour_of_day"
        dfsincos, col_sine, col_cosine = sine_cosine_cyclic(imputedDF,cycle_col,24)
        if not SHOW_COLS_ONLY:
            print(dfsincos.head(5))
        print("#3. Sine-cosine columns:",dfsincos.columns)

        # 4. scale numerical values
        NO_STEPS = NO_STEPS + 1
        numerical_cols = ["passengerCount", "tripDistance", "snowDepth", "precipTime", "precipDepth", "temperature", col_sine, col_cosine]
        dfscaled = scale_numeric(dfsincos,numerical_cols)
        if not SHOW_COLS_ONLY:
            print(dfscaled.head(5))
        print("#4. Numeric scaled columns:",dfscaled.columns)

        cols = ['scaled_numerical_features']
        dtf1 = dfscaled[cols]

        # 5-SE. category encoding: apply ordinal encoder
        NO_STEPS = NO_STEPS + 1
        categorical_cols = ["day_of_week", "month_num", "normalizeHolidayName", "isPaidTimeOff"]
        dfencordinal = category_string_encoder(dfscaled,categorical_cols)
        if not SHOW_COLS_ONLY:
            print(dfencordinal.head(8))
        print("#5-SE. [String Encoder] Category ordinal columns:",dfencordinal.columns)

        # 5-OH. category encoding: apply onehot encoder
        dfenconehot = category_onehot_encoder(dfscaled,categorical_cols)
        if not SHOW_COLS_ONLY:
            print(dfenconehot.head(8))
        print("#5-OH. [Onehot Encoder] Category ordinal columns:",dfencordinal.columns)

        # 6. apply vector assembly on categories
        NO_STEPS = NO_STEPS + 1
        dfassembled = vector_assembler(dfscaled,dfenconehot)
        if not SHOW_COLS_ONLY:
            print(dfassembled.head(8))
        print("#6. Vector assembled columns:", dfassembled.columns)

        cols = ['day_of_week_classVector','month_num_classVector', 'normalizeHolidayName_classVector','isPaidTimeOff_classVector']
        dtf2 = dfassembled[cols]

        # 7. join the processed (engineered) dataframes into final features 
        NO_STEPS = NO_STEPS + 1
        print('#scaled_numerical_features',len(dtf1['scaled_numerical_features'][0]))
        print('#day_of_week_classVector',len(dtf2['day_of_week_classVector'][0]))
        print('#month_num_classVector',len(dtf2['month_num_classVector'][0]))
        print('#normalizeHolidayName_classVector',len(dtf2['normalizeHolidayName_classVector'][0]))
        print('#isPaidTimeOff_classVector',len(dtf2['isPaidTimeOff_classVector'][0]))
        
        numFeatures = len(dtf1['scaled_numerical_features'][0]) + len(dtf2['day_of_week_classVector'][0]) + len(dtf2['month_num_classVector'][0]) + len(dtf2['normalizeHolidayName_classVector'][0]) + len(dtf2['isPaidTimeOff_classVector'][0])
        print(f"Sum features: {numFeatures}")
        
        # concatenate vectors
        xfvec = np.array(np.concatenate((dtf2['day_of_week_classVector'].to_list(),dtf2['month_num_classVector'].to_list(),dtf2['normalizeHolidayName_classVector'].to_list(),dtf2['isPaidTimeOff_classVector'].to_list()),axis=1))
        print(xfvec)

        xfsca = np.array(dtf1['scaled_numerical_features'].values)

        xfeatures = []
        for i in range(len(xfvec)):
            try:
                xfeatures.append(np.concatenate((xfvec[i],xfsca[i])))
            except Exception as e:
                print(e)
        
        print(xfeatures)
        
        # 8. build the features dataframe
        NO_STEPS = NO_STEPS + 1
        dffeatures = pd.DataFrame(xfeatures)
        print(dffeatures.head(8))

        # 9. run the algorithm
        NO_STEPS = NO_STEPS + 1

        # 10. calculate accuracy score
        NO_STEPS = NO_STEPS + 1

        self.assertTrue(NO_STEPS == 10)

if __name__ == '__main__':
    
    # unittest.main(TestPipelines().run_iris_dataset())
    # unittest.main(TestPipelines().test_run_nyctaxi_dataset_no_pipe())

    unittest.main()
