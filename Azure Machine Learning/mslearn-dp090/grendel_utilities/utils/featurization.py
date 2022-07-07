import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

def vector_assembler(*dfvectors):
    if len(dfvectors) == 0:
        raise Exception('No parameters passed')
        return None
    if len(dfvectors) == 1:
        return dfvectors[0]    
    
    dfresult = dfvectors[0]
    for dfv in dfvectors[1:]:
        dfresult = pd.concat([dfresult,dfv],axis=1,join='inner')
            
    # dfresult = pd.DataFrame(aR, columns = ['Column_A','Column_B','Column_C'])
    return dfresult

def get_sine(value, max_value):
    sine =  np.sin(value * (2.*np.pi/max_value))
    return (sine)

def get_cosine(value, max_value):
    cosine = np.cos(value * (2.*np.pi/max_value))
    return (cosine)

def sine_cosine_cyclic(df,col_name,max_range):
    # transform repeatitive attributes into cyclical features
    sin_col = f'{col_name}_sine'
    cos_col = f'{col_name}_cosine'
    df[sin_col] = df[col_name]
    df[cos_col] = df[col_name]
    dfs = df.apply(lambda c: get_sine(c,max_range) if c.name == sin_col else c)
    dfsc = dfs.apply(lambda c: get_cosine(c,max_range) if c.name == cos_col else c)

    dfsc.drop(col_name,axis=1,inplace=True)

    return dfsc, sin_col, cos_col

def impute_missing(df,col_name):
    # Missing values is represented using NaN and hence specified. If it 
    # is empty field, missing values will be specified as ''
    imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
    df["passengerCount"] = imputer.fit_transform(df["passengerCount"].values.reshape(-1,1))[:,0]
    return df

def scale_numeric(df,cols):
    xnum = df[cols].to_numpy()
    # print(xnum)
    
    scaler = MinMaxScaler()
    scaler.fit(xnum)
    # print(scaler.data_max_)

    scldnum = scaler.transform(xnum)
    
    # update the existing dataframe and remove the un-scaled columns
    for col in cols:
        df = df.drop(col,axis=1)

    # add the new scaled vector under column 'scaled_numerical_features'
    xdata = [ (i,scldnum[i])  for i in range(0,len(scldnum)-1) ]
    dfv = pd.DataFrame(data=xdata,columns=['temp_index','scaled_numerical_features'])
    
    dfscalenum = pd.concat([df,dfv],axis=1)
    dfscalenum = dfscalenum.drop('temp_index',axis=1)

    return dfscalenum

def category_dummy_encoder(df,cols):
    dfencodeonehot = df
    for c in cols:
        dfencodeonehot = pd.get_dummies(dfencodeonehot, columns=[c])
        # mask = dfencodeonehot.columns.str.contains('make_*')
        # x_vector = dfencodeonehot.loc[:,mask].to_numpy()
        x_vector = dfencodeonehot.to_numpy()
        col_name = f'{c}_classVector'
        
        empty_list = [[None]] * len(df)
        df_col = pd.DataFrame(empty_list)
                        
        for idx, row in dfencodeonehot.iterrows():
            df_col.loc[idx,col_name] = x_vector[idx]
            # dfencodeonehot.loc[idx,col_name] = x_vector[idx]
        # print(dfencodeonehot.head(5))
    
    return dfencodeonehot

def category_onehot_encoder(df,cols):
    
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

        df_encoded[[col_name]] = df_temp[[col_name]]
    
    return df_encoded

def category_string_encoder(df,cols,type='ordinal'):
    df_encoded = pd.DataFrame()
    if (type == 'ordinal'):
        enc = OrdinalEncoder()
        df[cols] = enc.fit_transform(df[cols])
    df_encoded = df
    return df_encoded
