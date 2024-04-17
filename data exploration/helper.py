import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler


def extract_iso_values(json_str:str, key:str ) -> list:
    """
    This function takes in a json string and a key, and returns a list of the values for the specified key in the json string.
    """
    try:
        data_list = json.loads(json_str)
        iso_values = [item[key] for item in data_list]
        return iso_values
    except json.JSONDecodeError as e:
        return ["error"]  
    except KeyError as e:
        return ["error"]   
    except Exception as e:
        return ["error"] 

def extract_iso_values_to_string(json_str:str, key:str ) -> str:
    """
    This function takes in a json string and a key, and returns a list of the values for the specified key in the json string.
    """
    try:
        data_list = json.loads(json_str)
        iso_values = ','.join(item[key] for item in data_list)
        return iso_values
    except json.JSONDecodeError as e:
        return "error"  
    except KeyError as e:
        return "error"   
    except Exception as e:
        return "error" 

def extract_first_iso_value(json_str:str, key:str) -> str:
    """
    This function takes in a json string and a key, and returns the first value for the specified key in the json string.
    """
    try:
        return extract_iso_values(json_str, key)[0]
    except Exception as e:
        return None


def convert_datestring_to_days_since_2000(date: str) -> int:
    try:
        year = int(date[:4])
        month = int(date[5:7])
        day = int(date[8:])
    except Exception as e:
        return 0
    given_date = datetime(year, month, day)

    start_date = datetime(2000, 1, 1)

    delta = given_date - start_date
    return delta.days




def extended_imputation(data_frame:pd.DataFrame) -> pd.DataFrame:
    """
    This function takes in a dataframe and a key, and returns a dataframe with the original dataframe and the imputed dataframe concatenated.
    """
    # Identify columns with missing data
    cols_with_missing_data = [col for col in data_frame.columns if data_frame[col].isnull().any()]
    print(f"cols with missing data : {cols_with_missing_data}")
    # Impute missing values with the mean for numerical columns and the most frequent value for categorical columns
    
    for col in cols_with_missing_data:
        if data_frame[col].dtype == "object":
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            imputer = SimpleImputer(strategy='mean')
        try:
            data_frame[col] = imputer.fit_transform(data_frame[[col]])
        except Exception as e:
            data_frame.drop(col,axis=1)
            print(f"droped : {col}")
            print(e)
    return data_frame

def one_hot_encode_column(data_frame, key) -> pd.DataFrame:
    """
    This function takes in a dataframe and a key (column name), and returns a dataframe with the original dataframe and the one-hot encoded dataframe concatenated.
    The key column is expected to contain comma-separated strings or NaNs.
    """
    # Handle NaN values by replacing them with an empty string before splitting
    series_clean = data_frame[key].fillna('').apply(lambda x: x.split(',') if x != '' else [])
    
    mlb = MultiLabelBinarizer()
    one_hot_encoded_data = mlb.fit_transform(series_clean.tolist())
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data, columns=mlb.classes_)
    num_columns_one_hot_encoded = one_hot_encoded_df.shape[1]
    print(f"Number of columns in the one-hot encoded data for {key}: {num_columns_one_hot_encoded}")

    final_df = pd.concat([data_frame, one_hot_encoded_df], axis=1)
    
    return final_df



def ordinal_encode_column(data_frame, key) -> pd.DataFrame:
    """
    This function takes in a dataframe and a key, and returns a dataframe with the original dataframe and the ordinal encoded dataframe concatenated.
    The column specified by 'key' is expected to contain comma-separated strings or NaNs.
    """
    cleaned_data = data_frame[key].fillna('').apply(lambda x: ','.join(sorted(x.split(','))) if x != '' else '')
    oe = OrdinalEncoder()
    ordinal_encoded_data = oe.fit_transform(cleaned_data.values.reshape(-1, 1))
    ordinal_encoded_df = pd.DataFrame(ordinal_encoded_data, columns=[key + '_encoded'])
    final_df = pd.concat([data_frame, ordinal_encoded_df], axis=1)
    
    return final_df

def data_standardizer(data_frame:pd.DataFrame):
    standardScaler = StandardScaler()
    data_frame = standardScaler.fit_transform(data_frame)
    data = pd.DataFrame(data_frame)
    return data

def revenue_log(y):
    res = np.log(y)
    if res == float('-inf'):
        return 0.1
    else:
        return res
    
def devide_by_1_000_000(y):
    return y/1000000
def revenue_exp(y):
    return np.exp(y)


def test_data_set(raw_data:pd.DataFrame):

    raw_data = raw_data[raw_data['revenue'] != 0]
    raw_data['revenue'] = raw_data['revenue'].apply(revenue_log)

    # Preparing the data
    X = raw_data.drop(columns=['revenue'])
    y = raw_data['revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
    estimators = list(range(1, 201))  
    rmses = []


    for n in estimators:
        model = RandomForestRegressor(n_estimators=n, random_state=42, max_depth=10, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmses.append(rmse)
        print(f"RMSE for {n} estimators: {rmse}")

  
    plt.figure(figsize=(10, 6))
    plt.plot(estimators, rmses, marker='o', linestyle='-', markersize=5)
    plt.title('RMSE vs. Number of Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.show()
    return plt