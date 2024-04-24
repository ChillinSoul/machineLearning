import json
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def extract_iso_values(
        json_str: str,
        key: str
        ) -> list:
    """
    Extracts values from a JSON string that contains a list of dictionaries,
    all of which are expected to contain the specified key.

    Parameters:
    json_str (str): The JSON string to parse.
    key (str): The key whose values are to be extracted.

    Returns:
    list: A list of values associated with the given key, or ["error"] if an error occurs.
    """
    try:
        data_list = json.loads(json_str)
        iso_values = [item[key] for item in data_list if key in item]
        return iso_values
    except json.JSONDecodeError:
        return ["error: invalid JSON format"]
    except KeyError:
        return ["error: key not found"]
    except Exception as e:
        return [f"error: {str(e)}"]


def extract_iso_values_to_string(
        json_str: str, 
        key: str
        ) -> str:
    """
    Converts extracted values from a JSON string into a single comma-separated string.

    Parameters:
    json_str (str): The JSON string to parse.
    key (str): The key whose values are to be concatenated.

    Returns:
    str: A comma-separated string of values for the given key, or "error" if an error occurs.
    """
    try:
        data_list = json.loads(json_str)
        iso_values = ','.join(str(item[key]) for item in data_list if key in item)
        return iso_values
    except json.JSONDecodeError:
        return "error: invalid JSON format"
    except KeyError:
        return "error: key not found"
    except Exception as e:
        return f"error: {str(e)}"


def extract_first_iso_value(
        json_str: str, 
        key: str
        ) -> str:
    """
    Extracts the first value associated with the specified key from a JSON string.

    Parameters:
    json_str (str): The JSON string to parse.
    key (str): The key whose first value is to be extracted.

    Returns:
    str: The first value found for the given key, or None if an error occurs.
    """
    try:
        values = extract_iso_values(json_str, key)
        return values[0] if values else None
    except Exception as e:
        return None


def convert_datestring_to_days_since_1900(
        date: str
        ) -> int:
    """
    Calculates the number of days from January 1, 1900, to a given date.

    Parameters:
    date (str): The date string in the format 'YYYY-MM-DD'.

    Returns:
    int: The number of days from January 1, 1900, to the given date.
         Returns 0 if the date format is incorrect or the date is before January 1, 1900.

    Examples:
    >>> convert_datestring_to_days_since_1900('2021-03-15')
    44299
    """
    try:
        given_date = datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        return 0

    start_date = datetime(1900, 1, 1)

    delta = given_date - start_date

    if delta.days < 0:
        return 0

    return delta.days


def extended_imputation(
        data_frame: pd.DataFrame, 
        verbose: bool = False
        ) -> pd.DataFrame:
    """
    Imputes missing values in a DataFrame using different strategies based on the column data type.
    For categorical data (object type), it uses the most frequent value for imputation.
    For numerical data, it uses the mean value for imputation.

    Parameters:
    data_frame (pd.DataFrame): The DataFrame to process for missing values.
    verbose (bool): If True, prints the columns being processed and any errors encountered.

    Returns:
    pd.DataFrame: The DataFrame with imputed values.

    Notes:
    If imputation fails for a column, that column is dropped from the DataFrame.
    """
    cols_with_missing_data = [col for col in data_frame.columns if data_frame[col].isnull().any()]
    if verbose:
        print(f"Columns with missing data: {cols_with_missing_data}")

    for col in cols_with_missing_data:
        try:
            if data_frame[col].dtype == "object":
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                imputer = SimpleImputer(strategy='mean')
            
            data_frame.loc[col] = imputer.fit_transform(data_frame.loc[[col]])
        except Exception as e:
            data_frame = data_frame.drop(columns=[col])
            if verbose:
                print(f"Dropped column '{col}' due to an error during imputation.")
                print(e)

    return data_frame


def one_hot_encode_column(
        data_frame: pd.DataFrame, 
        key: str, 
        verbose: bool = False
        ) -> pd.DataFrame:
    """
    One-hot encodes a specified column in a pandas DataFrame. This function is particularly useful
    for columns storing lists of categories as comma-separated strings. Each category becomes a 
    binary column in the resulting DataFrame.

    Parameters:
    data_frame (pd.DataFrame): The DataFrame containing the data.
    key (str): The column name to be one-hot encoded.
    verbose (bool): If True, prints the number of columns generated from one-hot encoding.

    Returns:
    pd.DataFrame: A DataFrame with the original data and new one-hot encoded columns.

    Raises:
    KeyError: If the specified key is not in the DataFrame's columns.
    """
    
    if key not in data_frame.columns:
        raise KeyError(f"Column '{key}' not found in the DataFrame.")

    series_clean = data_frame[key].fillna('').apply(lambda x: x.split(',') if x else [])

    mlb = MultiLabelBinarizer()
    one_hot_encoded_data = mlb.fit_transform(series_clean.tolist())
    one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data, columns=mlb.classes_)

    if verbose:
        num_columns_one_hot_encoded = one_hot_encoded_df.shape[1]
        print(f"Number of columns in the one-hot encoded data for '{key}': {num_columns_one_hot_encoded}")

    final_df = pd.concat([data_frame, one_hot_encoded_df], axis=1)

    return final_df


def ordinal_encode_column(
        data_frame: pd.DataFrame, 
        key
        ) -> pd.DataFrame:
    """
    Ordinal encodes a specified column in a pandas DataFrame. This function
    handles columns containing comma-separated strings by first sorting the strings,
    then applying the ordinal encoding. Missing values are filled with an empty string
    before encoding.

    Parameters:
    data_frame (pd.DataFrame): The DataFrame containing the data.
    key (str): The column name to be ordinal encoded.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column added for the
                  ordinal encoded values, named '{key}_encoded'.

    Raises:
    ValueError: If the specified key is not in the DataFrame's columns.
    """
    if key not in data_frame.columns:
        raise ValueError(f"Column '{key}' does not exist in the DataFrame.")

    cleaned_data = data_frame[key].fillna('').apply(
        lambda x: ','.join(sorted(x.split(','))) if x != '' else ''
    )

    oe = OrdinalEncoder()

    ordinal_encoded_data = oe.fit_transform(cleaned_data.values.reshape(-1, 1))
    
    ordinal_encoded_df = pd.DataFrame(ordinal_encoded_data, columns=[key + '_encoded'])
    
    final_df = pd.concat([data_frame, ordinal_encoded_df], axis=1)
    
    return final_df


def data_standardizer(
        data_frame: pd.DataFrame, 
        *,
        n_components: int = None
        )->pd.DataFrame:
    """
    Standardizes the data and optionally applies PCA to reduce dimensionality.

    Args:
    data_frame (pd.DataFrame): The data frame to be standardized and transformed.
    n_components (int, optional): The number of principal components to keep. If None, PCA is not applied.

    Returns:
    pd.DataFrame: The standardized and optionally PCA-transformed data.
    """
    standardScaler = StandardScaler()
    standardized_data = standardScaler.fit_transform(data_frame)

    if n_components is not None:
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(standardized_data)
        column_names = [f'Principal Component {i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(data=pca_data, columns=column_names)
        return pca_df
    
    return pd.DataFrame(standardized_data, columns=data_frame.columns)


def revenue_log(
        y: float
        )->float:
    res = np.log(y)
    if res == float('-inf'): return 0.1
    else: return res


def transparent(
        y: float
        )->float:
    return y


def devide_by_1_000_000(
        y: float
        )->float:
    return y/1000000


def revenue_exp(
        y:  float
        )-> float: 
    return np.exp(y)


def revenue_sqrt(
        y:  float
        )->float:
    return np.sqrt(y)


def revenue_transform(
        y: float,
        *,
        transformation: str ="revenue_log"
        )->float:
    """
    Transforms revenue data according to the specified method.

    :Parameters:
    y (Series, array-like, or scalar): The revenue data to transform.
    transformation (str): Specifies the type of transformation to apply. Options include:
        - "revenue_log": Logarithmic transformation (default)
        - "revenue_exp": Exponential transformation
        - "revenue_sqrt": Square root transformation
        - "transparent": No transformation
        - "devide_by_1_000_000": Division by one million

    :Returns:
    Transformed data, depending on the chosen transformation method.
    """
    match transformation:
        case "revenue_log":
            return revenue_log(y)
        case "revenue_exp":
            return revenue_exp(y)
        case "revenue_sqrt":
            return revenue_sqrt(y)
        case "transparent":
            return transparent(y)
        case "devide_by_1_000_000":
            return devide_by_1_000_000(y)
        case _:
            return revenue_log(y)


def test_data_set(
        raw_data: pd.DataFrame, 
        graph: bool = False, 
        pca_components: int = None,
        revenue_transform = revenue_log, 
        verbose: bool = False,
        n_estimators: int = 30
        )->float:
    """
    Evaluates the performance of a RandomForestRegressor on transformed data.

    Parameters:
    raw_data (pd.DataFrame): The DataFrame containing features and a 'revenue' target column.
    graph (bool): If True, plots the actual values vs predictions.
    pca_components (int): The number of principal components to reduce to. If None, PCA is not applied.
    revenue_transform (function): A function to apply to the 'revenue' column before modeling.
    verbose (bool): If True, prints additional model performance information.

    Returns:
    float: The root mean squared error of the model's predictions.
    """

    raw_data = raw_data[raw_data['revenue'] != 0]
    raw_data['revenue'] = raw_data['revenue']#.apply(revenue_transform)

    X = raw_data.drop(columns=['revenue'])
    X = data_standardizer(X, n_components=pca_components)
    y = raw_data['revenue']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, max_depth=10, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    #rmse = np.sqrt(mean_squared_error(revenue_exp(y_test), revenue_exp(predictions)))
    rmse= np.sqrt(mean_squared_error(y_test, predictions))
    if verbose:
        print(f"RMSE for {n_estimators} estimators: {rmse}")
        #ratio = revenue_exp(y_test) / revenue_exp(predictions)
        ratio = y_test / predictions
        print(f"ratio of y_test to predictions:")
        print(ratio.describe())


    if graph:
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([plt.xlim()[0], plt.xlim()[1]])
        plt.ylim([plt.ylim()[0], plt.ylim()[1]])
        plt.plot([-100, 100], [-100, 100], 'r')  # Diagonal line
        plt.show()

    return rmse


def plot_mse_pca(
        raw_data: pd.DataFrame, 
        revenue_transform = revenue_log
        )->None:
    """
    Plots the Root Mean Squared Error (RMSE) against the number of PCA components used in the model.

    Parameters:
    raw_data (pd.DataFrame): The DataFrame containing the data to be analyzed.
    revenue_transform (function): A function to transform the 'revenue' column before modeling.

    Returns:
    None: Displays a plot of RMSE versus the number of PCA components.
    """
    mse_values = []
    best_mse = float('inf')  
    best_pca = 0

    for i in range(1, raw_data.shape[1]):
        mse = test_data_set(raw_data, pca_components=i, revenue_transform=revenue_transform)
        mse_values.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_pca = i

    print(f"Best RMSE: {best_mse:.2f} for {best_pca} PCA components")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, raw_data.shape[1]), mse_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('RMSE')
    plt.title('RMSE vs. Number of PCA Components')
    plt.grid(True)
    plt.show()


def plot_pca_2d(
        data_frame: pd.DataFrame, 
        target_column: str
        )->None:
    """
    Generates a 2D PCA scatter plot from the given DataFrame, reducing feature dimensions to two principal components,
    and plots these components with points color-coded by the specified target column.

    Parameters:
    data_frame (pd.DataFrame): The DataFrame containing the dataset with features and target.
    target_column (str): The column in the DataFrame to use for color coding in the scatter plot.

    Raises:
    ValueError: If the specified `target_column` does not exist in the DataFrame.

    Returns:
    None: Displays a 2D scatter plot of the two principal components.
    """

    if target_column not in data_frame.columns:
        raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")
    
    features = data_frame.drop(columns=[target_column])
    targets = data_frame[target_column]

    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(standardized_features)
    pca_df = pd.DataFrame(data=pca_data, columns=['Principal Component 1', 'Principal Component 2'])

    pca_df[target_column] = targets
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue=target_column, data=pca_df, palette='viridis')
    plt.title('2D PCA of ' + target_column)
    plt.show()


def plot_pca_3d(
        data_frame:pd.DataFrame, 
        target_column: str
        )->None:
    """
    Generates a 3D scatter plot for the first three principal components of the given data.

    This function standardizes the features of the provided DataFrame, applies PCA to reduce
    the dimensionality to three components, and plots these components in a 3D scatter plot
    color-coded by the target column.

    Parameters:
    data_frame (pd.DataFrame): The DataFrame containing the data to analyze.
    target_column (str): The column name in `data_frame` used for color coding the plot points.

    Raises:
    ValueError: If the specified `target_column` is not in the DataFrame.

    Returns:
    None: The function directly displays the 3D scatter plot.
    """
    
    if target_column not in data_frame.columns:
        raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")
    
    features = data_frame.drop(columns=[target_column])
    targets = data_frame[target_column].apply(revenue_log)
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(standardized_features)
    pca_df = pd.DataFrame(data=pca_data, columns=['Principal Component 1', 'Principal Component 2', 'Principal Component 3'])

    pca_df[target_column] = targets

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], pca_df['Principal Component 3'],
                         c=targets, cmap='viridis', edgecolor='k', s=40, alpha=0.5)
    ax.set_title('3D PCA')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    legend = ax.legend(*scatter.legend_elements(), title=target_column)
    ax.add_artist(legend)
    plt.show()


def plot_pca_pairplot(
        data_frame:pd.DataFrame, 
        target_column: str
        )->None:
    """
    Generates a pair plot for the first three principal components of the given data.

    This function takes a DataFrame and a specified column for color coding (target variable),
    standardizes the features, applies PCA to reduce dimensions to three principal components,
    and creates a pair plot of these components with regression lines.

    Parameters:
    data_frame (pd.DataFrame): The DataFrame containing the data.
    target_column (str): The name of the column in `data_frame` to use as the target variable for color coding in the plot.

    Raises:
    ValueError: If the `target_column` is not in the DataFrame.

    Returns:
    None: The function directly displays the pair plot.
    """
    
    if target_column not in data_frame.columns:
        raise ValueError(f"Column '{target_column}' does not exist in the DataFrame.")
    

    features = data_frame.drop(columns=[target_column])
    targets = data_frame[target_column].apply(revenue_log)


    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(standardized_features)
    pca_df = pd.DataFrame(data=pca_data, columns=['Principal Component 1', 
                                                  'Principal Component 2', 
                                                  'Principal Component 3'])

    pca_df[target_column] = targets


    pair_plot = sns.pairplot(pca_df, kind='reg', plot_kws={'scatter_kws': {'alpha': 0.5}})
    pair_plot.figure.suptitle('PCA Pair Plot of ' + target_column, y=1.02)  
    plt.show()

