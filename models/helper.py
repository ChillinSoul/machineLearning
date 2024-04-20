import pandas as pd
from scipy import stats
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def data_standardizer(data_frame: pd.DataFrame, n_components: int = None):
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
        # Applying PCA
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(standardized_data)
        # Convert the PCA output back to DataFrame
        column_names = [f'Principal Component {i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(data=pca_data, columns=column_names)
        return pca_df
    
    return pd.DataFrame(standardized_data, columns=data_frame.columns)

def output_standardizer(y)->pd.Series:
    """
    Standardizes the output data.

    Args:
    y (pd.Series): The output data to be standardized.

    Returns:
    pd.Series: The standardized output data.
    """
    mean = y.mean()
    std = y.std()

    return (y - mean) / std

def remove_outliers_z_score(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from a DataFrame based on the Z-score method. This function calculates the Z-scores
    of all numerical columns in the DataFrame and filters out rows where any column's Z-score is greater
    than 3 or less than -3.

    Parameters:
    data_frame (pd.DataFrame): The DataFrame from which to remove outliers.

    Returns:
    pd.DataFrame: A DataFrame with outliers removed based on Z-scores.
    """
    z_scores = np.abs(stats.zscore(data_frame.select_dtypes(include=[np.number])))
    filtered_entries = (z_scores < 3).all(axis=1)
    new_df = data_frame[filtered_entries]
    return new_df

def remove_outliers_dbscan(data_frame: pd.DataFrame, eps=0.5, min_samples=10) -> pd.DataFrame:
    """
    Removes outliers from a DataFrame using the DBSCAN clustering algorithm. Points in low-density regions
    are marked as outliers based on the specified 'eps' and 'min_samples' parameters.

    Parameters:
    data_frame (pd.DataFrame): The DataFrame from which to remove outliers.
    eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
    pd.DataFrame: A DataFrame with outliers removed based on DBSCAN clustering.
    """

    outlier_detection = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = outlier_detection.fit_predict(data_frame.select_dtypes(include=[np.number]))
    data_frame['cluster'] = clusters

    outliers = data_frame[data_frame['cluster'] == -1]

    cleaned_data = data_frame[data_frame['cluster'] != -1].drop('cluster', axis=1)
    return cleaned_data




def revenue_log(y):
    res = np.log(y)
    if res == float('-inf'): return 0.1
    else: return res


def transparent(y): return y


def devide_by_1_000_000(y): return y/1000000


def revenue_exp(y): return np.exp(y)


def revenue_sqrt(y): return np.sqrt(y)


def revenue_transform(y, transformation="revenue_log"):
    """
    Transforms revenue data according to the specified method.

    Parameters:
    y (Series, array-like, or scalar): The revenue data to transform.
    transformation (str): Specifies the type of transformation to apply. Options include:
        - "revenue_log": Logarithmic transformation (default)
        - "revenue_exp": Exponential transformation
        - "revenue_sqrt": Square root transformation
        - "transparent": No transformation
        - "devide_by_1_000_000": Division by one million

    Returns:
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
