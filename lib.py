import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb



def transform_bool(col_data):
    """
    A function that transforms a Series input of values from {'t', 'f'}
    representing true and false, to values of 0 and 1.
    """
    return col_data.apply(lambda x: 0 if x == 'f' else 1)


def string_to_numeric_series(inputs, pattern = '[$,]'):
    """
    Transform a Series of string values into numberic and remove a specific
    regex pattern, e.g. "$xx.xx" becomes xx.xx

    Assumes the regex pattern is to remove the "$" by default
    """
    return inputs.str.replace('[$,]', '').astype('float')


def create_dummies_from_list_col(inputs, col, suffix, other_min = 1000):
    """
    Cleans a column that is made up of lists, creates dummy vars for it
    """
    mlb = MultiLabelBinarizer()
    
    dummies = pd \
        .DataFrame(mlb.fit_transform(inputs[col]), columns=mlb.classes_, index=inputs.index) \
        .add_suffix(suffix)

    others = dummies.sum() < 1000
    dummies_drop_cols = others[others == True].index
    
    dummies = dummies.drop(labels=dummies_drop_cols, axis='columns')

    return pd.concat([inputs.drop(col, axis=1), dummies], axis=1)


def create_dummies(inputs, col, dummy_na):
    """
    Creates dummy vars for a specified column in a dataframe
    """
    dummies_df = pd.get_dummies(inputs[col], prefix=col, prefix_sep='_', drop_first=True, dummy_na=dummy_na)
    
    return pd.concat([inputs.drop(col, axis=1), dummies_df], axis=1)


def create_other_category(inputs, col, n_top=5):
    """
    Given a categorial column, replace all non-common values with 'other'

    The threshold to determine which values are common is provided by n_top argument.
    """
    common_types = inputs[col].value_counts().head(n_top).index.values
    inputs.loc[~inputs[col].isin(common_types), col] = 'other'
    
    return inputs


def clean(inputs, include_dummies=True, dummy_na=False, cat_cols=[]):
    """
    A function to clean the AirBnB listings dataframe
    """
    df = inputs.copy()
    
    df.price = string_to_numeric_series(df.price)
    
    # transform cleaning_fee into a numberic value
    if 'cleaning_fee' in df.columns:
        df.cleaning_fee = string_to_numeric_series(df.cleaning_fee)
    
    # transform "host_response_rate" into a numeric column, also fill missing values with mean
    if 'host_response_rate' in df.columns:
        df.host_response_rate = string_to_number_series(df.host_response_rate, pattern='%')
        df.host_response_rate = df.host_response_rate.fillna(df.host_response_rate.mean())
    
    # Fill missing bedroom values
    if 'bedrooms' in df.columns:
        df.bedrooms = df.bedrooms.fillna(0) # assume studio / bachelor

    if 'beds' in df.columns:
        df.bedrooms = df.bedrooms.fillna(1) # assume at least one bed
    
    # transform the "amenities" into a column of arrays, then create dummy vals (0, 1) with many columns
    if 'amenities' in df.columns:
        df.amenities = df.amenities\
            .apply(lambda x: set(x.replace('{', '').replace('}', '').replace('"', '').split(',')))
    
    if include_dummies:
        # NOTE: this adds 190+ columns, some of these are amenities that are very specific and may not be 
        #       worth considering, circle back and check if that is true
        df = create_dummies_from_list_col(df, col='amenities', suffix='_amenities')

        # replace least common property types with "other"
        df = create_other_category(df, col='property_type', n_top=7)
        df = create_other_category(df, col='room_type', n_top=7)
        df = create_other_category(df, col='bed_type', n_top=7)

        # transform bool columns such that 'f' = 0 and 't' = 1
        # df.host_is_superhost = transform_bool(df.host_is_superhost).fillna(0)
        df.instant_bookable = transform_bool(df.instant_bookable).fillna(0)
        # df.require_guest_phone_verification = transform_bool(df.require_guest_phone_verification).fillna(0)

        # create dummies
        for col in  cat_cols:
            df = create_dummies(df, col=col, dummy_na=dummy_na)
    
    # drop rows with price outliers
    df = df[df.price <= (np.std(df.price) * 3)]
    
    # fill any remaining missing values with the mean
    df = df.fillna(df.mean())
    
    return df


def train_test(df, response_col='price', dummy_na=False, test_size=.3, rand_state=42, model):
    """
    INPUT:
    df - dataframe containing the data
    response_col - the column to predict values for
    dummy_na - weather to use the dummy_na option when for dummy vars
    test_size - the size of the test set as a value from 0 to 1
    rand_state - the random state for the train-test split
    model - the model which to train with the data

    Provided a dataframe and model, this method will split the data
    into a train and test set.
    The model will then be fit and used to predict on the test set.
    Finally, the model is scored with the r2_score method.
    """
    # Split into explanatory and response variables
    X = df.drop(response_col, axis=1)
    y = df[response_col]
    
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size, random_state=rand_state)

    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_score = r2_score(y_train, train_preds)
    test_score = r2_score(y_test, test_preds)

    return test_score, train_score, X_train, X_test, y_train, y_test, test_preds


def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df


def load_data(path = './data/toronto-listings-2018-10.csv'):
    """
    INPUT:
    path - the path of the csv to load data from, *optional

    OUTPUT:
    0 - dataframe of the csv with only keep columns
    1 - the columns which are keps
    2 - categorical columns in the dataset

    A function that loads the listings CSV into a pandas DataFrame
    """
    data = pd.read_csv(path)
    data.set_index('id', drop=True, inplace=True)

    # Some of the columns with valuable data
    keep_cols = [
        "price", "neighbourhood_cleansed", "room_type", 
        "bedrooms", "bed_type", 
        "maximum_nights", "beds", 
        "property_type", "amenities", 
        "number_of_reviews", "minimum_nights",
        "cleaning_fee", "instant_bookable",
        "review_scores_rating", "reviews_per_month", 
        "require_guest_phone_verification", "host_is_superhost",
        "host_response_rate" #, "host_acceptance_rate", "weekly_price", "monthly_price",
    ]
    
    # Define the categorical columns which will need dummy vars before modeling
    cat_cols = ['property_type', 'bed_type', 'neighbourhood_cleansed', 'room_type']
    
    return data[keep_cols], keep_cols, cat_cols