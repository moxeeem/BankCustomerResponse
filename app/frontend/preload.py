import requests
import pandas as pd
import numpy as np

BACKEND_PATH = "https://bankcustomers-backend.onrender.com/"


def get_test() -> tuple:
    """
    Retrieve the Xtest and ytest data from the backend API.

    Returns:
        tuple: A tuple containing the Xtest and ytest data.
    """
    response_Xtest = requests.get(
        f"{BACKEND_PATH}xtest/all?limit=1000000"
    )
    Xtest = response_Xtest.json()
    Xtest = pd.DataFrame(Xtest)
    Xtest.pop('id')

    response_ytest = requests.get(
        f"{BACKEND_PATH}ytest/all?limit=1000000"
    )
    ytest = response_ytest.json()
    ytest = pd.DataFrame(ytest)
    ytest.pop('id')

    return Xtest, ytest


def get_logreg_preds_probs() -> tuple:
    """
    Retrieve logistic regression without tunning predictions and probabilities
    from the backend API.

    Returns:
        tuple: A tuple containing the logistic regression predictions
        and probabilities.
    """
    response_logreg_probs = requests.get(
        f"{BACKEND_PATH}logreg_probs"
    )
    logreg_probs = response_logreg_probs.json()
    logreg_probs = np.array(logreg_probs["predict_proba"])

    response_logreg_preds = requests.get(
        f"{BACKEND_PATH}logreg_preds"
    )
    logreg_preds = response_logreg_preds.json()
    logreg_preds = np.array(logreg_preds["predict"])

    return logreg_preds, logreg_probs


def get_logreg_tuned_preds_probs() -> tuple:
    """
    Retrieve logistic regression with tunning predictions and probabilities
    from the backend API.

    Returns:
        tuple: A tuple containing the logistic regression predictions
        and probabilities.
    """
    response_logreg_tuned = requests.get(
        f"{BACKEND_PATH}logreg_tuned_probs"
    )
    logreg_tuned_probs = response_logreg_tuned.json()
    logreg_tuned_probs = np.array(logreg_tuned_probs["predict_proba"])

    response_logreg_tuned_preds = requests.get(
        f"{BACKEND_PATH}logreg_tuned_preds"
    )
    logreg_tuned_preds = response_logreg_tuned_preds.json()
    logreg_tuned_preds = np.array(logreg_tuned_preds["predict"])

    return logreg_tuned_preds, logreg_tuned_probs


def get_svc_preds_probs() -> tuple:
    """
    Retrieve SVC without tunning predictions and probabilities
    from the backend API.

    Returns:
        tuple: A tuple containing the SVC predictions
        and probabilities.
    """
    response_svc_probs = requests.get(
        f"{BACKEND_PATH}svc_probs"
    )
    svc_probs = response_svc_probs.json()
    svc_probs = np.array(svc_probs["predict_proba"])

    response_svc_preds = requests.get(
        f"{BACKEND_PATH}svc_preds"
    )
    svc_preds = response_svc_preds.json()
    svc_preds = np.array(svc_preds["predict"])

    return svc_preds, svc_probs


def get_svc_tuned_preds_probs() -> tuple:
    """
    Retrieve SVC with tunning predictions and probabilities
    from the backend API.

    Returns:
        tuple: A tuple containing the SVC predictions
        and probabilities.
    """
    response_svc_tuned = requests.get(
        f"{BACKEND_PATH}svc_tuned_probs"
    )
    svc_tuned_probs = response_svc_tuned.json()
    svc_tuned_probs = np.array(svc_tuned_probs["predict_proba"])

    response_svc_tuned_preds = requests.get(
        f"{BACKEND_PATH}svc_tuned_preds"
    )
    svc_tuned_preds = response_svc_tuned_preds.json()
    svc_tuned_preds = np.array(svc_tuned_preds["predict"])

    return svc_tuned_preds, svc_tuned_probs


def get_dirty_data() -> pd.DataFrame:
    """
    Retrieve data without preprocessing from the backend API.

    Returns:
        pd.DataFrame: The dirty data stored in a pandas DataFrame.
    """
    response_df_dirty = requests.get(
        f"{BACKEND_PATH}merged/all?limit=1000000"
    )
    df_dirty = response_df_dirty.json()
    df_dirty = pd.DataFrame(df_dirty)
    df_dirty.pop('id')

    return df_dirty


def get_data_without_duplicates() -> pd.DataFrame:
    """
    Retrieve data without duplicates from the backend API.

    Returns:
        pd.DataFrame: Data without duplicates stored in a pandas DataFrame.
    """
    response_df_without_duplicates = requests.get(
        f"{BACKEND_PATH}without_duplicates/all?limit=1000000"
    )
    df_without_duplicates = response_df_without_duplicates.json()
    df_without_duplicates = pd.DataFrame(df_without_duplicates)
    df_without_duplicates.pop('id')

    return df_without_duplicates


def get_data_preprocessed() -> pd.DataFrame:
    """
    Retrieve data after preprocessing from the backend API.

    Returns:
        pd.DataFrame: Data after preprocessing stored in a pandas DataFrame.
    """
    response_df_preprocessed = requests.get(
        f"{BACKEND_PATH}filled_nans/all?limit=1000000"
    )
    df_preprocessed = response_df_preprocessed.json()
    df_preprocessed = pd.DataFrame(df_preprocessed)
    df_preprocessed.pop('id')

    return df_preprocessed
