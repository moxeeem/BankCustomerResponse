import pandas as pd
import copy


def open_data(path="data/datasets"):
    D_clients = pd.read_csv(path + "/D_clients.csv")
    D_close_loan = pd.read_csv(path + "/D_close_loan.csv")
    D_job = pd.read_csv(path + "/D_job.csv")
    D_last_credit = pd.read_csv(path + "/D_last_credit.csv")
    D_loan = pd.read_csv(path + "/D_loan.csv")
    D_pens = pd.read_csv(path + "/D_pens.csv")
    D_salary = pd.read_csv(path + "/D_salary.csv")
    D_target = pd.read_csv(path + "/D_target.csv")
    D_work = pd.read_csv(path + "/D_work.csv")

    return D_clients, D_close_loan, D_job, D_last_credit, D_loan, D_pens, D_salary, D_target, D_work


def concat_data(D_clients, D_close_loan, D_job, D_last_credit, D_loan, D_pens, D_salary, D_target, D_work):
    D_clients = pd.merge(D_clients, D_target, left_on='ID', right_on='ID_CLIENT')
    D_clients = D_clients.drop('ID_CLIENT', axis=1)

    D_clients = pd.merge(D_clients, D_job, left_on='ID', right_on='ID_CLIENT')
    D_clients = D_clients.drop('ID_CLIENT', axis=1)

    D_clients = pd.merge(D_clients, D_salary, left_on='ID', right_on='ID_CLIENT')
    D_clients = D_clients.drop('ID_CLIENT', axis=1)

    D_clients = pd.merge(D_clients, D_last_credit, left_on='ID', right_on='ID_CLIENT')
    D_clients = D_clients.drop('ID_CLIENT', axis=1)

    merged_df = pd.merge(D_loan, D_close_loan, on='ID_LOAN', how='left')

    grouped_df = merged_df.groupby('ID_CLIENT').agg(LOAD_NUM_TOTAL=('ID_LOAN', 'count'),
                                                    LOAN_NUM_CLOSED=('CLOSED_FL', 'sum')).reset_index()
    result_df = grouped_df[['ID_CLIENT', 'LOAD_NUM_TOTAL', 'LOAN_NUM_CLOSED']]

    D_clients = pd.merge(D_clients, result_df, left_on='ID', right_on='ID_CLIENT')
    D_clients = D_clients.drop('ID_CLIENT', axis=1)

    y = D_clients["TARGET"]
    D_clients = D_clients.drop(columns=["TARGET"])
    D_clients = pd.concat([D_clients, y], axis=1)

    D_clients = D_clients.drop('ID', axis=1)

    y = D_clients["AGREEMENT_RK"]
    D_clients = D_clients.drop(columns=["AGREEMENT_RK"])
    D_clients = pd.concat([y, D_clients], axis=1)

    return D_clients


def drop_duplicates(df):
    df1 = copy.deepcopy(df)
    df1.drop_duplicates(inplace=True)
    df1.reset_index(drop=True, inplace=True)

    return df1


def fill_nans(df):
    df1 = copy.deepcopy(df)
    df1['GEN_INDUSTRY'].fillna(df1['GEN_INDUSTRY'].mode()[0], inplace=True)
    df1['GEN_TITLE'].fillna(df1['GEN_TITLE'].mode()[0], inplace=True)
    df1['JOB_DIR'].fillna(df1['JOB_DIR'].mode()[0], inplace=True)

    median_WT = df1['WORK_TIME'].median()
    df1['WORK_TIME'].fillna(median_WT, inplace=True)

    return df1


if __name__ == "__main__":
    pass
