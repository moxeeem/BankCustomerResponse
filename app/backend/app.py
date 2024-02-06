from fastapi import FastAPI, Depends
from typing import List, Any
from sqlalchemy.orm import Session
from .database import SessionLocal
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
from .models import Client, CloseLoan, Job, LastCredit, Loan, Pens, Salary, \
    Work, Target, Merged, WithoutDuplicates, FilledNans, Xtest, Xtrain, \
    Ytrain, Ytest
from .schemas import ClientGet, CloseLoanGet, JobGet, LastCreditGet, LoanGet, \
    PensGet, SalaryGet, WorkGet, TargetGet, MergedGet, XGet, YGet


app = FastAPI()


def get_session() -> Session:
    with SessionLocal() as session:
        return session


@app.get("/clients/all", response_model=List[ClientGet])
def get_all_clients(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Client)]:
    return db.query(Client).limit(limit).all()


@app.get("/closeloan/all", response_model=List[CloseLoanGet],
         summary="Get all Close Loan")
def get_all_closeloan(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(CloseLoan)]:
    return db.query(CloseLoan).limit(limit).all()


@app.get("/jobs/all", response_model=List[JobGet])
def get_all_jobs(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Job)]:
    return db.query(Job).limit(limit).all()


@app.get("/lastcredit/all", response_model=List[LastCreditGet],
         summary="Get all Last Credit")
def get_all_lastcredit(limit: int = 10, db: Session = Depends(get_session)) \
        -> list[type(LastCredit)]:
    return db.query(LastCredit).limit(limit).all()


@app.get("/loan/all", response_model=List[LoanGet])
def get_all_loan(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Loan)]:
    return db.query(Loan).limit(limit).all()


@app.get("/pens/all", response_model=List[PensGet])
def get_all_pens(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Pens)]:
    return db.query(Pens).limit(limit).all()


@app.get("/salary/all", response_model=List[SalaryGet])
def get_all_salary(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Salary)]:
    return db.query(Salary).limit(limit).all()


@app.get("/work/all", response_model=List[WorkGet])
def get_all_work(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Work)]:
    return db.query(Work).limit(limit).all()


@app.get("/target/all", response_model=List[TargetGet])
def get_all_target(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Target)]:
    return db.query(Target).limit(limit).all()


@app.get("/merged/all", response_model=List[MergedGet],
         summary="Get merged dataset")
def get_all_merged(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Merged)]:
    return db.query(Merged).limit(limit).all()


@app.get("/without_duplicates/all", response_model=List[MergedGet],
         summary="Get merged dataset without duplicates")
def get_all_without_duplicates(limit: int = 10,
                               db: Session = Depends(get_session)) -> \
        list[type(WithoutDuplicates)]:
    return db.query(WithoutDuplicates).limit(limit).all()


@app.get("/filled_nans/all", response_model=List[MergedGet],
         summary="Get merged dataset without duplicates with filled nans")
def get_all_filled_nans(limit: int = 10,
                        db: Session = Depends(get_session)) -> \
        list[type(FilledNans)]:
    return db.query(FilledNans).limit(limit).all()


@app.get("/xtest/all", response_model=List[XGet],
         summary="Get Xtest")
def get_all_xtest(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Xtest)]:
    return db.query(Xtest).limit(limit).all()


@app.get("/ytest/all", response_model=List[YGet],
         summary="Get ytest")
def get_all_ytest(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Ytest)]:
    return db.query(Ytest).limit(limit).all()


@app.get("/xtrain/all", response_model=List[XGet],
         summary="Get Xtrain")
def get_all_xtrain(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Xtrain)]:
    return db.query(Xtrain).limit(limit).all()


@app.get("/ytrain/all", response_model=List[YGet],
         summary="Get ytrain")
def get_all_ytrain(limit: int = 10, db: Session = Depends(get_session)) -> \
        list[type(Ytrain)]:
    return db.query(Ytrain).limit(limit).all()

#############################################################################


@app.get("/logreg_metrics",
         summary="Metrics of Logistic Regression without tunning")
def get_logreg_metrics(db: Session = Depends(get_session)) -> \
        dict[str, float | Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()
    y_test = db.query(Ytest).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]
    y_test_data = [y.target for y in y_test]

    model = LogisticRegression(random_state=42)
    model.fit(x_train_data, y_train_data)
    y_pred = model.predict(x_test_data)

    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred)
    recall = recall_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred)
    roc_auc = roc_auc_score(y_test_data, y_pred)

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc}


@app.get("/logreg_probs", summary="Predict probabilities using "
                                  "Logistic Regression without tunning")
def get_logreg_probs(db: Session = Depends(get_session)) -> \
        dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]

    model = LogisticRegression(random_state=42)
    model.fit(x_train_data, y_train_data)
    y_proba = model.predict_proba(x_test_data)

    return {"predict_proba": y_proba.tolist()}


@app.get("/logreg_preds",
         summary="Predicts using Logistic Regression without tunning")
def get_logreg_preds(db: Session = Depends(get_session)) -> dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]

    model = LogisticRegression(random_state=42)
    model.fit(x_train_data, y_train_data)
    y_pred = model.predict(x_test_data)

    return {"predict": y_pred.tolist()}

#############################################################################


@app.get("/logreg_tuned_metrics",
         summary="Metrics of Logistic Regression with tunning")
def get_logreg_tuned_metrics(db: Session = Depends(get_session)) -> \
        dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()
    y_test = db.query(Ytest).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]
    y_test_data = [y.target for y in y_test]

    model = LogisticRegression(C=0.1, random_state=42)
    model.fit(x_train_data, y_train_data)
    y_pred = model.predict(x_test_data)

    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred)
    recall = recall_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred)
    roc_auc = roc_auc_score(y_test_data, y_pred)

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc}


@app.get("/logreg_tuned_probs", summary="Predict probabilities using Logistic "
                                        "Regression with tunning")
def get_logreg_tuned_probs(db: Session = Depends(get_session)) \
        -> dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]

    model = LogisticRegression(C=0.1, random_state=42)
    model.fit(x_train_data, y_train_data)
    y_proba = model.predict_proba(x_test_data)

    return {"predict_proba": y_proba.tolist()}


@app.get("/logreg_tuned_preds",
         summary="Predicts using Logistic Regression with tunning")
def get_logreg_tuned_preds(db: Session = Depends(get_session)) -> dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]

    model = LogisticRegression(C=0.1, random_state=42)
    model.fit(x_train_data, y_train_data)
    y_pred = model.predict(x_test_data)

    return {"predict": y_pred.tolist()}

#############################################################################


@app.get("/svc_metrics",
         summary="Metrics of SVC without tunning")
def get_svc_metrics(db: Session = Depends(get_session)) -> dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()
    y_test = db.query(Ytest).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]
    y_test_data = [y.target for y in y_test]

    model = svm.SVC(C=0.1, probability=True, random_state=42)
    model.fit(x_train_data, y_train_data)
    y_pred = model.predict(x_test_data)

    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred)
    recall = recall_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred)
    roc_auc = roc_auc_score(y_test_data, y_pred)

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc}


@app.get("/svc_probs",
         summary="Predict probabilities using SVC without tunning")
def get_svc_proba(db: Session = Depends(get_session)) -> dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]

    model = svm.SVC(C=0.1, probability=True, random_state=42)
    model.fit(x_train_data, y_train_data)
    y_proba = model.predict_proba(x_test_data)

    return {"predict_proba": y_proba.tolist()}


@app.get("/svc_preds",
         summary="Predicts using SVC without tunning")
def get_svc_preds(db: Session = Depends(get_session)) -> dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]

    model = svm.SVC(C=0.1, probability=True, random_state=42)
    model.fit(x_train_data, y_train_data)
    y_pred = model.predict(x_test_data)

    return {"predict": y_pred.tolist()}

#############################################################################


@app.get("/svc_tuned_metrics",
         summary="Metrics of SVC with tunning")
def get_svc_tuned_metrics(db: Session = Depends(get_session)) \
        -> dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()
    y_test = db.query(Ytest).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]
    y_test_data = [y.target for y in y_test]

    model = svm.SVC(C=0.1, gamma=0.1, kernel='linear', probability=True,
                    random_state=42)
    model.fit(x_train_data, y_train_data)
    y_pred = model.predict(x_test_data)

    accuracy = accuracy_score(y_test_data, y_pred)
    precision = precision_score(y_test_data, y_pred)
    recall = recall_score(y_test_data, y_pred)
    f1 = f1_score(y_test_data, y_pred)
    roc_auc = roc_auc_score(y_test_data, y_pred)

    return {"accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc}


@app.get("/svc_tuned_probs",
         summary="Predict probabilities using SVC with tunning")
def get_svc_tuned_probs(db: Session = Depends(get_session)) -> dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]

    # Загружаем обученную модель
    model = svm.SVC(C=0.1, gamma=0.1, kernel='linear', probability=True,
                    random_state=42)
    model.fit(x_train_data, y_train_data)
    y_proba = model.predict_proba(x_test_data)

    return {"predict_proba": y_proba.tolist()}


@app.get("/svc_tuned_preds",
         summary="Predicts using SVC with tunning")
def get_svc_tuned_preds(db: Session = Depends(get_session)) -> dict[str, Any]:
    x_train = db.query(Xtrain).all()
    x_test = db.query(Xtest).all()
    y_train = db.query(Ytrain).all()

    x_train_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_train]

    x_test_data = [
        [x.agreement_rk,
         x.age,
         x.gender,
         x.child_total,
         x.dependants,
         x.socstatus_work_fl,
         x.socstatus_pens_fl,
         x.own_auto,
         x.fl_presence_fl,
         x.personal_income,
         x.credit,
         x.load_num_total,
         x.loan_num_closed,
         x.edu_2plus,
         x.edu_high_unfinished,
         x.edu_mid_unfinished,
         x.edu_mid,
         x.edu_mid_spec,
         x.edu_scholar,
         x.ms_civil,
         x.ms_unmarried,
         x.ms_divorced,
         x.ms_married,
         x.fi_10k_20k,
         x.fi_20k_50k,
         x.fi_5k_10k,
         x.fi_50k_plus]
        for x in x_test]

    y_train_data = [y.target for y in y_train]

    # Загружаем обученную модель
    model = svm.SVC(C=0.1, gamma=0.1, kernel='linear', probability=True,
                    random_state=42)
    model.fit(x_train_data, y_train_data)
    y_pred = model.predict(x_test_data)

    return {"predict": y_pred.tolist()}

#############################################################################


