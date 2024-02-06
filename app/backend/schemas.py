from pydantic import BaseModel
from typing import Optional


class ClientGet(BaseModel):
    id: int
    age: int
    gender: int
    education: str
    marital_status: str
    child_total: int
    dependants: int
    socstatus_work_fl: int
    socstatus_pens_fl: int
    reg_address_province: str
    fact_address_province: str
    postal_address_province: str
    fl_presence_fl: int
    own_auto: int


class JobGet(BaseModel):
    gen_industry: str
    gen_title: str
    job_dir: str
    work_time: float
    id_client: int


class LastCreditGet(BaseModel):
    credit: float
    term: int
    fst_payment: float
    id_client: int


class PensGet(BaseModel):
    id: int
    flag: int
    COMMENT: str


class SalaryGet(BaseModel):
    family_income: str
    personal_income: float
    id_client: int


class TargetGet(BaseModel):
    agreement_rk: int
    id_client: int
    target: int


class WorkGet(BaseModel):
    id: int
    flag: int
    COMMENT: str


class CloseLoanGet(BaseModel):
    id_loan: int
    closed_fl: int


class LoanGet(BaseModel):
    id_loan: int
    id_client: int


class MergedGet(BaseModel):
    agreement_rk: int
    age: int
    gender: int
    education: str
    marital_status: str
    child_total: int
    dependants: int
    socstatus_work_fl: int
    socstatus_pens_fl: int
    reg_address_province: str
    fact_address_province: str
    postal_address_province: str
    fl_presence_fl: int
    own_auto: int
    gen_industry: str
    gen_title: str
    job_dir: str
    work_time: Optional[float]
    family_income: str
    personal_income: float
    credit: float
    term: int
    fst_payment: float
    load_num_total: int
    loan_num_closed: int
    target: int
    id: int


class XGet(BaseModel):
    agreement_rk: float
    age: float
    gender: float
    child_total: float
    dependants: float
    socstatus_work_fl: float
    socstatus_pens_fl: float
    own_auto: float
    fl_presence_fl: float
    personal_income: float
    credit: float
    load_num_total: float
    loan_num_closed: float
    edu_2plus: float
    edu_high_unfinished: float
    edu_mid_unfinished: float
    edu_mid: float
    edu_mid_spec: float
    edu_scholar: float
    ms_civil: float
    ms_unmarried: float
    ms_divorced: float
    ms_married: float
    fi_10k_20k: float
    fi_20k_50k: float
    fi_5k_10k: float
    fi_50k_plus: float
    id: int


class YGet(BaseModel):
    target: int
    id: int
