from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base, SessionLocal


class Client(Base):
    __tablename__ = "d_clients"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, primary_key=True)
    age = Column(Integer)
    gender = Column(Integer)
    education = Column(String(50))
    marital_status = Column(String(50))
    child_total = Column(Integer)
    dependants = Column(Integer)
    socstatus_work_fl = Column(Integer)
    socstatus_pens_fl = Column(Integer)
    reg_address_province = Column(String(50))
    fact_address_province = Column(String(50))
    postal_address_province = Column(String(50))
    fl_presence_fl = Column(Integer)
    own_auto = Column(Integer)

    jobs = relationship("Job",
                        back_populates="client")

    last_credit = relationship("LastCredit",
                               back_populates="client")

    pens = relationship("Pens",
                        back_populates="client")

    salary = relationship("Salary",
                          back_populates="client")

    target = relationship("Target",
                          back_populates="client")

    work = relationship("Work",
                        back_populates="client")

    loan = relationship("Loan",
                        back_populates="client")


class Job(Base):
    __tablename__ = "d_job"
    __table_args__ = {"schema": "public"}

    gen_industry = Column(String(50))
    gen_title = Column(String(50))
    job_dir = Column(String(50))
    work_time = Column(Float)

    id_client = Column(Integer, ForeignKey('public.d_clients.id'),
                       primary_key=True)
    client = relationship("Client",
                          back_populates="jobs")


class LastCredit(Base):
    __tablename__ = "d_last_credit"
    __table_args__ = {"schema": "public"}

    credit = Column(Float)
    term = Column(Integer)
    fst_payment = Column(Float)
    id_client = Column(Integer, ForeignKey('public.d_clients.id'),
                       primary_key=True)
    client = relationship("Client",
                          back_populates="last_credit")


class Pens(Base):
    __tablename__ = "d_pens"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, ForeignKey('public.d_clients.id'),
                primary_key=True)
    client = relationship("Client",
                          back_populates="pens")
    flag = Column(Integer)
    COMMENT = Column(String(50))


class Salary(Base):
    __tablename__ = "d_salary"
    __table_args__ = {"schema": "public"}

    family_income = Column(String(50))
    personal_income = Column(Float)

    id_client = Column(Integer, ForeignKey('public.d_clients.id'),
                       primary_key=True)
    client = relationship("Client",
                          back_populates="salary")


class Target(Base):
    __tablename__ = "d_target"
    __table_args__ = {"schema": "public"}

    agreement_rk = Column(Integer, primary_key=True)

    id_client = Column(Integer, ForeignKey('public.d_clients.id'))
    client = relationship("Client",
                          back_populates="target")

    target = Column(Integer)


class Work(Base):
    __tablename__ = "d_work"
    __table_args__ = {"schema": "public"}

    id = Column(Integer, ForeignKey('public.d_clients.id'),
                primary_key=True)
    client = relationship("Client",
                          back_populates="work")

    flag = Column(Integer)
    COMMENT = Column(String(50))


class CloseLoan(Base):
    __tablename__ = "d_close_loan"
    __table_args__ = {"schema": "public"}

    id_loan = Column(Integer, ForeignKey('public.d_loan.id_loan'),
                     primary_key=True)
    loan = relationship("Loan",
                        back_populates="closed_fl")

    closed_fl = Column(Integer)


class Loan(Base):
    __tablename__ = "d_loan"
    __table_args__ = {"schema": "public"}

    id_loan = Column(Integer, primary_key=True)

    id_client = Column(Integer, ForeignKey('public.d_clients.id'))
    client = relationship("Client",
                          back_populates="loan")

    closed_fl = relationship("CloseLoan",
                             back_populates="loan")


class Merged(Base):
    __tablename__ = "1merged"
    __table_args__ = {"schema": "public"}

    agreement_rk = Column(Integer)
    age = Column(Integer)
    gender = Column(Integer)
    education = Column(String(50))
    marital_status = Column(String(50))
    child_total = Column(Integer)
    dependants = Column(Integer)
    socstatus_work_fl = Column(Integer)
    socstatus_pens_fl = Column(Integer)
    reg_address_province = Column(String(50))
    fact_address_province = Column(String(50))
    postal_address_province = Column(String(50))
    fl_presence_fl = Column(Integer)
    own_auto = Column(Integer)
    gen_industry = Column(String(50))
    gen_title = Column(String(50))
    job_dir = Column(String(50))
    work_time = Column(Float, nullable=True)
    family_income = Column(String(50))
    personal_income = Column(Float)
    credit = Column(Float)
    term = Column(Integer)
    fst_payment = Column(Float)
    load_num_total = Column(Integer)
    loan_num_closed = Column(Integer)
    target = Column(Integer)
    id = Column(Integer, primary_key=True)


class WithoutDuplicates(Base):
    __tablename__ = "2without_duplicates"
    __table_args__ = {"schema": "public"}

    agreement_rk = Column(Integer)
    age = Column(Integer)
    gender = Column(Integer)
    education = Column(String(50))
    marital_status = Column(String(50))
    child_total = Column(Integer)
    dependants = Column(Integer)
    socstatus_work_fl = Column(Integer)
    socstatus_pens_fl = Column(Integer)
    reg_address_province = Column(String(50))
    fact_address_province = Column(String(50))
    postal_address_province = Column(String(50))
    fl_presence_fl = Column(Integer)
    own_auto = Column(Integer)
    gen_industry = Column(String(50))
    gen_title = Column(String(50))
    job_dir = Column(String(50))
    work_time = Column(Float)
    family_income = Column(String(50))
    personal_income = Column(Float)
    credit = Column(Float)
    term = Column(Integer)
    fst_payment = Column(Float)
    load_num_total = Column(Integer)
    loan_num_closed = Column(Integer)
    target = Column(Integer)
    id = Column(Integer, primary_key=True)


class FilledNans(Base):
    __tablename__ = "3filled_nans"
    __table_args__ = {"schema": "public"}

    agreement_rk = Column(Integer)
    age = Column(Integer)
    gender = Column(Integer)
    education = Column(String(50))
    marital_status = Column(String(50))
    child_total = Column(Integer)
    dependants = Column(Integer)
    socstatus_work_fl = Column(Integer)
    socstatus_pens_fl = Column(Integer)
    reg_address_province = Column(String(50))
    fact_address_province = Column(String(50))
    postal_address_province = Column(String(50))
    fl_presence_fl = Column(Integer)
    own_auto = Column(Integer)
    gen_industry = Column(String(50))
    gen_title = Column(String(50))
    job_dir = Column(String(50))
    work_time = Column(Float)
    family_income = Column(String(50))
    personal_income = Column(Float)
    credit = Column(Float)
    term = Column(Integer)
    fst_payment = Column(Float)
    load_num_total = Column(Integer)
    loan_num_closed = Column(Integer)
    target = Column(Integer)
    id = Column(Integer, primary_key=True)


class Xtrain(Base):
    __tablename__ = "xtrain"
    __table_args__ = {"schema": "public"}

    agreement_rk = Column(Float)
    age = Column(Float)
    gender = Column(Float)
    child_total = Column(Float)
    dependants = Column(Float)
    socstatus_work_fl = Column(Float)
    socstatus_pens_fl = Column(Float)
    own_auto = Column(Float)
    fl_presence_fl = Column(Float)
    personal_income = Column(Float)
    credit = Column(Float)
    load_num_total = Column(Float)
    loan_num_closed = Column(Float)
    edu_2plus = Column(Float, name="EDUCATION_Два и более высших образован")
    edu_high_unfinished = Column(Float, name="EDUCATION_Неоконченное высшее")
    edu_mid_unfinished = Column(Float, name="EDUCATION_Неполное среднее")
    edu_mid = Column(Float, name="education_среднее")
    edu_mid_spec = Column(Float, name="EDUCATION_Среднее специальное")
    edu_scholar = Column(Float, name="EDUCATION_Ученая степень")
    ms_civil = Column(Float, name="MARITAL_STATUS_Гражданский брак")
    ms_unmarried = Column(Float, name="MARITAL_STATUS_Не состоял в браке")
    ms_divorced = Column(Float, name="MARITAL_STATUS_Разведен(а)")
    ms_married = Column(Float, name="MARITAL_STATUS_Состою в браке")
    fi_10k_20k = Column(Float, name="FAMILY_INCOME_от 10000 до 20000 руб.")
    fi_20k_50k = Column(Float, name="FAMILY_INCOME_от 20000 до 50000 руб.")
    fi_5k_10k = Column(Float, name="FAMILY_INCOME_от 5000 до 10000 руб.")
    fi_50k_plus = Column(Float, name="FAMILY_INCOME_свыше 50000 руб.")
    id = Column(Integer, primary_key=True)


class Xtest(Base):
    __tablename__ = "xtest"
    __table_args__ = {"schema": "public"}

    agreement_rk = Column(Float)
    age = Column(Float)
    gender = Column(Float)
    child_total = Column(Float)
    dependants = Column(Float)
    socstatus_work_fl = Column(Float)
    socstatus_pens_fl = Column(Float)
    own_auto = Column(Float)
    fl_presence_fl = Column(Float)
    personal_income = Column(Float)
    credit = Column(Float)
    load_num_total = Column(Float)
    loan_num_closed = Column(Float)
    edu_2plus = Column(Float, name="EDUCATION_Два и более высших образован")
    edu_high_unfinished = Column(Float, name="EDUCATION_Неоконченное высшее")
    edu_mid_unfinished = Column(Float, name="EDUCATION_Неполное среднее")
    edu_mid = Column(Float, name="education_среднее")
    edu_mid_spec = Column(Float, name="EDUCATION_Среднее специальное")
    edu_scholar = Column(Float, name="EDUCATION_Ученая степень")
    ms_civil = Column(Float, name="MARITAL_STATUS_Гражданский брак")
    ms_unmarried = Column(Float, name="MARITAL_STATUS_Не состоял в браке")
    ms_divorced = Column(Float, name="MARITAL_STATUS_Разведен(а)")
    ms_married = Column(Float, name="MARITAL_STATUS_Состою в браке")
    fi_10k_20k = Column(Float, name="FAMILY_INCOME_от 10000 до 20000 руб.")
    fi_20k_50k = Column(Float, name="FAMILY_INCOME_от 20000 до 50000 руб.")
    fi_5k_10k = Column(Float, name="FAMILY_INCOME_от 5000 до 10000 руб.")
    fi_50k_plus = Column(Float, name="FAMILY_INCOME_свыше 50000 руб.")
    id = Column(Integer, primary_key=True)


class Ytrain(Base):
    __tablename__ = "ytrain"
    __table_args__ = {"schema": "public"}

    target = Column(Integer)
    id = Column(Integer, primary_key=True)


class Ytest(Base):
    __tablename__ = "ytest"
    __table_args__ = {"schema": "public"}

    target = Column(Integer)
    id = Column(Integer, primary_key=True)


if __name__ == "__main__":
    session = SessionLocal()
    results = (session.query(Client).limit(5).all())
    for result in results:
        print(
            f'''
            id = {result.id}
            age = {result.age}
            gender = {result.gender}
            education = {result.education}
            '''
        )
