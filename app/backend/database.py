from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

URL = "postgresql://moxeeem:JSqcgw1DEr4P@ep-rough-water-90638384.eu-central-1.aws.neon.tech/bankcustomers"

engine = create_engine(URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
