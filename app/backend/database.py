from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

URL = "DATABASE_URL"

engine = create_engine(URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
