from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, Boolean, DateTime
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import os
from urllib.parse import urlparse

DATABASE_URL = "sqlite:///database.db"

# Define the database connection
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define table
class UserSession(Base):
    __tablename__ = 'user_session'
    session_id = Column(String, primary_key=True)
    username = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_modified = Column(DateTime, default=func.now(), nullable=False)

    def __repr__(self):
        return f"<UserSession(session_id='{self.session_id}', username='{self.username}', is_active={self.is_active}, last_modified={self.last_modified})>"


def initialise_database():
    Base.metadata.create_all(engine)


def add_session(username, session_id) -> None:
    with Session() as session:
        new_session = UserSession(username=username, session_id=session_id)
        session.add(new_session)
        session.commit()

def set_session_inactive(session_id) -> None:
    with Session() as session:
        session.query(UserSession)\
            .filter(UserSession.session_id == session_id)\
            .update({"is_active": False})
        session.commit()

def get_active_session_ids(username):
    with Session() as session:
        sessions = session.query(UserSession.session_id)\
            .filter(UserSession.username == username, UserSession.is_active == True)\
            .order_by(UserSession.last_modified.desc())\
            .all()
        return [s[0] for s in sessions]

def update_last_modified(session_id):
    with Session() as session:
        # Fetch the session object
        user_session = session.query(UserSession).filter_by(session_id=session_id).one()
        if user_session:
            user_session.last_modified = func.now()
            session.commit()