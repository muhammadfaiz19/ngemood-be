from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Enum, ForeignKey
from datetime import datetime, timedelta, timezone
from database import Base

# Helper untuk waktu Indonesia (WIB = UTC+7)
def get_wib_time():
    return datetime.now(timezone(timedelta(hours=7)))

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True)
    password_hash = Column(Text)
    created_at = Column(DateTime, default=get_wib_time)

class Mood(Base):
    __tablename__ = "moods"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    mood = Column(String(20))
    confidence = Column(Float)
    source = Column(Enum('face', 'journal'))
    created_at = Column(DateTime, default=get_wib_time)

class Journal(Base):
    __tablename__ = "journals"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    text = Column(Text)
    mood = Column(String(20))
    summary = Column(Text)
    created_at = Column(DateTime, default=get_wib_time)