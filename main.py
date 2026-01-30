import os
import json
import base64
import numpy as np
import tensorflow as tf
import cv2
from fastapi import FastAPI, Depends, HTTPException, status, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from groq import Groq
from dotenv import load_dotenv

import models
import database

# --- INIT ---
load_dotenv()
models.Base.metadata.create_all(bind=database.engine)
app = FastAPI(title="NgeMood API")

# --- CONFIG ---
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
GROQ_MODEL = "llama-3.3-70b-versatile"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
try:
    model_fer = tf.keras.models.load_model('fer_model.h5')
    print("✅ AI Model Loaded")
except:
    model_fer = None
    print("⚠️ AI Model NOT Found")

EMOTION_LABELS = ['angry', 'happy', 'neutral', 'sad', 'stress']
MOOD_TRANSLATION = {'angry': 'marah', 'happy': 'senang', 'neutral': 'netral', 'sad': 'sedih', 'stress': 'stres'}

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SCHEMAS ---
class UserAuth(BaseModel):
    email: EmailStr
    password: str

class FaceCheckInReq(BaseModel):
    image: str

class JournalReq(BaseModel):
    text: str

# --- AUTH HELPER ---
def get_password_hash(password): return pwd_context.hash(password)
def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=1440)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str, db: Session = Depends(database.get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None: raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user: raise HTTPException(status_code=401, detail="User not found")
    return user

async def get_user_from_header(authorization: Optional[str] = Header(None), db: Session = Depends(database.get_db)):
    if not authorization: raise HTTPException(status_code=401, detail="Missing Token")
    try:
        scheme, token = authorization.split()
        if scheme.lower() != 'bearer': raise HTTPException(status_code=401, detail="Invalid scheme")
        return await get_current_user(token, db)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")

# --- AI HELPER ---
def ask_groq_json(prompt: str):
    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role":"user","content":prompt}], 
            model=GROQ_MODEL, 
            response_format={"type":"json_object"}
        )
        return json.loads(chat.choices[0].message.content)
    except Exception as e:
        print(f"Groq Error: {e}")
        return None

# --- ENDPOINTS ---
@app.get("/auth/me")
def get_me(user: models.User = Depends(get_user_from_header)):
    return {"id": user.id, "email": user.email, "joined": user.created_at}

@app.post("/auth/register")
def register(user: UserAuth, db: Session = Depends(database.get_db)):
    if db.query(models.User).filter(models.User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email exists")
    new_user = models.User(email=user.email, password_hash=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    return {"msg": "Registered"}

@app.post("/auth/login")
def login(user: UserAuth, db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user or not verify_password(user.password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"access_token": create_access_token({"sub": db_user.email}), "token_type": "bearer"}

@app.post("/face-checkin")
def face_checkin(req: FaceCheckInReq, user: models.User = Depends(get_user_from_header), db: Session = Depends(database.get_db)):
    if not model_fer: raise HTTPException(status_code=500, detail="AI Model Error")
    try:
        encoded = req.image.split(',')[1] if ',' in req.image else req.image
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        face_img = img
        if len(faces) > 0:
            best_face = max(faces, key=lambda rect: rect[2] * rect[3])
            x, y, w, h = best_face
            face_img = img[y:y+h, x:x+w]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_img = clahe.apply(face_img)
        face_img = cv2.resize(face_img, (48, 48)).astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=[0, -1])
        
        preds = model_fer.predict(face_img)
        idx = np.argmax(preds)
        mood_indo = MOOD_TRANSLATION[EMOTION_LABELS[idx]]
        confidence = float(np.max(preds))

        # NEW PERSONA: CHILL & ROASTING TIPIS
        ai_prompt = f"""
        User abis scan muka, hasilnya: {mood_indo} (Confidence: {int(confidence*100)}%).
        
        Peran lo: Temen seumuran yang asik, chill, ngomong lo-gue.
        Tugas: Kasih komentar singkat (1 kalimat).
        Style: 
        - Kalau mood jelek: Support tapi jangan menye-menye. Validasi perasaannya.
        - Kalau mood bagus: Ikut seneng tapi boleh roasting dikit biar gak cringe.
        - Kalau stress: Suruh istirahat dengan gaya santai.
        
        Contoh: "Muka lo kusut amat, tidur gih jangan scroll sosmed mulu." atau "Nah gitu dong senyum, kan cakepan dikit."
        
        Output JSON: {{"recommendation": "..."}}
        """
        ai_res = ask_groq_json(ai_prompt)
        rec = ai_res.get("recommendation", "Jaga mood ya!") if ai_res else "Jaga mood ya!"
        
        db.add(models.Mood(user_id=user.id, mood=mood_indo, confidence=confidence, source='face'))
        db.commit()
        return {"mood": mood_indo, "confidence": confidence, "recommendation": rec}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Gagal proses wajah")

@app.post("/journal")
def journal(req: JournalReq, user: models.User = Depends(get_user_from_header), db: Session = Depends(database.get_db)):
    prompt = f"""
    Analisis curhatan ini: "{req.text}".
    
    Peran lo: Temen deket yang enak diajak ngobrol (Gen Z vibes).
    Style: Santai, Lo-Gue, boleh pake emoji, Real Talk (ngomong fakta).
    
    Tugas:
    1. Tentukan mood (senang/sedih/stres/marah/netral).
    2. Skor emosi.
    3. Respon singkat (maks 2 kalimat). Kalau dia salah, tegor halus. Kalau bener, dukung.

    Output JSON Only:
    {{
        "mood": "...",
        "emotion_score": 0.0,
        "summary": "..."
    }}
    """
    ai_res = ask_groq_json(prompt)
    if not ai_res: raise HTTPException(status_code=500, detail="AI lagi bengong...")

    mood_res = ai_res.get('mood', 'netral').lower()
    if mood_res not in MOOD_TRANSLATION.values(): mood_res = 'netral'

    db.add(models.Journal(user_id=user.id, text=req.text, mood=mood_res, summary=ai_res.get('summary', '')))
    db.add(models.Mood(user_id=user.id, mood=mood_res, confidence=abs(ai_res.get('emotion_score', 0)), source='journal'))
    db.commit()
    return ai_res

@app.get("/moods/history")
def history(user: models.User = Depends(get_user_from_header), db: Session = Depends(database.get_db)):
    return db.query(models.Mood).filter(models.Mood.user_id == user.id).order_by(models.Mood.created_at.desc()).limit(30).all()

@app.delete("/moods/history/{item_id}")
def delete_item(item_id: int, user: models.User = Depends(get_user_from_header), db: Session = Depends(database.get_db)):
    db.query(models.Mood).filter(models.Mood.id == item_id, models.Mood.user_id == user.id).delete()
    db.commit()
    return {"msg": "Deleted"}

@app.delete("/moods/history")
def delete_all(user: models.User = Depends(get_user_from_header), db: Session = Depends(database.get_db)):
    db.query(models.Mood).filter(models.Mood.user_id == user.id).delete()
    db.commit()
    return {"msg": "All deleted"}

@app.get("/moods/recommendation")
def recommendation(user: models.User = Depends(get_user_from_header), db: Session = Depends(database.get_db)):
    moods = db.query(models.Mood).filter(models.Mood.user_id==user.id).order_by(models.Mood.created_at.desc()).limit(7).all()
    if not moods: return {"dominant_mood": "netral", "recommendation": ["Data masih kosong. Jangan males check-in napa!"]}
    
    history_str = ", ".join([f"{m.mood}" for m in moods])
    prompt = f"""
    Histori mood 7 hari terakhir: {history_str}.
    
    Peran: Life Coach tapi gaya tongkrongan.
    Tugas:
    1. Tentukan vibes dominan.
    2. Kasih 3 saran kegiatan yang masuk akal (actionable).
    3. Kalau mood dia jelek terus, kasih saran agak tegas biar dia sadar (Roasting tipis).
    
    Output JSON:
    {{
        "dominant_mood": "...",
        "recommendation": ["Saran 1", "Saran 2", "Saran 3"]
    }}
    """
    ai_res = ask_groq_json(prompt)
    return ai_res if ai_res else {"dominant_mood": "netral", "recommendation": ["Jaga kesehatan mental ya!"]}