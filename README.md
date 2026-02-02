# NgeMood - Backend API 🐍

Backend server berbasis **FastAPI** yang berfungsi sebagai **otak aplikasi NgeMood**.  
Menangani **logika bisnis**, **autentikasi pengguna**, **koneksi database**, serta **inferensi AI** (Wajah & Teks).

---

## ⚙️ Prasyarat File

Pastikan file-file berikut tersedia di **root folder backend/**:

- `fer_model.h5`  
  Model hasil training dari folder `ml/`.

- `haarcascade_frontalface_default.xml`  
  File XML OpenCV untuk deteksi wajah  
  (dapat diunduh dari repository resmi OpenCV).

- `.env`  
  File konfigurasi environment.

---

## 🔧 Instalasi & Setup

Masuk ke folder backend:

```bash
cd backend
```

Install dependencies:

```bash
pip install fastapi uvicorn sqlalchemy pymysql python-jose[cryptography] passlib[bcrypt] python-multipart tensorflow numpy opencv-python-headless groq python-dotenv email-validator
```

> ⚠️ **Catatan**  
> Jika terjadi error pada `bcrypt`, gunakan versi berikut:
>
> ```bash
> pip install bcrypt==3.2.2
> ```

---

## ⚙️ Konfigurasi Environment (`.env`)

Buat file `.env` dan isi dengan konfigurasi berikut:

```env
DATABASE_URL=mysql+pymysql://root:@localhost/ngemood
JWT_SECRET=rahasia_super_aman_ngemood_2026
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Catatan penting:**
- Ganti `root:@localhost` sesuai dengan `user:password` database Anda.
- Gunakan API Key Groq milik Anda sendiri.

---

## 🏃‍♂️ Menjalankan Server

Jalankan server menggunakan **Uvicorn** (development mode):

```bash
uvicorn main:app --reload
```

- **API Base URL**:  
  `http://localhost:8000`

---

## 🧠 Fitur & Logic AI

### 😶 Face Check-In
- Menerima gambar wajah dalam format **Base64**
- **Preprocessing**:
  - Deteksi wajah (Haar Cascade)
  - Crop wajah
  - Grayscale
  - CLAHE (peningkatan kontras)
  - Resize
- **Inference**:
  - Prediksi emosi menggunakan model `fer_model.h5`
- **Generative AI**:
  - Hasil emosi dikirim ke **Groq LLM**
  - Menghasilkan saran singkat bernuansa *"roasting tipis tapi peduli"*

---

### ✍️ Journaling
- Menerima teks curhatan pengguna
- Diproses menggunakan **Groq LLM**
- Menggunakan **Prompt Engineering khusus**
- Respon AI dibuat seperti **teman Gen Z**:
  - Santai
  - Supportive
  - Relatable

---

## 📡 Daftar Endpoints

### 🔐 Auth
- `POST /auth/register` — Daftar akun
- `POST /auth/login` — Login user
- `GET /auth/me` — Ambil profil user

### 🙂 Mood & AI
- `POST /face-checkin` — Deteksi emosi wajah
- `POST /journal` — Analisis jurnal

### 📊 Riwayat & Insight
- `GET /moods/history` — Ambil riwayat mood
- `DELETE /moods/history` — Hapus riwayat
- `GET /moods/recommendation` — Analisis tren mood mingguan

---

✨ **NgeMood Backend API**  
Fondasi logika, data, dan AI untuk pengalaman emosional yang lebih sadar dan kontekstual.
