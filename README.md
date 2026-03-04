# Professional Indian House Price Predictor

A high-end, production-ready house price estimation tool designed specifically for the Indian real estate market. Built with a "Light & Clean" aesthetic and modern full-stack technologies.

## 🚀 Tech Stack

### **Backend**
- **FastAPI**: High-performance Python framework for building APIs.
- **Pydantic**: Robust data validation and settings management.
- **Scikit-Learn**: Used for data cleaning logic and heuristic inference.

### **Frontend**
- **Next.js 15+**: React framework for the modern web.
- **Tailwind CSS**: Utility-first CSS for professional styling.
- **Lucide-React**: Clean, consistent iconography.
- **Framer Motion**: Smooth animations for the "Reveal" effect.

## ✨ Key Features

- **Indian Context**: Handles "BHK" terminology, sqft ranges, and local amenities.
- **Professional UI**: Clean light theme with #2563EB Royal Blue accents.
- **Real-time Prediction**: Instant price estimation with a "fade-in" animation.
- **Currency Formatting**: Results displayed in Indian numbering system (Lakhs/Crores).
- **Validation**: Client-side and server-side validation for property details.

## 🛠️ Installation & Setup

### **1. Backend Setup**
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Server runs at `http://localhost:8000`.

### **2. Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```
Application runs at `http://localhost:3000`.

## 📂 Project Structure

- `backend/`: FastAPI server, data cleaning utilities, and prediction logic.
- `frontend/`: Next.js application with Tailwind CSS and Framer Motion.
- `models/`: (Optional) Storage for pre-trained scikit-learn pipelines.

---
Developed with a focus on UX and precision for Indian home buyers and sellers.
