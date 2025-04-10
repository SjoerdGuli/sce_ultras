# 🎭 Emotion App

A Python app for reminiscence therapy using facial and message-based emotion recognition.

## ✅ Quickstart

Follow these steps to set up and run the app:

### 1. Clone the Repository

```bash
git clone https://github.com/SjoerdGuli/sce_ultras.git
cd emotion-app
```

### 2. Create and Activate a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
./venv/scripts/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Key to `.env`

The application requires an API key for emotion detection. You must create a `.env` file based on the provided sample:

```bash
cp .env.sample .env
```

Then open `.env` in your favorite editor and replace the placeholder with your actual API key:

```
API_KEY=your_real_api_key_here
```

### 5. Run the App

```bash
python emotion_app.py
```

---

## 📂 Project Structure

```
emotion-app/
├── emotion_app.py
├── requirements.txt
├── .env.sample
├── .env               # (You create this)
└── ...
```

---

## 💡 About

This app uses facial and textual inputs to recognize emotions, designed to support reminiscence therapy. Built with Python and designed to be easily extendable.
