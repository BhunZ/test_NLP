# YouTube RAG Scraper - Complete Setup Guide

**For complete beginners** - No prior programming knowledge required!

---

## Table of Contents
1. [What is this?](#what-is-this)
2. [What do you need?](#what-do-you-need)
3. [Step 1: Install Python](#step-1-install-python)
4. [Step 2: Install Node.js (for frontend)](#step-2-install-nodejs-for-frontend)
5. [Step 3: Download the project](#step-3-download-the-project)
6. [Step 4: Install Python dependencies](#step-4-install-python-dependencies)
7. [Step 5: Get API Keys](#step-5-get-api-keys)
8. [Step 6: Run the Backend](#step-6-run-the-backend)
9. [Step 7: Run the Frontend](#step-7-run-the-frontend)
10. [Troubleshooting](#troubleshooting)

---

## What is this?

This is a **YouTube RAG (Retrieval Augmented Generation) system** that:
- Scrapes YouTube videos from NLP courses (Stanford CS224N, CS124, etc.)
- Creates a searchable knowledge base
- Lets you ask questions in Vietnamese or English
- Uses AI to answer with citations from the videos

Think of it as a **smart search engine for YouTube lecture videos**.

---

## What do you need?

| Software | Why | Where to download |
|----------|-----|------------------|
| **Python 3.10+** | Run the backend | https://www.python.org/downloads/ |
| **Node.js 18+** | Run the frontend | https://nodejs.org/ |
| **Git** | Clone the project | https://git-scm.com/ |
| **API Key** | Use AI services | Free at Groq (see below) |

---

## Step 1: Install Python

### Windows
1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.12.x" (or latest)
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Open Command Prompt and type:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`

### Mac
1. Install Homebrew (if not installed):
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/brew/install.sh)"
   ```
2. Install Python:
   ```
   brew install python
   ```
3. Verify:
   ```
   python3 --version
   ```

### Linux (Ubuntu/Debian)
```
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

---

## Step 2: Install Node.js (for frontend)

### Windows & Mac
1. Go to https://nodejs.org/
2. Download the LTS version (left button)
3. Install with default settings
4. Open Terminal/Command Prompt and type:
   ```
   node --version
   npm --version
   ```

### Linux (Ubuntu/Debian)
```
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version
```

---

## Step 3: Download the project

### Using Git (recommended)
```bash
git clone https://github.com/your-username/youtube-rag-scraper.git
cd youtube-rag-scraper
```

### Or download as ZIP
1. Go to https://github.com/your-repo/youtube-rag-scraper
2. Click green "Code" button → "Download ZIP"
3. Extract the ZIP file
4. Open terminal in the extracted folder

---

## Step 4: Install Python dependencies

### Create a virtual environment (recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal line.

### Install requirements
```bash
pip install -r requirements.txt
```

This may take 5-10 minutes as it downloads all the AI libraries.

---

## Step 5: Get API Keys

### Get a FREE Groq API Key (required)
1. Go to https://console.groq.com/
2. Click "Create API Key"
3. Copy the key
4. Create a file called `.env` in the project root:
   ```
   GROQ_API_KEY=your_key_here
   ```

### Optional: Google AI (for evaluation)
- Go to https://aistudio.google.com/app/apikey
- Create a key if you want to use Gemini for evaluation

---

## Step 6: Run the Backend

In your terminal (with virtual environment activated):

```bash
python -m uvicorn backend.api:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Keep this terminal open!**

Test if it's working:
1. Open http://127.0.0.1:8000/api/v1/health in your browser
2. You should see `{"status":"ok"}`

---

## Step 7: Run the Frontend

Open a **new** terminal (don't close the backend):

```bash
cd frontend
npm install
npm run dev
```

You should see:
```
VITE v5.x.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
```

---

## 🎉 You're Done!

Open http://localhost:5173 in your browser.

Try asking:
- "Transformer là gì?" (What is transformer?)
- "How does attention work?"
- "attention mechanism la gi"

---

## Troubleshooting

### "pip: command not found"
- Windows: Add Python to PATH or use `py -m pip install`
- Mac/Linux: `python3 -m pip install`

### "Module not found" errors
- Make sure you ran `pip install -r requirements.txt`
- Make sure virtual environment is activated

### "Port already in use"
- Change port: `python -m uvicorn backend.api:app --port 8001`

### "npm: command not found"
- Reinstall Node.js
- Restart your terminal

### Frontend won't start
- Make sure you're in the `frontend` folder
- Delete `node_modules` and try again: `rm -rf node_modules && npm install`

### Langchain/pydantic version errors
- The requirements.txt has version pins to prevent this
- Make sure you're using Python 3.10-3.12

---

## Need Help?

1. Check the API docs at http://127.0.0.1:8000/docs
2. Check existing issues on GitHub
3. The backend runs on port 8000, frontend on port 5173

---

## Quick Commands Reference

```bash
# Activate virtual environment
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# Run backend
python -m uvicorn backend.api:app --reload --port 8000

# Run frontend
cd frontend && npm run dev

# Install dependencies
pip install -r requirements.txt
```

---

*Last updated: May 2026*