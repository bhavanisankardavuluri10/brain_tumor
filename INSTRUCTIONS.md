# How to Run the Brain Tumor Detection Project

**For B.Tech Students - Simple Step-by-Step Guide**

This guide will help you run the project on your computer. No technical knowledge required.

---

## Step 1: Install Required Software

You need to install 3 programs on your computer:

### 1.1 Install Python

Python is needed to run the AI model.

1. Go to: https://www.python.org/downloads/
2. Download Python (version 3.8 or higher)
3. **IMPORTANT**: During installation, check the box that says "Add Python to PATH"
4. Click "Install Now"
5. Wait for installation to complete

**Check if Python is installed:**
- Open Command Prompt (search "cmd" in Windows)
- Type: `python --version`
- You should see something like "Python 3.12.5"

### 1.2 Install Node.js

Node.js is needed to run the website.

1. Go to: https://nodejs.org/
2. Download the LTS version (recommended for most users)
3. Run the installer
4. Keep clicking "Next" with default settings
5. Click "Install"

**Check if Node.js is installed:**
- Open Command Prompt
- Type: `node --version`
- You should see something like "v20.x.x"

### 1.3 Install VS Code (Optional but Recommended)

VS Code makes it easier to work with the project.

1. Go to: https://code.visualstudio.com/
2. Download and install VS Code
3. Open VS Code after installation

---

## Step 2: Open the Project

1. Extract the ZIP file you received
2. You will see a folder named `Advanced-Brain-Tumor-Detection-in-MRI-Images-using-Deep-Learning`
3. **If you have VS Code:**
   - Right-click on the project folder
   - Select "Open with Code"
   - Go to Terminal menu → New Terminal
4. **If you don't have VS Code:**
   - Open Command Prompt
   - Type: `cd` followed by space
   - Drag and drop the project folder into Command Prompt
   - Press Enter

---

## Step 3: Run the Backend (AI Server)

The backend is the brain of the project. It runs the AI model.

**Open terminal and type these commands one by one:**

```bash
cd backend
```

Press Enter. Then type:

```bash
pip install -r requirements.txt
```

Press Enter and wait (this may take 5-10 minutes).

After installation completes, type:

```bash
python app.py
```

Press Enter.

**You should see:**
```
[OK] Model loaded successfully
* Running on http://127.0.0.1:5000
```

**IMPORTANT: Keep this terminal window open!** The backend must keep running.

---

## Step 4: Run the Frontend (Website)

The frontend is the website where you upload MRI images.

**Open a NEW terminal window** (don't close the backend terminal).

**In the new terminal, type these commands:**

```bash
cd frontend
```

Press Enter. Then type:

```bash
npm install
```

Press Enter and wait (this may take 5-10 minutes).

After installation completes, type:

```bash
npm start
```

Press Enter.

**You should see:**
```
Compiled successfully!
Local: http://localhost:3000
```

**IMPORTANT: Keep this terminal window open too!** The frontend must keep running.

---

## Step 5: Open in Browser

The website will open automatically in your browser.

If it doesn't open automatically:
1. Open any web browser (Chrome, Firefox, Edge)
2. Type in address bar: `http://localhost:3000`
3. Press Enter

**You should see the Brain Tumor Detection website!**

---

## How to Use the Website

1. **Upload an MRI Image:**
   - Click "Browse Files" or drag and drop an image
   - Use images from the `data/raw/Testing` folder for testing

2. **Start Analysis:**
   - Click the "Start Analysis" button
   - Wait a few seconds

3. **View Results:**
   - You will see the tumor type prediction
   - Confidence score shows how sure the AI is
   - View all probability distributions

4. **Upload Another Image:**
   - Click "Clear Study" button
   - Upload a new image and repeat

---

## Common Problems and Solutions

### Problem 1: "Python is not recognized"
**Solution:**
- Uninstall Python
- Reinstall Python and CHECK the box "Add Python to PATH"

### Problem 2: "pip is not recognized"
**Solution:**
- Close all terminals
- Open a new terminal
- Try again

### Problem 3: "Port 5000 is already in use"
**Solution:**
- Close all Command Prompt windows
- Try again
- OR restart your computer

### Problem 4: "Port 3000 is already in use"
**Solution:**
- Close all Command Prompt windows
- Try again

### Problem 5: Installation takes too long
**Solution:**
- Be patient, it can take 10-15 minutes
- Make sure you have good internet connection
- Don't close the terminal while installing

### Problem 6: "Model not loaded"
**Solution:**
- Make sure you have the trained model file in `backend/models/trained/` folder
- The file should be named `final_model.keras`
- If missing, you need to train the model first (see README.md)

### Problem 7: Website not opening
**Solution:**
- Check if backend is running (first terminal should show "Running on...")
- Check if frontend is running (second terminal should show "Compiled successfully")
- Try typing `http://localhost:3000` manually in browser

---

## Important Notes

1. **Always run backend first, then frontend**
2. **Keep both terminal windows open** while using the website
3. **Don't close the terminals** or the website will stop working
4. **To stop the project:**
   - Press `Ctrl + C` in both terminal windows
   - Or simply close the terminal windows

5. **To run again later:**
   - Just repeat Step 3, Step 4, and Step 5
   - No need to reinstall (skip `pip install` and `npm install`)

---

## Quick Commands Summary

**For Backend:**
```bash
cd backend
python app.py
```

**For Frontend (in new terminal):**
```bash
cd frontend
npm start
```

**That's it!** Open browser at http://localhost:3000

---

## Project Folder Structure

You don't need to understand this, but here's what each folder does:

```
Project Folder/
├── backend/           → AI model and server
├── frontend/          → Website interface
├── data/             → MRI images for testing
├── README.md         → Project overview
└── INSTRUCTIONS.md   → This file
```

---

## Need Help?

1. Read the error message carefully
2. Check "Common Problems and Solutions" section above
3. Make sure Python and Node.js are installed correctly
4. Try restarting your computer
5. Ask your project guide or teacher

---

## For Demonstration/Presentation

When showing this project to teachers or examiners:

1. **Before the presentation:**
   - Start backend: `cd backend && python app.py`
   - Start frontend: `cd frontend && npm start`
   - Open http://localhost:3000

2. **During the presentation:**
   - Show the website interface
   - Upload a test MRI image from `data/raw/Testing/`
   - Click "Start Analysis"
   - Explain the AI prediction results

3. **After the presentation:**
   - Press Ctrl+C in both terminals to stop
   - Or just close the terminal windows

---

**Project Status:** Ready to Run
**Last Updated:** January 26, 2026
**Developed By:** B.Tech 4th Year Students

---

**Good luck with your project! 🎓**
