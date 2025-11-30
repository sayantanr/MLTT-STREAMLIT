# MLTT - Multilingual Language Translation & Transliteration System

MLTT is a comprehensive multilingual platform supporting **transliteration, translation, OCR, and speech tools** for over 120 scripts and 132 languages.

---

## Features

- **Transliteration**
  - Supports 120+ scripts (e.g., Devanagari, Bengali, Tamil, Arabic, Cyrillic)
  - 21 Romanization methods (e.g., IAST, ITRANS, WX)
  - Auto-detect source script

- **Translation**
  - Translation to 132+ languages using OpenAI GPT models
  - Quick, accurate, and context-aware translation

- **OCR**
  - Extract text from **images (PNG, JPG, JPEG)** and **PDFs**
  - Integrates **Tesseract OCR** and **pdf2image**

- **Speech Tools**
  - **Speech-to-Text** using OpenAI Whisper
  - **Text-to-Speech** using gTTS
  - Supports multiple languages

- **Batch Processing**
  - Process ZIP files containing multiple documents (planned full implementation)

- **Statistics Dashboard**
  - Tracks processed files, words, scripts, and processing times

---

## Installation

```bash
pip install streamlit aksharamukha openai pytesseract pdf2image pillow gtts SpeechRecognition langdetect matplotlib pandas streamlit-audiorecorder pydub
