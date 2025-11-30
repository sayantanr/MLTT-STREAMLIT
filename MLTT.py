

"""
MLTT - Multilingual Language Translation & Transliteration System

Installation:
pip install streamlit aksharamukha openai pytesseract pdf2image pillow gtts SpeechRecognition langdetect matplotlib pandas streamlit-audiorecorder pydub

System Requirements:
- Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- Install poppler for pdf2image: https://poppler.freedesktop.org/
"""

import streamlit as st
import aksharamukha.transliterate as transliterate
from openai import OpenAI
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import zipfile
import os
import time
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from gtts import gTTS
import tempfile
from langdetect import detect, DetectorFactory
from audio_recorder_streamlit import audio_recorder
import base64

# Set seed for consistent language detection
DetectorFactory.seed = 0

# ============================================================================
# CONSTANTS
# ============================================================================

AKSHARAMUKHA_SCRIPTS = [
    "Ahom", "Arab", "Ariyaka", "Assamese", "Avestan", "Balinese", "BatakKaro",
    "BatakManda", "BatakPakpak", "BatakSima", "BatakToba", "Bengali", "Bhaiksuki",
    "Brahmi", "Buginese", "Buhid", "Burmese", "Chakma", "Cham", "RussianCyrillic",
    "Devanagari", "DivesAkuru", "Dogra", "Elym", "Ethi", "GunjalaGondi",
    "MasaramGondi", "Grantha", "GranthaPandya", "Gujarati", "Hanunoo", "Hatr",
    "Hebrew", "Hebr-Ar", "Armi", "Phli", "Prti", "Hiragana", "Katakana",
    "Javanese", "Kaithi", "Kannada", "Kawi", "KhamtiShan", "Kharoshthi",
    "Khmer", "Khojki", "KhomThai", "Khudawadi", "Lao", "LaoPali", "Lepcha",
    "Limbu", "Mahajani", "Makasar", "Malayalam", "Mani", "Marchen",
    "MeeteiMayek", "Modi", "Mon", "Mongolian", "Mro", "Multani", "Nbat",
    "Nandinagari", "Newa", "Narb", "OldPersian", "Sogo", "Sarb", "Oriya",
    "Pallava", "Palm", "Arab-Fa", "PhagsPa", "Phnx", "Phlp", "Gurmukhi",
    "Ranjana", "Rejang", "HanifiRohingya", "BarahaNorth", "BarahaSouth",
    "RomanColloquial", "PersianDMG", "HK", "IAST", "IASTPali", "ISO", "ISOPali",
    "Itrans", "RomanLoC", "RomanReadable", "HebrewSBL", "SLP1", "Type", "Latn",
    "Titus", "Velthuis", "WX", "Samr", "Santali", "Saurashtra", "Shahmukhi",
    "Shan", "Sharada", "Siddham", "Sinhala", "Sogd", "SoraSompeng", "Soyombo",
    "Sundanese", "SylotiNagri", "Syrn", "Syre", "Syrj", "Tagalog", "Tagbanwa",
    "TaiLaing", "Takri", "Tamil", "TamilExtended", "TamilBrahmi", "Telugu",
    "Thaana", "Thai", "TaiTham", "LaoTham", "KhuenTham", "LueTham", "Tibetan",
    "Tirhuta", "Ugar", "Urdu", "Vatteluttu", "Wancho", "WarangCiti",
    "ZanabazarSquare"
]



ROMANIZATION_METHODS = [
    "HK", "ITRANS", "Velthuis", "IAST", "IASTPali", "ISO", "ISOPali", "Titus", 
    "SLP1", "WX", "Roman", "RomanColloquial", "Aksharamukha", "SemiticTypeable", 
    "ISO259Hebrew", "SBLHebrew", "ISO233Arabic", "DMGPersian"
]

TRANSLATION_LANGUAGES = [
    "Afrikaans", "Albanian", "Amharic", "Arabic", "Armenian", "Azerbaijani", "Basque",
    "Belarusian", "Bengali", "Bosnian", "Bulgarian", "Catalan", "Cebuano", "Chinese (Simplified)",
    "Chinese (Traditional)", "Corsican", "Croatian", "Czech", "Danish", "Dutch", "English",
    "Esperanto", "Estonian", "Finnish", "French", "Frisian", "Galician", "Georgian", "German",
    "Greek", "Gujarati", "Haitian Creole", "Hausa", "Hawaiian", "Hebrew", "Hindi", "Hmong",
    "Hungarian", "Icelandic", "Igbo", "Indonesian", "Irish", "Italian", "Japanese", "Javanese",
    "Kannada", "Kazakh", "Khmer", "Kinyarwanda", "Korean", "Kurdish", "Kyrgyz", "Lao",
    "Latin", "Latvian", "Lithuanian", "Luxembourgish", "Macedonian", "Malagasy", "Malay",
    "Malayalam", "Maltese", "Maori", "Marathi", "Mongolian", "Myanmar (Burmese)", "Nepali",
    "Norwegian", "Nyanja (Chichewa)", "Odia (Oriya)", "Pashto", "Persian", "Polish",
    "Portuguese", "Punjabi", "Romanian", "Russian", "Samoan", "Scots Gaelic", "Serbian",
    "Sesotho", "Shona", "Sindhi", "Sinhala", "Slovak", "Slovenian", "Somali", "Spanish",
    "Sundanese", "Swahili", "Swedish", "Tagalog (Filipino)", "Tajik", "Tamil", "Tatar",
    "Telugu", "Thai", "Turkish", "Turkmen", "Ukrainian", "Urdu", "Uyghur", "Uzbek",
    "Vietnamese", "Welsh", "Xhosa", "Yiddish", "Yoruba", "Zulu", "Assamese", "Bhojpuri",
    "Dhivehi", "Dogri", "Konkani", "Maithili", "Manipuri", "Mizo", "Sanskrit", "Tsonga",
    "Twi", "Tigrinya", "Bambara", "Lingala", "Luganda", "Sepedi", "Krio", "Ilocano",
    "Meiteilon (Manipuri)", "Oromo", "Quechua", "Guarani", "Aymara"
]

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'stats' not in st.session_state:
    st.session_state.stats = {
        'files_processed': 0,
        'languages_detected': [],
        'scripts_used': [],
        'total_words': 0,
        'processing_times': []
    }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_openai_client() -> OpenAI:
    """Initialize OpenAI client with API key from sidebar."""
    api_key = st.session_state.get('openai_api_key', '')
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=api_key)

def transliterate_text(source: str, target: str, text: str) -> str:
    """Transliterate text using Aksharamukha."""
    try:
        result = transliterate.process(source, target, text)
        return result
    except Exception as e:
        st.error(f"Transliteration error: {str(e)}")
        return text

def translate_text(text: str, target_lang: str, api_key: str) -> str:
    """Translate text using OpenAI API."""
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a translator. Translate the following text to {target_lang}. Provide only the translation without any additional text."},
                {"role": "user", "content": text}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def detect_script(text: str) -> str:
    """Detect script of input text."""
    try:
        lang = detect(text)
        script_map = {
            'hi': 'Devanagari', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
            'ml': 'Malayalam', 'kn': 'Kannada', 'gu': 'Gujarati', 'pa': 'Gurmukhi',
            'or': 'Oriya', 'ar': 'Arabic', 'he': 'Hebrew', 'th': 'Thai',
            'my': 'Burmese', 'km': 'Khmer', 'lo': 'Lao', 'si': 'Sinhala',
            'ja': 'Hiragana', 'zh-cn': 'Chinese', 'zh-tw': 'Chinese',
            'ru': 'Cyrillic', 'ur': 'Urdu'
        }
        return script_map.get(lang, 'Devanagari')
    except:
        return 'Devanagari'

def ocr_image(image: Image.Image) -> str:
    """Extract text from image using Tesseract."""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"OCR error: {str(e)}")
        return ""

def ocr_pdf(pdf_bytes: bytes) -> str:
    """Extract text from PDF using OCR."""
    try:
        images = convert_from_bytes(pdf_bytes)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n\n"
        return text
    except Exception as e:
        st.error(f"PDF OCR error: {str(e)}")
        return ""

def text_to_speech(text: str, lang: str = 'en') -> bytes:
    """Convert text to speech using gTTS."""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        st.error(f"TTS error: {str(e)}")
        return b""

def process_zip_batch(zip_file, source_script: str, target_scripts: List[str], 
                     translate_lang: str = None, api_key: str = None) -> bytes:
    """Process ZIP file containing text files."""
    output_zip = io.BytesIO()
    
    with zipfile.ZipFile(zip_file, 'r') as zip_in:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for file_info in zip_in.filelist:
                if file_info.filename.endswith('.txt'):
                    content = zip_in.read(file_info.filename).decode('utf-8')
                    
                    # Transliterate to each target script
                    for target in target_scripts:
                        result = transliterate_text(source_script, target, content)
                        output_filename = f"{os.path.splitext(file_info.filename)[0]}_{target}.txt"
                        zip_out.writestr(output_filename, result)
                    
                    # Translate if requested
                    if translate_lang and api_key:
                        translated = translate_text(content, translate_lang, api_key)
                        output_filename = f"{os.path.splitext(file_info.filename)[0]}_{translate_lang}.txt"
                        zip_out.writestr(output_filename, translated)
                    
                    # Update stats
                    st.session_state.stats['files_processed'] += 1
                    st.session_state.stats['total_words'] += len(content.split())
    
    output_zip.seek(0)
    return output_zip.read()

def update_stats(script: str, words: int, proc_time: float):
    """Update statistics."""
    st.session_state.stats['scripts_used'].append(script)
    st.session_state.stats['total_words'] += words
    st.session_state.stats['processing_times'].append(proc_time)

# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="MLTT - Multilingual Translation & Transliteration",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 MLTT - Multilingual Language Translation & Transliteration System")
    
    # Sidebar for API Key
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
        st.markdown("---")
        st.markdown("### Features")
        st.markdown("✅ 120+ Scripts")
        st.markdown("✅ 21 Romanization Methods")
        st.markdown("✅ 132 Languages")
        st.markdown("✅ OCR (Image/PDF)")
        st.markdown("✅ Speech Tools")
        st.markdown("✅ Batch Processing")
    
    # Main Tabs
    tabs = st.tabs([
        "🔤 Transliteration",
        "🌍 Translation",
        "📄 OCR",
        "🎤 Speech Tools",
        "📦 Batch ZIP Processor",
        "📊 Statistics"
    ])
    
    # ========================================================================
    # TAB 1: TRANSLITERATION
    # ========================================================================
    with tabs[0]:
        st.header("Aksharamukha Transliteration Engine")
        
        col1, col2 = st.columns(2)
        with col1:
            source_script = st.selectbox("Source Script", ["Auto-detect"] + AKSHARAMUKHA_SCRIPTS + ROMANIZATION_METHODS)
        with col2:
            target_script = st.selectbox("Target Script", AKSHARAMUKHA_SCRIPTS + ROMANIZATION_METHODS)
        
        input_text = st.text_area("Input Text", height=200, placeholder="Enter text to transliterate...")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            transliterate_btn = st.button("🔄 Transliterate", type="primary")
        
        if transliterate_btn and input_text:
            start_time = time.time()
            
            # Auto-detect source if needed
            if source_script == "Auto-detect":
                detected = detect_script(input_text)
                st.info(f"Detected script: {detected}")
                source_script = detected
            
            result = transliterate_text(source_script, target_script, input_text)
            
            st.text_area("Output Text", value=result, height=200)
            
            # Download button
            st.download_button(
                "💾 Download Output",
                data=result,
                file_name=f"transliteration_{target_script}.txt",
                mime="text/plain"
            )
            
            # Update stats
            proc_time = time.time() - start_time
            update_stats(target_script, len(input_text.split()), proc_time)
    
    # ========================================================================
    # TAB 2: TRANSLATION
    # ========================================================================
    with tabs[1]:
        st.header("AI-Powered Translation (132 Languages)")
        
        if not st.session_state.get('openai_api_key'):
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use translation.")
        else:
            target_lang = st.selectbox("Target Language", TRANSLATION_LANGUAGES)
            
            input_text = st.text_area("Input Text", height=200, placeholder="Enter text to translate...", key="trans_input")
            
            if st.button("🌐 Translate", type="primary"):
                if input_text:
                    start_time = time.time()
                    
                    with st.spinner("Translating..."):
                        result = translate_text(input_text, target_lang, st.session_state.openai_api_key)
                    
                    st.text_area("Translated Text", value=result, height=200)
                    
                    st.download_button(
                        "💾 Download Translation",
                        data=result,
                        file_name=f"translation_{target_lang}.txt",
                        mime="text/plain"
                    )
                    
                    proc_time = time.time() - start_time
                    st.success(f"✅ Translation completed in {proc_time:.2f}s")
    
    # ========================================================================
    # TAB 3: OCR
    # ========================================================================
    with tabs[2]:
        st.header("OCR - Image & PDF Text Extraction")
        
        file_type = st.radio("Select File Type", ["Image (PNG/JPG)", "PDF"])
        
        if file_type == "Image (PNG/JPG)":
            uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("🔍 Extract Text", type="primary"):
                    with st.spinner("Extracting text..."):
                        extracted_text = ocr_image(image)
                    
                    st.text_area("Extracted Text", value=extracted_text, height=200)
                    
                    # Options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.checkbox("Transliterate Result"):
                            src = st.selectbox("From", AKSHARAMUKHA_SCRIPTS, key="ocr_src")
                            tgt = st.selectbox("To", AKSHARAMUKHA_SCRIPTS, key="ocr_tgt")
                            if st.button("Transliterate OCR"):
                                result = transliterate_text(src, tgt, extracted_text)
                                st.text_area("Transliterated", value=result, height=150)
                    
                    with col2:
                        if st.checkbox("Translate Result") and st.session_state.get('openai_api_key'):
                            lang = st.selectbox("To Language", TRANSLATION_LANGUAGES, key="ocr_lang")
                            if st.button("Translate OCR"):
                                result = translate_text(extracted_text, lang, st.session_state.openai_api_key)
                                st.text_area("Translated", value=result, height=150)
        
        else:  # PDF
            uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
            
            if uploaded_file:
                if st.button("🔍 Extract Text from PDF", type="primary"):
                    with st.spinner("Processing PDF..."):
                        pdf_bytes = uploaded_file.read()
                        extracted_text = ocr_pdf(pdf_bytes)
                    
                    st.text_area("Extracted Text", value=extracted_text, height=300)
                    
                    st.download_button(
                        "💾 Download Extracted Text",
                        data=extracted_text,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
    
    # ========================================================================
    # TAB 4: SPEECH TOOLS
    # ========================================================================
    with tabs[3]:
        st.header("Speech Tools")
        
        tab_speech = st.tabs(["🎤 Speech-to-Text", "🔊 Text-to-Speech"])
        
        # Speech-to-Text
        with tab_speech[0]:
            st.subheader("Record Audio and Convert to Text")
            
            audio_bytes = audio_recorder(text="Click to record", icon_size="2x")
            
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                
                if st.button("📝 Convert to Text") and st.session_state.get('openai_api_key'):
                    try:
                        # Save audio temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        
                        # Use OpenAI Whisper
                        client = get_openai_client()
                        with open(tmp_path, "rb") as audio_file:
                            transcript = client.audio.transcriptions.create(
                                model="whisper-1",
                                file=audio_file
                            )
                        
                        st.text_area("Transcription", value=transcript.text, height=150)
                        
                        # Cleanup
                        os.unlink(tmp_path)
                    except Exception as e:
                        st.error(f"Transcription error: {str(e)}")
        
        # Text-to-Speech
        with tab_speech[1]:
            st.subheader("Convert Text to Speech")
            
            tts_text = st.text_area("Enter Text", height=150, placeholder="Type text to convert to speech...")
            
            tts_lang_map = {
                'English': 'en', 'Hindi': 'hi', 'Spanish': 'es', 'French': 'fr',
                'German': 'de', 'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru',
                'Japanese': 'ja', 'Korean': 'ko', 'Chinese': 'zh', 'Arabic': 'ar'
            }
            
            tts_lang = st.selectbox("Language", list(tts_lang_map.keys()))
            
            if st.button("🔊 Generate Speech"):
                if tts_text:
                    with st.spinner("Generating audio..."):
                        audio_data = text_to_speech(tts_text, tts_lang_map[tts_lang])
                    
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3")
                        st.download_button(
                            "💾 Download Audio",
                            data=audio_data,
                            file_name="speech.mp3",
                            mime="audio/mp3"
                        )
    
    # ========================================================================
    # TAB 5: BATCH ZIP PROCESSOR
    # ========================================================================
    with tabs[4]:
        st.header("Batch ZIP Processing")
        
        st.info("📦 Upload a ZIP file containing TXT files for batch processing")
        
        uploaded_zip = st.file_uploader("Upload ZIP File", type=['zip'])
        
        if uploaded_zip:
            col1, col2 = st.columns(2)
            
            with col1:
                source_script = st.selectbox("Source Script", AKSHARAMUKHA_SCRIPTS + ROMANIZATION_METHODS, key="batch_src")
                target_scripts = st.multiselect("Target Scripts", AKSHARAMUKHA_SCRIPTS + ROMANIZATION_METHODS, key="batch_tgt")
            
            with col2:
                enable_translation = st.checkbox("Enable Translation")
                if enable_translation:
                    trans_lang = st.selectbox("Translation Language", TRANSLATION_LANGUAGES, key="batch_lang")
                else:
                    trans_lang = None
            
            if st.button("⚙️ Process ZIP", type="primary"):
                if target_scripts:
                    with st.spinner("Processing batch..."):
                        start_time = time.time()
                        
                        api_key = st.session_state.get('openai_api_key') if enable_translation else None
                        output_zip = process_zip_batch(uploaded_zip, source_script, target_scripts, trans_lang, api_key)
                        
                        proc_time = time.time() - start_time
                    
                    st.success(f"✅ Processing completed in {proc_time:.2f}s")
                    
                    st.download_button(
                        "💾 Download Processed ZIP",
                        data=output_zip,
                        file_name="processed_output.zip",
                        mime="application/zip"
                    )
                else:
                    st.warning("Please select at least one target script.")
    
    # ========================================================================
    # TAB 6: STATISTICS
    # ========================================================================
    with tabs[5]:
        st.header("Statistics Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Files Processed", st.session_state.stats['files_processed'])
        with col2:
            st.metric("Total Words", st.session_state.stats['total_words'])
        with col3:
            unique_langs = len(set(st.session_state.stats['languages_detected']))
            st.metric("Languages Detected", unique_langs)
        with col4:
            avg_time = sum(st.session_state.stats['processing_times']) / len(st.session_state.stats['processing_times']) if st.session_state.stats['processing_times'] else 0
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        
        st.markdown("---")
        
        if st.session_state.stats['scripts_used']:
            # Script usage frequency
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Script Usage Frequency")
                script_counts = pd.Series(st.session_state.stats['scripts_used']).value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                script_counts.plot(kind='bar', ax=ax, color='skyblue')
                ax.set_xlabel("Script")
                ax.set_ylabel("Count")
                ax.set_title("Most Used Scripts")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            
            with col2:
                st.subheader("Processing Times")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(st.session_state.stats['processing_times'], marker='o', color='green')
                ax.set_xlabel("Operation #")
                ax.set_ylabel("Time (seconds)")
                ax.set_title("Processing Time per Operation")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        else:
            st.info("No statistics available yet. Start using the application to see data here.")
        
        if st.button("🔄 Reset Statistics"):
            st.session_state.stats = {
                'files_processed': 0,
                'languages_detected': [],
                'scripts_used': [],
                'total_words': 0,
                'processing_times': []
            }
            st.success("Statistics reset successfully!")
            st.rerun()

if __name__ == "__main__":
    main()
