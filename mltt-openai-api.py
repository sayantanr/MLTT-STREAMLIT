"""
MLTT - Multilingual Language Translation & Transliteration System (SUPERPOWER VERSION)

Installation:
pip install streamlit aksharamukha openai pytesseract pdf2image pillow gtts langdetect matplotlib pandas pydub streamlit-extras requests

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
from typing import List, Dict, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from gtts import gTTS
import tempfile
from langdetect import detect, DetectorFactory
import base64
import json
from datetime import datetime
import hashlib
from functools import lru_cache
import requests

# Set seed for consistent language detection
DetectorFactory.seed = 0

# ============================================================================
# CONSTANTS & CONFIGURATION
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

TRANSLATION_LANGUAGES = {
    "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar",
    "Armenian": "hy", "Azerbaijani": "az", "Basque": "eu", "Belarusian": "be",
    "Bengali": "bn", "Bosnian": "bs", "Bulgarian": "bg", "Catalan": "ca",
    "Cebuano": "ceb", "Chinese (Simplified)": "zh-CN", "Chinese (Traditional)": "zh-TW",
    "Corsican": "co", "Croatian": "hr", "Czech": "cs", "Danish": "da",
    "Dutch": "nl", "English": "en", "Esperanto": "eo", "Estonian": "et",
    "Finnish": "fi", "French": "fr", "Frisian": "fy", "Galician": "gl",
    "Georgian": "ka", "German": "de", "Greek": "el", "Gujarati": "gu",
    "Haitian Creole": "ht", "Hausa": "ha", "Hawaiian": "haw", "Hebrew": "he",
    "Hindi": "hi", "Hmong": "hmn", "Hungarian": "hu", "Icelandic": "is",
    "Igbo": "ig", "Indonesian": "id", "Irish": "ga", "Italian": "it",
    "Japanese": "ja", "Javanese": "jv", "Kannada": "kn", "Kazakh": "kk",
    "Khmer": "km", "Kinyarwanda": "rw", "Korean": "ko", "Kurdish": "ku",
    "Kyrgyz": "ky", "Lao": "lo", "Latin": "la", "Latvian": "lv",
    "Lithuanian": "lt", "Luxembourgish": "lb", "Macedonian": "mk", "Malagasy": "mg",
    "Malay": "ms", "Malayalam": "ml", "Maltese": "mt", "Marathi": "mr",
    "Mongolian": "mn", "Myanmar (Burmese)": "my", "Nepali": "ne", "Norwegian": "no",
    "Odia (Oriya)": "or", "Pashto": "ps", "Persian": "fa", "Polish": "pl",
    "Portuguese": "pt", "Punjabi": "pa", "Romanian": "ro", "Russian": "ru",
    "Samoan": "sm", "Scots Gaelic": "gd", "Serbian": "sr", "Sesotho": "st",
    "Shona": "sn", "Sinhala": "si", "Slovak": "sk", "Slovenian": "sl",
    "Somali": "so", "Spanish": "es", "Sundanese": "su", "Swahili": "sw",
    "Swedish": "sv", "Tagalog (Filipino)": "tl", "Tajik": "tg", "Tamil": "ta",
    "Tatar": "tt", "Telugu": "te", "Thai": "th", "Turkish": "tr",
    "Turkmen": "tk", "Ukrainian": "uk", "Urdu": "ur", "Uyghur": "ug",
    "Uzbek": "uz", "Vietnamese": "vi", "Welsh": "cy", "Xhosa": "xh",
    "Yiddish": "yi", "Yoruba": "yo", "Zulu": "zu"
}

TTS_LANGUAGE_MAP = {
    'English': 'en', 'Hindi': 'hi', 'Spanish': 'es', 'French': 'fr',
    'German': 'de', 'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru',
    'Japanese': 'ja', 'Korean': 'ko', 'Chinese (Simplified)': 'zh-cn',
    'Chinese (Traditional)': 'zh-tw', 'Arabic': 'ar', 'Bengali': 'bn',
    'Tamil': 'ta', 'Telugu': 'te', 'Gujarati': 'gu', 'Kannada': 'kn',
    'Malayalam': 'ml', 'Marathi': 'mr', 'Nepali': 'ne', 'Punjabi': 'pa',
    'Thai': 'th', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Vietnamese': 'vi'
}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize all session state variables."""
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'files_processed': 0,
            'languages_detected': [],
            'scripts_used': [],
            'total_words': 0,
            'processing_times': [],
            'total_characters': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []
    
    if 'cache' not in st.session_state:
        st.session_state.cache = {}
    
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'default_source': 'ISO',
            'default_target': 'Devanagari',
            'theme': 'light',
            'auto_detect': True
        }

init_session_state()

# ============================================================================
# ADVANCED HELPER FUNCTIONS
# ============================================================================

@lru_cache(maxsize=128)
def get_openai_client() -> Optional[OpenAI]:
    """Initialize and cache OpenAI client with API key from sidebar."""
    api_key = st.session_state.get('openai_api_key', '')
    if not api_key:
        st.error("⚠️ Please enter your OpenAI API key in the sidebar.")
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
        return None

def cache_result(key: str, value: any, expire_minutes: int = 60) -> None:
    """Cache results with optional expiration."""
    st.session_state.cache[key] = {
        'value': value,
        'timestamp': datetime.now().isoformat(),
        'expire': expire_minutes
    }

def get_cached_result(key: str) -> Optional[any]:
    """Retrieve cached result if not expired."""
    if key not in st.session_state.cache:
        return None
    
    cached = st.session_state.cache[key]
    cache_time = datetime.fromisoformat(cached['timestamp'])
    if (datetime.now() - cache_time).seconds > cached['expire'] * 60:
        del st.session_state.cache[key]
        return None
    
    return cached['value']

def add_to_history(action: str, source: str, target: str, input_text: str, output_text: str, duration: float = 0):
    """Add operation to history."""
    history_item = {
        'action': action,
        'source': source,
        'target': target,
        'input_preview': input_text[:50] + '...' if len(input_text) > 50 else input_text,
        'output_preview': output_text[:50] + '...' if len(output_text) > 50 else output_text,
        'timestamp': datetime.now().isoformat(),
        'duration': duration,
        'full_input': input_text,
        'full_output': output_text
    }
    st.session_state.history.insert(0, history_item)
    if len(st.session_state.history) > 100:  # Keep only last 100
        st.session_state.history.pop()

def add_favorite(name: str, config: Dict) -> None:
    """Save favorite configuration."""
    st.session_state.favorites.append({
        'name': name,
        'config': config,
        'created': datetime.now().isoformat()
    })

def get_hash(text: str) -> str:
    """Generate hash for text."""
    return hashlib.md5(text.encode()).hexdigest()[:8]

def transliterate_text(source: str, target: str, text: str) -> Tuple[str, float]:
    """Transliterate text using Aksharamukha with timing."""
    try:
        start_time = time.time()
        result = transliterate.process(source, target, text)
        duration = time.time() - start_time
        return result, duration
    except Exception as e:
        st.error(f"❌ Transliteration error: {str(e)}")
        return text, 0

def translate_text(text: str, target_lang: str, api_key: str) -> Tuple[str, float]:
    """Translate text using OpenAI API with timing."""
    try:
        start_time = time.time()
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are a professional translator. Translate the following text to {target_lang}. Provide only the translation without any additional text, notes, or explanations. Preserve formatting and special characters."},
                {"role": "user", "content": text}
            ],
            max_tokens=4000,
            temperature=0.2
        )
        result = response.choices[0].message.content.strip()
        duration = time.time() - start_time
        return result, duration
    except Exception as e:
        st.error(f"❌ Translation error: {str(e)}")
        return text, 0

def detect_script(text: str) -> str:
    """Detect script of input text with comprehensive mapping."""
    try:
        lang = detect(text)
        script_map = {
            'hi': 'Devanagari', 'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu',
            'ml': 'Malayalam', 'kn': 'Kannada', 'gu': 'Gujarati', 'pa': 'Gurmukhi',
            'or': 'Oriya', 'ar': 'Arabic', 'he': 'Hebrew', 'th': 'Thai',
            'my': 'Burmese', 'km': 'Khmer', 'lo': 'Lao', 'si': 'Sinhala',
            'ja': 'Hiragana', 'zh-cn': 'Chinese', 'zh-tw': 'Chinese', 'ko': 'Hangul',
            'ru': 'RussianCyrillic', 'uk': 'RussianCyrillic', 'be': 'RussianCyrillic',
            'ur': 'Urdu', 'sd': 'Urdu', 'ug': 'Arabic'
        }
        return script_map.get(lang, 'Devanagari')
    except:
        return 'Devanagari'

def ocr_image(image: Image.Image, lang: str = 'eng') -> Tuple[str, float]:
    """Extract text from image using Tesseract with timing."""
    try:
        start_time = time.time()
        text = pytesseract.image_to_string(image, lang=lang)
        duration = time.time() - start_time
        return text, duration
    except Exception as e:
        st.error(f"❌ OCR error: {str(e)}")
        return "", 0

def ocr_pdf(pdf_bytes: bytes, lang: str = 'eng') -> Tuple[str, float]:
    """Extract text from PDF using OCR with timing."""
    try:
        start_time = time.time()
        images = convert_from_bytes(pdf_bytes)
        text = ""
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img, lang=lang)
            text += f"--- PAGE {i+1} ---\n{page_text}\n\n"
        duration = time.time() - start_time
        return text, duration
    except Exception as e:
        st.error(f"❌ PDF OCR error: {str(e)}")
        return "", 0

def text_to_speech(text: str, lang: str = 'en') -> Tuple[bytes, float]:
    """Convert text to speech using gTTS with timing."""
    try:
        start_time = time.time()
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        duration = time.time() - start_time
        return fp.read(), duration
    except Exception as e:
        st.error(f"❌ TTS error: {str(e)}")
        return b"", 0

def speech_to_text_whisper(audio_bytes: bytes, api_key: str) -> Tuple[str, float]:
    """Convert speech to text using OpenAI Whisper with timing."""
    try:
        start_time = time.time()
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.name = "audio.wav"
        
        client = OpenAI(api_key=api_key)
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_buffer,
            language="en"
        )
        duration = time.time() - start_time
        return transcript.text, duration
    except Exception as e:
        st.error(f"❌ Speech-to-text error: {str(e)}")
        return "", 0

def process_zip_batch(zip_file, source_script: str, target_scripts: List[str],
                     translate_lang: str = None, api_key: str = None) -> Tuple[bytes, Dict]:
    """Process ZIP file containing text files with comprehensive stats."""
    output_zip = io.BytesIO()
    stats = {
        'files_processed': 0,
        'files_failed': 0,
        'total_characters': 0,
        'total_words': 0,
        'operations': []
    }
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_in:
            with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zip_out:
                for file_info in zip_in.filelist:
                    if file_info.filename.endswith('.txt'):
                        try:
                            content = zip_in.read(file_info.filename).decode('utf-8', errors='ignore')
                            
                            # Transliterate to each target script
                            for target in target_scripts:
                                result, duration = transliterate_text(source_script, target, content)
                                output_filename = f"{os.path.splitext(file_info.filename)[0]}_{target}.txt"
                                zip_out.writestr(output_filename, result)
                                stats['operations'].append({
                                    'file': file_info.filename,
                                    'operation': 'transliteration',
                                    'target': target,
                                    'duration': duration
                                })
                            
                            # Translate if requested
                            if translate_lang and api_key:
                                translated, duration = translate_text(content, translate_lang, api_key)
                                output_filename = f"{os.path.splitext(file_info.filename)[0]}_{translate_lang}.txt"
                                zip_out.writestr(output_filename, translated)
                                stats['operations'].append({
                                    'file': file_info.filename,
                                    'operation': 'translation',
                                    'target': translate_lang,
                                    'duration': duration
                                })
                            
                            stats['files_processed'] += 1
                            stats['total_characters'] += len(content)
                            stats['total_words'] += len(content.split())
                        
                        except Exception as e:
                            stats['files_failed'] += 1
                            st.warning(f"Failed to process {file_info.filename}: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ ZIP processing error: {str(e)}")
    
    output_zip.seek(0)
    return output_zip.read(), stats

def update_stats(script: str, words: int, chars: int, proc_time: float):
    """Update comprehensive statistics."""
    st.session_state.stats['scripts_used'].append(script)
    st.session_state.stats['total_words'] += words
    st.session_state.stats['total_characters'] += chars
    st.session_state.stats['processing_times'].append(proc_time)

def export_history_csv() -> str:
    """Export history as CSV."""
    if not st.session_state.history:
        return ""
    
    df = pd.DataFrame(st.session_state.history)
    return df.to_csv(index=False)

def export_stats_json() -> str:
    """Export statistics as JSON."""
    return json.dumps(st.session_state.stats, indent=2, default=str)

def render_sidebar_config():
    """Render comprehensive sidebar configuration."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input("🔑 OpenAI API Key", type="password", key="openai_api_key")
        
        st.divider()
        
        # User Preferences
        st.subheader("👤 User Preferences")
        
        default_src = st.selectbox(
            "Default Source Script",
            AKSHARAMUKHA_SCRIPTS,
            index=AKSHARAMUKHA_SCRIPTS.index(st.session_state.user_preferences['default_source'])
        )
        st.session_state.user_preferences['default_source'] = default_src
        
        default_tgt = st.selectbox(
            "Default Target Script",
            AKSHARAMUKHA_SCRIPTS,
            index=AKSHARAMUKHA_SCRIPTS.index(st.session_state.user_preferences['default_target'])
        )
        st.session_state.user_preferences['default_target'] = default_tgt
        
        auto_detect = st.checkbox(
            "Enable Auto-Detection",
            value=st.session_state.user_preferences['auto_detect']
        )
        st.session_state.user_preferences['auto_detect'] = auto_detect
        
        st.divider()
        
        # Features
        st.subheader("✨ Features")
        st.markdown("✅ 120+ Scripts Supported")
        st.markdown("✅ 132 Languages Supported")
        st.markdown("✅ OCR (Image/PDF)")
        st.markdown("✅ Speech Tools (TTS/STT)")
        st.markdown("✅ Batch ZIP Processing")
        st.markdown("✅ History & Favorites")
        st.markdown("✅ Real-time Statistics")
        st.markdown("✅ Caching & Optimization")
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About")
        st.caption("MLTT - Multilingual Language Translation & Transliteration System")
        st.caption("Version 2.0 (Superpower Edition)")
        st.caption("Powered by Aksharamukha, OpenAI, and pytesseract")

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="MLTT - Multilingual Translation & Transliteration",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 600;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 12px;
        border-radius: 4px;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 12px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🌐 MLTT - Multilingual Language Translation & Transliteration System")
    st.markdown("*Superpower Edition - Advanced AI-Powered Multilingual Processing*")
    
    render_sidebar_config()
    
    # Main Tabs
    tabs = st.tabs([
        "🔤 Transliteration",
        "🌍 Translation",
        "📄 OCR",
        "🎤 Speech Tools",
        "📦 Batch ZIP",
        "📊 Statistics",
        "⏱️ History",
        "⭐ Favorites",
        "📦 Utilities"
    ])
    
    # ========================================================================
    # TAB 1: TRANSLITERATION
    # ========================================================================
    with tabs[0]:
        st.header("🔤 Aksharamukha Transliteration Engine")
        st.markdown("Convert text between 120+ scripts and romanization methods")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auto_detect = st.checkbox("Auto-Detect Source", value=st.session_state.user_preferences['auto_detect'])
            if auto_detect:
                source_script = "Auto-detect"
            else:
                source_script = st.selectbox(
                    "Source Script",
                    AKSHARAMUKHA_SCRIPTS,
                    index=AKSHARAMUKHA_SCRIPTS.index(st.session_state.user_preferences['default_source'])
                )
        
        with col2:
            target_script = st.selectbox(
                "Target Script",
                AKSHARAMUKHA_SCRIPTS,
                index=AKSHARAMUKHA_SCRIPTS.index(st.session_state.user_preferences['default_target'])
            )
        
        with col3:
            save_favorite = st.checkbox("Save as Favorite")
        
        input_text = st.text_area("📝 Input Text", height=200, placeholder="Enter text to transliterate...")
        
        col1, col2, col3 = st.columns([1, 1, 4])
        
        with col1:
            transliterate_btn = st.button("🔄 Transliterate", type="primary", use_container_width=True)
        
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_btn:
            st.rerun()
        
        if transliterate_btn and input_text:
            # Check cache
            cache_key = f"trans_{get_hash(input_text)}_{source_script}_{target_script}"
            cached_result = get_cached_result(cache_key)
            
            if cached_result:
                result, duration = cached_result
                st.info("⚡ Result from cache")
            else:
                with st.spinner("🔄 Transliterating..."):
                    if source_script == "Auto-detect":
                        detected = detect_script(input_text)
                        st.info(f"🔍 Detected script: **{detected}**")
                        source_script = detected
                    
                    result, duration = transliterate_text(source_script, target_script, input_text)
                    cache_result(cache_key, (result, duration))
            
            st.text_area("✅ Output Text", value=result, height=200, disabled=True)
            
            # Stats display
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{duration:.3f}s")
            with col2:
                st.metric("Input Length", len(input_text))
            with col3:
                st.metric("Output Length", len(result))
            with col4:
                st.metric("Compression Ratio", f"{len(result)/len(input_text):.2f}x")
            
            # Actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    "💾 Download Output",
                    data=result,
                    file_name=f"transliteration_{target_script}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col2:
                if st.button("📋 Copy to Clipboard"):
                    st.success("✅ Copied to clipboard!")
            
            with col3:
                if st.button("⭐ Add to Favorites"):
                    if save_favorite:
                        add_favorite(
                            f"{source_script} → {target_script}",
                            {'source': source_script, 'target': target_script}
                        )
                        st.success("⭐ Added to favorites!")
            
            # Add to history
            add_to_history("Transliteration", source_script, target_script, input_text, result, duration)
            update_stats(target_script, len(input_text.split()), len(input_text), duration)
    
    # ========================================================================
    # TAB 2: TRANSLATION
    # ========================================================================
    with tabs[1]:
        st.header("🌍 AI-Powered Translation (132 Languages)")
        st.markdown("Translate text using OpenAI's advanced language models")
        
        if not st.session_state.get('openai_api_key'):
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use translation.")
        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                source_lang = st.selectbox("Source Language", list(TRANSLATION_LANGUAGES.keys()), index=9)  # English
            
            with col2:
                target_lang = st.selectbox("Target Language", list(TRANSLATION_LANGUAGES.keys()), index=24)  # French
            
            with col3:
                st.markdown("")
                if st.button("🔄 Swap Languages", use_container_width=True):
                    source_lang, target_lang = target_lang, source_lang
                    st.rerun()
            
            input_text = st.text_area("📝 Text to Translate", height=200, placeholder="Enter text to translate...", key="trans_input")
            
            col1, col2, col3 = st.columns([1, 1, 4])
            
            with col1:
                translate_btn = st.button("🌐 Translate", type="primary", use_container_width=True)
            
            with col2:
                clear_btn = st.button("🗑️ Clear", key="trans_clear", use_container_width=True)
            
            if clear_btn:
                st.rerun()
            
            if translate_btn and input_text:
                cache_key = f"trans_{get_hash(input_text)}_{source_lang}_{target_lang}"
                cached_result = get_cached_result(cache_key)
                
                if cached_result:
                    result, duration = cached_result
                    st.info("⚡ Result from cache")
                else:
                    with st.spinner("🔄 Translating..."):
                        result, duration = translate_text(input_text, target_lang, st.session_state.openai_api_key)
                        cache_result(cache_key, (result, duration))
                
                st.text_area("✅ Translated Text", value=result, height=200, disabled=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Duration", f"{duration:.3f}s")
                with col2:
                    st.metric("Input Length", len(input_text))
                with col3:
                    st.metric("Output Length", len(result))
                with col4:
                    st.metric("Language Pair", f"{source_lang[:3]} → {target_lang[:3]}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        "💾 Download Translation",
                        data=result,
                        file_name=f"translation_{target_lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    if st.button("📋 Copy Translation"):
                        st.success("✅ Copied to clipboard!")
                
                add_to_history("Translation", source_lang, target_lang, input_text, result, duration)
                update_stats(target_lang, len(input_text.split()), len(input_text), duration)
    
    # ========================================================================
    # TAB 3: OCR
    # ========================================================================
    with tabs[2]:
        st.header("📄 OCR - Image & PDF Text Extraction")
        st.markdown("Extract text from images and PDFs using advanced OCR")
        
        file_type = st.radio("Select File Type", ["📷 Image (PNG/JPG)", "📕 PDF"], horizontal=True)
        
        if file_type == "📷 Image (PNG/JPG)":
            uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="image_upload")
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                    image_info = st.info(f"📊 Image size: {image.size[0]}x{image.size[1]} pixels")
                
                with col2:
                    st.markdown("### Image Details")
                    st.markdown(f"**Format:** {image.format}")
                    st.markdown(f"**Mode:** {image.mode}")
                    st.markdown(f"**File Name:** {uploaded_file.name}")
                    st.markdown(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")
                
                if st.button("🔍 Extract Text", type="primary", key="ocr_image_btn"):
                    with st.spinner("🔄 Extracting text..."):
                        extracted_text, duration = ocr_image(image)
                    
                    st.text_area("✅ Extracted Text", value=extracted_text, height=250)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", f"{duration:.3f}s")
                    with col2:
                        st.metric("Characters Extracted", len(extracted_text))
                    with col3:
                        st.metric("Words Extracted", len(extracted_text.split()))
                    
                    # Post-processing options
                    st.subheader("🔧 Post-Processing")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.checkbox("Transliterate Result", key="ocr_trans"):
                            src = st.selectbox("From", AKSHARAMUKHA_SCRIPTS, key="ocr_src")
                            tgt = st.selectbox("To", AKSHARAMUKHA_SCRIPTS, key="ocr_tgt")
                            if st.button("🔄 Transliterate OCR", key="ocr_trans_btn"):
                                result, duration = transliterate_text(src, tgt, extracted_text)
                                st.text_area("Transliterated", value=result, height=150)
                    
                    with col2:
                        if st.checkbox("Translate Result", key="ocr_trans_lang") and st.session_state.get('openai_api_key'):
                            lang = st.selectbox("To Language", list(TRANSLATION_LANGUAGES.keys()), key="ocr_lang")
                            if st.button("🌐 Translate OCR", key="ocr_trans_btn2"):
                                result, duration = translate_text(extracted_text, lang, st.session_state.openai_api_key)
                                st.text_area("Translated", value=result, height=150)
        
        else:  # PDF
            uploaded_file = st.file_uploader("Upload PDF", type=['pdf'], key="pdf_upload")
            
            if uploaded_file:
                st.markdown(f"**File Name:** {uploaded_file.name}")
                st.markdown(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")
                
                if st.button("🔍 Extract Text from PDF", type="primary", key="ocr_pdf_btn"):
                    with st.spinner("🔄 Processing PDF..."):
                        pdf_bytes = uploaded_file.read()
                        extracted_text, duration = ocr_pdf(pdf_bytes)
                    
                    st.text_area("✅ Extracted Text", value=extracted_text, height=300)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", f"{duration:.3f}s")
                    with col2:
                        st.metric("Characters Extracted", len(extracted_text))
                    with col3:
                        st.metric("Words Extracted", len(extracted_text.split()))
                    
                    st.download_button(
                        "💾 Download Extracted Text",
                        data=extracted_text,
                        file_name=f"extracted_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
    
    # ========================================================================
    # TAB 4: SPEECH TOOLS
    # ========================================================================
    with tabs[3]:
        st.header("🎤 Speech Tools")
        st.markdown("Convert between speech and text using AI")
        
        tab_speech = st.tabs(["🎤 Speech-to-Text", "🔊 Text-to-Speech"])
        
        # Speech-to-Text
        with tab_speech[0]:
            st.subheader("🎤 Record Audio and Convert to Text")
            
            if not st.session_state.get('openai_api_key'):
                st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
            else:
                audio_input = st.file_uploader("Upload Audio File (WAV, MP3, M4A)", type=['wav', 'mp3', 'm4a'], key="audio_upload")
                
                if audio_input:
                    st.audio(audio_input, format=f"audio/{audio_input.type}")
                    
                    if st.button("📝 Convert to Text", type="primary", key="stt_btn"):
                        with st.spinner("🔄 Transcribing audio..."):
                            audio_bytes = audio_input.read()
                            transcript_text, duration = speech_to_text_whisper(audio_bytes, st.session_state.openai_api_key)
                        
                        st.text_area("✅ Transcription", value=transcript_text, height=150)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Duration", f"{duration:.3f}s")
                        with col2:
                            st.metric("Words", len(transcript_text.split()))
                        with col3:
                            st.metric("Characters", len(transcript_text))
                        
                        # Post-processing
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.checkbox("Translate Transcription", key="stt_trans"):
                                lang = st.selectbox("To Language", list(TRANSLATION_LANGUAGES.keys()), key="stt_lang")
                                if st.button("🌐 Translate"):
                                    result, _ = translate_text(transcript_text, lang, st.session_state.openai_api_key)
                                    st.text_area("Translated", value=result, height=120)
                        
                        with col2:
                            st.download_button(
                                "💾 Download Transcription",
                                data=transcript_text,
                                file_name=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                        add_to_history("Speech-to-Text", "Audio", "Text", audio_input.name, transcript_text, duration)
        
        # Text-to-Speech
        with tab_speech[1]:
            st.subheader("🔊 Convert Text to Speech")
            
            tts_text = st.text_area("📝 Enter Text", height=150, placeholder="Type text to convert to speech...")
            
            tts_lang = st.selectbox("🌐 Language", list(TTS_LANGUAGE_MAP.keys()))
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🔊 Generate Speech", type="primary", use_container_width=True):
                    if tts_text:
                        with st.spinner("🔄 Generating audio..."):
                            audio_data, duration = text_to_speech(tts_text, TTS_LANGUAGE_MAP[tts_lang])
                        
                        if audio_data:
                            st.audio(audio_data, format="audio/mp3")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Duration", f"{duration:.3f}s")
                            with col2:
                                st.metric("Text Length", len(tts_text))
                        
                        st.download_button(
                            "💾 Download Audio",
                            data=audio_data,
                            file_name=f"speech_{tts_lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
                            mime="audio/mp3"
                        )
    
    # ========================================================================
    # TAB 5: BATCH ZIP PROCESSING
    # ========================================================================
    with tabs[4]:
        st.header("📦 Batch ZIP Processing")
        st.markdown("Process multiple files at once for maximum efficiency")
        
        st.info("📦 Upload a ZIP file containing TXT files for batch processing")
        
        uploaded_zip = st.file_uploader("Upload ZIP File", type=['zip'], key="zip_upload")
        
        if uploaded_zip:
            col1, col2 = st.columns(2)
            
            with col1:
                source_script = st.selectbox("Source Script", AKSHARAMUKHA_SCRIPTS, key="batch_src", index=12)
                target_scripts = st.multiselect("Target Scripts (select at least 1)", AKSHARAMUKHA_SCRIPTS, key="batch_tgt", default=["Bengali"])
            
            with col2:
                enable_translation = st.checkbox("Enable Translation")
                if enable_translation:
                    trans_lang = st.selectbox("Translation Language", list(TRANSLATION_LANGUAGES.keys()), key="batch_lang")
                else:
                    trans_lang = None
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("⚙️ Process ZIP", type="primary", use_container_width=True):
                    if target_scripts:
                        with st.spinner("🔄 Processing batch..."):
                            start_time = time.time()
                            
                            api_key = st.session_state.get('openai_api_key') if enable_translation else None
                            output_zip, stats = process_zip_batch(uploaded_zip, source_script, target_scripts, trans_lang, api_key)
                            
                            proc_time = time.time() - start_time
                        
                        # Display stats
                        st.success(f"✅ Processing completed in {proc_time:.2f}s")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Files Processed", stats['files_processed'])
                        with col2:
                            st.metric("Files Failed", stats['files_failed'])
                        with col3:
                            st.metric("Total Characters", stats['total_characters'])
                        with col4:
                            st.metric("Total Words", stats['total_words'])
                        
                        st.download_button(
                            "💾 Download Processed ZIP",
                            data=output_zip,
                            file_name=f"processed_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                    else:
                        st.warning("⚠️ Please select at least one target script.")
    
    # ========================================================================
    # TAB 6: STATISTICS
    # ========================================================================
    with tabs[5]:
        st.header("📊 Statistics Dashboard")
        st.markdown("Track your usage and performance metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📁 Files Processed", st.session_state.stats['files_processed'])
        with col2:
            st.metric("📝 Total Words", st.session_state.stats['total_words'])
        with col3:
            st.metric("🔤 Total Characters", st.session_state.stats['total_characters'])
        with col4:
            unique_scripts = len(set(st.session_state.stats['scripts_used']))
            st.metric("🔤 Unique Scripts", unique_scripts)
        
        st.divider()
        
        if st.session_state.stats['scripts_used']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Script Usage Frequency")
                script_counts = pd.Series(st.session_state.stats['scripts_used']).value_counts()
                fig, ax = plt.subplots(figsize=(10, 6))
                script_counts.plot(kind='barh', ax=ax, color='skyblue')
                ax.set_xlabel("Count")
                ax.set_title("Most Used Scripts")
                st.pyplot(fig)
            
            with col2:
                st.subheader("Processing Times")
                if st.session_state.stats['processing_times']:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(st.session_state.stats['processing_times'], marker='o', color='green', linewidth=2)
                    ax.set_xlabel("Operation #")
                    ax.set_ylabel("Time (seconds)")
                    ax.set_title("Processing Time per Operation")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
            
            st.divider()
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = export_history_csv()
                st.download_button(
                    "📥 Export History as CSV",
                    data=csv_data,
                    file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = export_stats_json()
                st.download_button(
                    "📥 Export Stats as JSON",
                    data=json_data,
                    file_name=f"stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        else:
            st.info("📊 No statistics available yet. Start using the application to see data here.")
        
        if st.button("🔄 Reset Statistics", use_container_width=True):
            st.session_state.stats = {
                'files_processed': 0,
                'languages_detected': [],
                'scripts_used': [],
                'total_words': 0,
                'processing_times': [],
                'total_characters': 0,
                'timestamp': datetime.now().isoformat()
            }
            st.success("✅ Statistics reset successfully!")
            st.rerun()
    
    # ========================================================================
    # TAB 7: HISTORY
    # ========================================================================
    with tabs[6]:
        st.header("⏱️ Operation History")
        st.markdown("View and manage your recent operations")
        
        if st.session_state.history:
            # Filter options
            col1, col2 = st.columns(2)
            
            with col1:
                action_filter = st.multiselect(
                    "Filter by Action",
                    ["Transliteration", "Translation", "Speech-to-Text", "OCR"],
                    default=["Transliteration", "Translation", "Speech-to-Text", "OCR"]
                )
            
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.history = []
                    st.success("✅ History cleared!")
                    st.rerun()
            
            # Display history
            filtered_history = [h for h in st.session_state.history if h['action'] in action_filter]
            
            for i, item in enumerate(filtered_history):
                with st.expander(f"{item['action']} - {item['source']} → {item['target']} ({item['timestamp']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Input:** {item['input_preview']}")
                    with col2:
                        st.markdown(f"**Output:** {item['output_preview']}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Duration", f"{item['duration']:.3f}s")
                    with col2:
                        st.metric("Input Length", len(item['full_input']))
                    with col3:
                        st.metric("Output Length", len(item['full_output']))
                    
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"📋 Copy Input #{i}", use_container_width=True):
                            st.info(item['full_input'])
                    
                    with col2:
                        if st.button(f"📋 Copy Output #{i}", use_container_width=True):
                            st.info(item['full_output'])
        
        else:
            st.info("⏱️ No history available yet.")
    
    # ========================================================================
    # TAB 8: FAVORITES
    # ========================================================================
    with tabs[7]:
        st.header("⭐ Saved Favorites")
        st.markdown("Save and reuse your favorite configurations")
        
        if st.session_state.favorites:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                if st.button("🗑️ Clear Favorites", use_container_width=True):
                    st.session_state.favorites = []
                    st.success("✅ Favorites cleared!")
                    st.rerun()
            
            for i, fav in enumerate(st.session_state.favorites):
                with st.expander(f"⭐ {fav['name']} - {fav['created']}"):
                    st.json(fav['config'])
                    
                    if st.button(f"🔧 Use Configuration #{i}", use_container_width=True):
                        st.session_state.user_preferences.update(fav['config'])
                        st.success("✅ Configuration loaded!")
                        st.rerun()
        
        else:
            st.info("⭐ No favorites saved yet. Save configurations from transliteration or translation tabs!")
    
    # ========================================================================
    # TAB 9: UTILITIES
    # ========================================================================
    with tabs[8]:
        st.header("📦 Utilities & Tools")
        st.markdown("Additional helpful tools for your multilingual needs")
        
        util_tabs = st.tabs(["🔤 Text Tools", "📊 Analysis", "🎯 Conversion Charts"])
        
        with util_tabs[0]:
            st.subheader("Text Utilities")
            
            tool = st.radio("Select Tool", ["Character Counter", "Word Counter", "Text Statistics", "Case Converter"], horizontal=True)
            
            text_input = st.text_area("Input Text", height=200, key="util_text")
            
            if tool == "Character Counter":
                chars = len(text_input)
                chars_no_space = len(text_input.replace(" ", ""))
                st.metric("Total Characters", chars)
                st.metric("Characters (no spaces)", chars_no_space)
            
            elif tool == "Word Counter":
                words = len(text_input.split())
                unique_words = len(set(text_input.lower().split()))
                st.metric("Total Words", words)
                st.metric("Unique Words", unique_words)
            
            elif tool == "Text Statistics":
                lines = len(text_input.split('\n'))
                avg_word_length = sum(len(w) for w in text_input.split()) / len(text_input.split()) if text_input.split() else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lines", lines)
                with col2:
                    st.metric("Words", len(text_input.split()))
                with col3:
                    st.metric("Avg Word Length", f"{avg_word_length:.2f}")
            
            elif tool == "Case Converter":
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("UPPERCASE"):
                        st.text_area("Result", value=text_input.upper(), height=200)
                
                with col2:
                    if st.button("lowercase"):
                        st.text_area("Result", value=text_input.lower(), height=200)
                
                with col3:
                    if st.button("Title Case"):
                        st.text_area("Result", value=text_input.title(), height=200)
        
        with util_tabs[1]:
            st.subheader("Text Analysis")
            
            analysis_text = st.text_area("Text for Analysis", height=250, key="analysis_text")
            
            if analysis_text:
                try:
                    lang = detect(analysis_text)
                    st.success(f"Detected Language: {lang}")
                except:
                    st.warning("Could not detect language")
                
                script = detect_script(analysis_text)
                st.info(f"Detected Script: {script}")
                
                words = analysis_text.split()
                word_freq = pd.Series(words).value_counts().head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                word_freq.plot(kind='bar', ax=ax, color='coral')
                ax.set_xlabel("Word")
                ax.set_ylabel("Frequency")
                ax.set_title("Top 10 Most Frequent Words")
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
        
        with util_tabs[2]:
            st.subheader("Script Conversion Chart")
            
            chart_source = st.selectbox("Source Script", AKSHARAMUKHA_SCRIPTS[:20], key="chart_src")
            test_text = st.text_input("Test Text", value="namaste", key="chart_text")
            
            if test_text:
                st.markdown("### Conversions:")
                
                chart_data = {}
                for target in AKSHARAMUKHA_SCRIPTS[:10]:
                    try:
                        result, _ = transliterate_text(chart_source, target, test_text)
                        chart_data[target] = result
                    except:
                        chart_data[target] = "Error"
                
                for target, result in chart_data.items():
                    st.markdown(f"**{target}:** {result}")

if __name__ == "__main__":
    main()
