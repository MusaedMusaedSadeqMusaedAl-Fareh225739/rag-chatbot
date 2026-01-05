"""
🧠 RAG Document Chatbot - Enhanced Professional Edition
Built by Musaed Al-Fareh

Features:
- Premium Light/Dark theme with professional styling
- Refined typography and spacing
- Subtle shadows and depth
- Smooth hover animations
- Collapsible sidebar
- RAG with FAISS + Groq
- Streaming responses
- Source document display
"""

import os
import time
import streamlit as st
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

# Import your RAG utilities
from rag_utils import load_and_chunk, build_store, init_groq, prompt_tpl


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

if "history" not in st.session_state:
    st.session_state.history = []

if "last_docs" not in st.session_state:
    st.session_state.last_docs = []


# ==============================================================================
# ENHANCED THEME CONFIGURATION
# ==============================================================================
THEMES = {
    "dark": {
        "bg": "#0f0f0f",
        "bg_gradient": "radial-gradient(ellipse at top, #1a1a2e 0%, #0f0f0f 50%)",
        "sidebar_bg": "#161616",
        "sidebar_border": "#2a2a2a",
        "card_bg": "#1e1e1e",
        "card_hover": "#252525",
        "input_bg": "#1e1e1e",
        "input_focus": "#252525",
        "border": "#2a2a2a",
        "border_hover": "#3a3a3a",
        "text": "#f5f5f5",
        "text_secondary": "#a0a0a0",
        "text_muted": "#666666",
        "accent": "#00d4aa",
        "accent_hover": "#00f5c4",
        "accent_subtle": "rgba(0, 212, 170, 0.1)",
        "user_avatar_bg": "linear-gradient(135deg, #00d4aa 0%, #00a085 100%)",
        "bot_avatar_bg": "linear-gradient(135deg, #2a2a3e 0%, #1a1a2e 100%)",
        "shadow_sm": "0 2px 8px rgba(0, 0, 0, 0.3)",
        "shadow_md": "0 4px 16px rgba(0, 0, 0, 0.4)",
        "shadow_lg": "0 8px 32px rgba(0, 0, 0, 0.5)",
        "glow": "0 0 20px rgba(0, 212, 170, 0.15)",
    },
    "light": {
        "bg": "#fafbfc",
        "bg_gradient": "linear-gradient(180deg, #ffffff 0%, #f0f4f8 100%)",
        "sidebar_bg": "#ffffff",
        "sidebar_border": "#e8edf3",
        "card_bg": "#ffffff",
        "card_hover": "#f8fafc",
        "input_bg": "#ffffff",
        "input_focus": "#f8fafc",
        "border": "#e2e8f0",
        "border_hover": "#cbd5e1",
        "text": "#1e293b",
        "text_secondary": "#64748b",
        "text_muted": "#94a3b8",
        "accent": "#0891b2",
        "accent_hover": "#06b6d4",
        "accent_subtle": "rgba(8, 145, 178, 0.08)",
        "user_avatar_bg": "linear-gradient(135deg, #0891b2 0%, #0e7490 100%)",
        "bot_avatar_bg": "linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%)",
        "shadow_sm": "0 1px 3px rgba(0, 0, 0, 0.08), 0 1px 2px rgba(0, 0, 0, 0.04)",
        "shadow_md": "0 4px 12px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.04)",
        "shadow_lg": "0 8px 24px rgba(0, 0, 0, 0.1), 0 4px 8px rgba(0, 0, 0, 0.04)",
        "glow": "0 0 20px rgba(8, 145, 178, 0.1)",
    }
}

# Get current theme colors
theme = THEMES[st.session_state.theme]
is_dark = st.session_state.theme == "dark"


# ==============================================================================
# ENHANCED CSS STYLING
# ==============================================================================
st.markdown(f"""
<style>
    /* ========== FONTS ========== */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    * {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    code, pre, .stCode {{
        font-family: 'JetBrains Mono', monospace !important;
    }}
    
    /* ========== MAIN BACKGROUND ========== */
    .stApp {{
        background: {theme["bg_gradient"]};
        background-attachment: fixed;
    }}
    
    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {{
        background: {theme["sidebar_bg"]};
        border-right: 1px solid {theme["sidebar_border"]};
        box-shadow: {theme["shadow_md"]};
    }}
    
    [data-testid="stSidebar"] > div:first-child {{
        padding: 1.5rem 1rem;
    }}
    
    /* Sidebar text */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {{
        color: {theme["text_secondary"]};
    }}
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: {theme["text"]} !important;
    }}
    
    /* ========== SIDEBAR TITLE ========== */
    .sidebar-title {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 4px;
    }}
    
    .sidebar-title-icon {{
        font-size: 24px;
    }}
    
    .sidebar-title-text {{
        color: {theme["text"]};
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.3px;
    }}
    
    .sidebar-subtitle {{
        color: {theme["text_muted"]};
        font-size: 13px;
        margin-left: 34px;
        margin-top: -2px;
    }}
    
    /* ========== SECTION LABELS ========== */
    .section-label {{
        color: {theme["text_muted"]};
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 24px 0 12px 0;
        padding-left: 2px;
    }}
    
    /* ========== STATUS INDICATORS ========== */
    .status-container {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 0;
        padding: 10px 14px;
        background: {theme["accent_subtle"]};
        border-radius: 10px;
        border: 1px solid {"rgba(0, 212, 170, 0.2)" if is_dark else "rgba(8, 145, 178, 0.15)"};
    }}
    
    .status-container.offline {{
        background: {"rgba(239, 68, 68, 0.1)" if is_dark else "rgba(239, 68, 68, 0.06)"};
        border-color: {"rgba(239, 68, 68, 0.2)" if is_dark else "rgba(239, 68, 68, 0.15)"};
    }}
    
    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
        animation: pulse 2s infinite;
    }}
    
    .status-dot.online {{
        background-color: {theme["accent"]};
        box-shadow: 0 0 8px {theme["accent"]};
    }}
    
    .status-dot.offline {{
        background-color: #ef4444;
        animation: none;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    
    .status-text {{
        color: {theme["text_secondary"]};
        font-size: 13px;
        font-weight: 500;
    }}
    
    /* ========== DIVIDER ========== */
    .divider {{
        height: 1px;
        background: {"linear-gradient(90deg, transparent, " + theme["border"] + ", transparent)"};
        margin: 20px 0;
        border: none;
    }}
    
    /* ========== HEADER ========== */
    .chat-header {{
        text-align: center;
        padding: 32px 20px;
        margin-bottom: 24px;
        background: {theme["card_bg"]};
        border-radius: 16px;
        border: 1px solid {theme["border"]};
        box-shadow: {theme["shadow_sm"]};
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }}
    
    .chat-header-title {{
        color: {theme["text"]};
        font-size: 22px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }}
    
    .chat-header-subtitle {{
        color: {theme["text_secondary"]};
        font-size: 14px;
        margin: 8px 0 0 0;
        font-weight: 500;
    }}
    
    /* ========== WELCOME SCREEN ========== */
    .welcome-container {{
        text-align: center;
        padding: 80px 24px;
        max-width: 640px;
        margin: 0 auto;
    }}
    
    .welcome-icon {{
        font-size: 56px;
        margin-bottom: 24px;
        display: block;
    }}
    
    .welcome-title {{
        color: {theme["text"]};
        font-size: 32px;
        font-weight: 700;
        margin: 0 0 16px 0;
        letter-spacing: -0.5px;
        line-height: 1.2;
    }}
    
    .welcome-subtitle {{
        color: {theme["text_secondary"]};
        font-size: 16px;
        margin: 0;
        line-height: 1.6;
        font-weight: 500;
    }}
    
    .welcome-link {{
        color: {theme["accent"]};
        text-decoration: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }}
    
    .welcome-link:hover {{
        color: {theme["accent_hover"]};
        text-decoration: underline;
    }}
    
    /* ========== CHAT MESSAGES ========== */
    .messages-container {{
        max-width: 800px;
        margin: 0 auto;
        padding: 0 16px;
    }}
    
    .message-row {{
        display: flex;
        gap: 16px;
        padding: 28px 24px;
        margin: 12px 0;
        background: {theme["card_bg"]};
        border-radius: 16px;
        border: 1px solid {theme["border"]};
        box-shadow: {theme["shadow_sm"]};
        transition: all 0.2s ease;
    }}
    
    .message-row:hover {{
        box-shadow: {theme["shadow_md"]};
        border-color: {theme["border_hover"]};
    }}
    
    .message-avatar {{
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
        box-shadow: {theme["shadow_sm"]};
    }}
    
    .message-avatar.user {{
        background: {theme["user_avatar_bg"]};
        color: white;
    }}
    
    .message-avatar.assistant {{
        background: {theme["bot_avatar_bg"]};
        border: 1px solid {theme["border"]};
    }}
    
    .message-content {{
        flex: 1;
        color: {theme["text"]};
        font-size: 15px;
        line-height: 1.75;
        padding-top: 8px;
    }}
    
    .message-meta {{
        color: {theme["text_muted"]};
        font-size: 12px;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid {theme["border"]};
        font-weight: 500;
    }}
    
    /* ========== SUGGESTION BUTTONS ========== */
    .suggestions-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin-top: 40px;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }}
    
    /* ========== INPUT FIELDS ========== */
    .stTextInput > div > div > input {{
        background: {theme["input_bg"]} !important;
        border: 2px solid {theme["border"]} !important;
        border-radius: 12px !important;
        color: {theme["text"]} !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        box-shadow: {theme["shadow_sm"]} !important;
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: {theme["accent"]} !important;
        box-shadow: {theme["glow"]} !important;
        background: {theme["input_focus"]} !important;
    }}
    
    .stTextInput > div > div > input::placeholder {{
        color: {theme["text_muted"]} !important;
    }}
    
    /* ========== SELECT BOX ========== */
    .stSelectbox > div > div {{
        background: {theme["input_bg"]} !important;
        border: 2px solid {theme["border"]} !important;
        border-radius: 12px !important;
        box-shadow: {theme["shadow_sm"]} !important;
    }}
    
    .stSelectbox > div > div:hover {{
        border-color: {theme["border_hover"]} !important;
    }}
    
    .stSelectbox > div > div > div {{
        color: {theme["text"]} !important;
        font-weight: 500 !important;
    }}
    
    .stSelectbox label {{
        color: {theme["text_secondary"]} !important;
        font-weight: 600 !important;
    }}
    
    /* ========== SLIDERS ========== */
    .stSlider label {{
        color: {theme["text_secondary"]} !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }}
    
    .stSlider > div > div > div > div {{
        background: {theme["border"]} !important;
    }}
    
    .stSlider > div > div > div > div > div {{
        background: {theme["accent"]} !important;
    }}
    
    /* ========== BUTTONS ========== */
    .stButton > button {{
        background: {theme["card_bg"]} !important;
        color: {theme["text"]} !important;
        border: 2px solid {theme["border"]} !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
        box-shadow: {theme["shadow_sm"]} !important;
    }}
    
    .stButton > button:hover {{
        background: {theme["card_hover"]} !important;
        border-color: {theme["accent"]} !important;
        box-shadow: {theme["shadow_md"]} !important;
        transform: translateY(-1px) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}
    
    /* Primary action button style */
    .primary-btn > button {{
        background: {theme["user_avatar_bg"]} !important;
        color: white !important;
        border: none !important;
    }}
    
    .primary-btn > button:hover {{
        filter: brightness(1.1) !important;
    }}
    
    /* ========== RADIO BUTTONS (Theme Toggle) ========== */
    .stRadio > div {{
        gap: 8px !important;
        background: {theme["card_bg"]};
        padding: 8px;
        border-radius: 12px;
        border: 1px solid {theme["border"]};
    }}
    
    .stRadio label {{
        color: {theme["text"]} !important;
        font-weight: 500 !important;
    }}
    
    .stRadio > div > label > div:first-child {{
        background-color: {theme["accent"]} !important;
    }}
    
    /* ========== CHAT INPUT ========== */
    [data-testid="stChatInput"] > div {{
        background: {theme["input_bg"]} !important;
        border: 2px solid {theme["border"]} !important;
        border-radius: 16px !important;
        box-shadow: {theme["shadow_md"]} !important;
        transition: all 0.2s ease !important;
    }}
    
    [data-testid="stChatInput"] > div:focus-within {{
        border-color: {theme["accent"]} !important;
        box-shadow: {theme["glow"]}, {theme["shadow_md"]} !important;
    }}
    
    [data-testid="stChatInput"] textarea {{
        color: {theme["text"]} !important;
        font-size: 15px !important;
        font-weight: 500 !important;
    }}
    
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {theme["text_muted"]} !important;
    }}
    
    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {{
        background: {theme["card_bg"]} !important;
        border: 2px solid {theme["border"]} !important;
        border-radius: 12px !important;
        color: {theme["text_secondary"]} !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        padding: 14px 16px !important;
        transition: all 0.2s ease !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        color: {theme["text"]} !important;
        border-color: {theme["accent"]} !important;
        box-shadow: {theme["shadow_sm"]} !important;
    }}
    
    .streamlit-expanderContent {{
        background: {theme["sidebar_bg"]} !important;
        border: 2px solid {theme["border"]} !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 16px !important;
    }}
    
    /* ========== SOURCE DOCUMENTS ========== */
    .source-card {{
        background: {theme["card_bg"]};
        border: 2px solid {theme["border"]};
        border-radius: 12px;
        padding: 18px;
        margin: 12px 0;
        transition: all 0.2s ease;
        box-shadow: {theme["shadow_sm"]};
    }}
    
    .source-card:hover {{
        border-color: {theme["accent"]};
        box-shadow: {theme["shadow_md"]};
    }}
    
    .source-card-title {{
        color: {theme["accent"]};
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .source-card-title::before {{
        content: '📄';
        font-size: 14px;
    }}
    
    .source-card-content {{
        color: {theme["text"]};
        font-size: 14px;
        line-height: 1.7;
    }}
    
    /* ========== ALERTS ========== */
    .stAlert {{
        background: {theme["card_bg"]} !important;
        border: 2px solid {theme["border"]} !important;
        border-radius: 12px !important;
        box-shadow: {theme["shadow_sm"]} !important;
    }}
    
    /* ========== CHECKBOX ========== */
    .stCheckbox label {{
        color: {theme["text"]} !important;
        font-weight: 500 !important;
    }}
    
    .stCheckbox > label > div:first-child {{
        background-color: {theme["card_bg"]} !important;
        border: 2px solid {theme["border"]} !important;
        border-radius: 6px !important;
    }}
    
    /* ========== HIDE STREAMLIT DEFAULTS ========== */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    
    /* ========== CUSTOM SCROLLBAR ========== */
    ::-webkit-scrollbar {{
        width: 10px;
        height: 10px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {theme["bg"]};
        border-radius: 5px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {theme["border"]};
        border-radius: 5px;
        border: 2px solid {theme["bg"]};
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {theme["text_muted"]};
    }}
    
    /* ========== API LINK ========== */
    .api-link {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: {theme["accent"]};
        font-size: 13px;
        font-weight: 600;
        text-decoration: none;
        padding: 8px 14px;
        background: {theme["accent_subtle"]};
        border-radius: 8px;
        border: 1px solid {"rgba(0, 212, 170, 0.2)" if is_dark else "rgba(8, 145, 178, 0.15)"};
        transition: all 0.2s ease;
        margin-top: 8px;
    }}
    
    .api-link:hover {{
        background: {"rgba(0, 212, 170, 0.2)" if is_dark else "rgba(8, 145, 178, 0.12)"};
        transform: translateX(4px);
    }}
    
    /* ========== FOOTER ========== */
    .sidebar-footer {{
        text-align: center;
        padding: 16px;
        margin-top: 20px;
        background: {theme["accent_subtle"]};
        border-radius: 12px;
        border: 1px solid {"rgba(0, 212, 170, 0.15)" if is_dark else "rgba(8, 145, 178, 0.1)"};
    }}
    
    .sidebar-footer p {{
        color: {theme["text_secondary"]};
        font-size: 12px;
        margin: 0;
        font-weight: 500;
    }}
    
    .sidebar-footer a {{
        color: {theme["accent"]};
        text-decoration: none;
        font-weight: 600;
    }}
    
    .sidebar-footer a:hover {{
        text-decoration: underline;
    }}
    
    /* ========== DOC COUNT BADGE ========== */
    .doc-badge {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        background: {theme["accent_subtle"]};
        border-radius: 10px;
        border: 1px solid {"rgba(0, 212, 170, 0.2)" if is_dark else "rgba(8, 145, 178, 0.15)"};
        margin-top: 8px;
    }}
    
    .doc-badge-icon {{
        font-size: 16px;
    }}
    
    .doc-badge-text {{
        color: {theme["text_secondary"]};
        font-size: 13px;
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    # ----- Title -----
    st.markdown(f"""
        <div class="sidebar-title">
            <span class="sidebar-title-icon"></span>
            <span class="sidebar-title-text">RAG Chatbot</span>
        </div>
        <p class="sidebar-subtitle">Chat with your documents</p>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ----- Theme Toggle -----
    st.markdown('<p class="section-label">Appearance</p>', unsafe_allow_html=True)
    
    theme_selection = st.radio(
        "Select theme",
        options=[" Dark", " Light"],
        index=0 if st.session_state.theme == "dark" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    # Update theme if changed
    new_theme = "dark" if theme_selection == " Dark" else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ----- API Key -----
    st.markdown('<p class="section-label">API Configuration</p>', unsafe_allow_html=True)
    
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="Enter your Groq API key",
        label_visibility="collapsed"
    )
    
    if api_key:
        st.markdown(f'''
            <div class="status-container">
                <div class="status-dot online"></div>
                <span class="status-text">API Connected</span>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div class="status-container offline">
                <div class="status-dot offline"></div>
                <span class="status-text">Not connected</span>
            </div>
        ''', unsafe_allow_html=True)
        st.markdown(f"<a href='https://console.groq.com' target='_blank' class='api-link'> Get free API key →</a>", unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ----- Model Selection -----
    st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
    
    model_options = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]
    
    model_name = st.selectbox(
        "Select Model",
        options=model_options,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ----- Documents Folder -----
    st.markdown('<p class="section-label">Documents</p>', unsafe_allow_html=True)
    
    data_folder = st.text_input(
        "Documents Folder",
        value="data/sample_docs",
        placeholder="Path to documents folder",
        label_visibility="collapsed"
    )
    
    # Show document count
    if Path(data_folder).exists():
        txt_files = list(Path(data_folder).glob("*.txt"))
        doc_count = len(txt_files)
        st.markdown(f'''
            <div class="doc-badge">
                <span class="doc-badge-icon"></span>
                <span class="doc-badge-text">{doc_count} document{"s" if doc_count != 1 else ""} loaded</span>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div class="status-container offline">
                <div class="status-dot offline"></div>
                <span class="status-text">Folder not found</span>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ----- RAG Parameters -----
    st.markdown('<p class="section-label">RAG Settings</p>', unsafe_allow_html=True)
    
    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=1000,
        value=500,
        step=50
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=200,
        value=50,
        step=10
    )
    
    k_docs = st.slider(
        "Documents to Retrieve (k)",
        min_value=1,
        max_value=10,
        value=3
    )
    
    # ----- Advanced Options -----
    with st.expander(" Advanced Options"):
        show_sources = st.checkbox("Show source documents", value=True)
        show_debug = st.checkbox("Show debug info", value=False)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ----- Action Buttons -----
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Reload", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button(" Clear", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_docs = []
            st.rerun()
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # ----- Footer -----
    st.markdown(f"""
        <div class="sidebar-footer">
            <p>Built with ❤️ by <a href='https://github.com/MusaedAl-Fareh' target='_blank'>Musaed Al-Fareh</a></p>
        </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# MAIN CONTENT AREA
# ==============================================================================

# ----- Header -----
st.markdown(f"""
    <div class="chat-header">
        <p class="chat-header-title">💬 RAG Chatbot</p>
        <p class="chat-header-subtitle">Ask questions about your documents</p>
    </div>
""", unsafe_allow_html=True)


# ==============================================================================
# VALIDATION
# ==============================================================================

# Check API key
if not api_key:
    st.markdown(f"""
        <div class="welcome-container">
            <span class="welcome-icon"></span>
            <h1 class="welcome-title">Welcome to RAG Chatbot</h1>
            <p class="welcome-subtitle">
                Add your Groq API key in the sidebar to start chatting with your documents.
                <br><br>
                <a href="https://console.groq.com" target="_blank" class="welcome-link">
                    🔑 Get a free API key →
                </a>
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# Check data folder
if not Path(data_folder).exists():
    st.error(f" Folder not found: `{data_folder}`")
    st.info(" Create the folder and add some .txt files, or change the path in the sidebar.")
    st.stop()

# Check for txt files
txt_files = list(Path(data_folder).glob("*.txt"))
if not txt_files:
    st.error(f" No .txt files found in `{data_folder}`")
    st.info(" Add some .txt documents to the folder.")
    st.stop()


# ==============================================================================
# LOAD RESOURCES (CACHED)
# ==============================================================================

@st.cache_resource(show_spinner=" Loading and indexing documents...")
def get_vector_store(_folder, _chunk_size, _chunk_overlap):
    """Load documents and create vector store."""
    chunks, metas = load_and_chunk(_folder, _chunk_size, _chunk_overlap)
    return build_store(chunks, metas)

@st.cache_resource(show_spinner=" Connecting to Groq...")
def get_llm(_api_key, _model):
    """Initialize the LLM."""
    return init_groq(_api_key, _model)

# Load vector store
store = get_vector_store(data_folder, chunk_size, chunk_overlap)

# Initialize LLM
llm = get_llm(api_key, model_name)


# ==============================================================================
# CONSTANTS
# ==============================================================================
MAX_HISTORY_TURNS = 10


# ==============================================================================
# WELCOME SCREEN (when no messages)
# ==============================================================================

if not st.session_state.history:
    st.markdown(f"""
        <div class="welcome-container">
            <span class="welcome-icon">✨</span>
            <h1 class="welcome-title">How can I help you?</h1>
            <p class="welcome-subtitle">Ask me anything about your documents</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Suggestion buttons
    st.markdown('<div class="suggestions-grid">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    suggestions = [
        "📋 What services are available?",
        "💰 Tell me about the prices",
        "🎯 What activities can I do?"
    ]
    
    for col, suggestion_text in zip([col1, col2, col3], suggestions):
        with col:
            if st.button(suggestion_text, use_container_width=True, key=f"suggestion_{suggestion_text}"):
                clean_text = suggestion_text.split(" ", 1)[1] if " " in suggestion_text else suggestion_text
                st.session_state.history.append(HumanMessage(content=clean_text))
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# DISPLAY CHAT HISTORY
# ==============================================================================

st.markdown('<div class="messages-container">', unsafe_allow_html=True)

for message in st.session_state.history:
    if isinstance(message, HumanMessage):
        st.markdown(f"""
            <div class="message-row">
                <div class="message-avatar user">👤</div>
                <div class="message-content">{message.content}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="message-row">
                <div class="message-avatar assistant">🤖</div>
                <div class="message-content">{message.content}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# CHAT INPUT & RESPONSE
# ==============================================================================

user_input = st.chat_input("Message RAG Chatbot...")

if user_input:
    # Display user message immediately
    st.markdown(f"""
        <div class="messages-container">
            <div class="message-row">
                <div class="message-avatar user">👤</div>
                <div class="message-content">{user_input}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Add to history
    st.session_state.history.append(HumanMessage(content=user_input))
    
    # Trim history if too long
    if len(st.session_state.history) > MAX_HISTORY_TURNS * 2:
        st.session_state.history = st.session_state.history[-MAX_HISTORY_TURNS * 2:]
    
    # ----- Retrieve relevant documents -----
    try:
        docs = store.similarity_search(user_input, k=k_docs)
        context = "\n\n".join(doc.page_content for doc in docs)
        st.session_state.last_docs = docs
    except Exception as e:
        st.error(f" Retrieval error: {e}")
        context = ""
        st.session_state.last_docs = []
    
    # ----- Build prompt -----
    full_prompt = prompt_tpl.format(context=context, question=user_input)
    
    # Show debug info if enabled
    if show_debug:
        with st.expander("🔍 Debug: Full Prompt"):
            st.code(full_prompt, language="text")
    
    # ----- Stream response -----
    response_placeholder = st.empty()
    collected_response = ""
    start_time = time.time()
    
    try:
        for chunk in llm.stream([HumanMessage(content=full_prompt)]):
            collected_response += chunk.content or ""
            response_placeholder.markdown(f"""
                <div class="messages-container">
                    <div class="message-row">
                        <div class="message-avatar assistant">🤖</div>
                        <div class="message-content">{collected_response}▌</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        # Calculate response time
        elapsed_time = time.time() - start_time
        
        # Final response with metadata
        response_placeholder.markdown(f"""
            <div class="messages-container">
                <div class="message-row">
                    <div class="message-avatar assistant">🤖</div>
                    <div class="message-content">
                        {collected_response}
                        <div class="message-meta">⚡ {elapsed_time:.2f}s</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f" API Error: {e}")
        st.info(" Check your API key or try a different model.")
        collected_response = "I encountered an error. Please check your API key or try again."
        response_placeholder.empty()
    
    # Add assistant response to history
    st.session_state.history.append(AIMessage(content=collected_response))


# ==============================================================================
# SOURCE DOCUMENTS
# ==============================================================================

if show_sources and st.session_state.last_docs:
    with st.expander(" Retrieved Source Documents"):
        for idx, doc in enumerate(st.session_state.last_docs, 1):
            source_name = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:400]
            if len(doc.page_content) > 400:
                content_preview += "..."
            
            st.markdown(f"""
                <div class="source-card">
                    <div class="source-card-title">{idx}. {source_name}</div>
                    <div class="source-card-content">{content_preview}</div>
                </div>
            """, unsafe_allow_html=True)