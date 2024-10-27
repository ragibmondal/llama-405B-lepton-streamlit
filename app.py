import streamlit as st
import os
import base64
import openai
from typing import List, Optional
import tempfile
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
import time
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Voice & Text Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with fixed bottom input
st.markdown("""
    <style>
        /* Main container styles */
        .main {
            background-color: #f5f5f5;
        }
        
        /* Chat container styles */
        .chat-container {
            height: calc(100vh - 300px);
            overflow-y: auto;
            padding: 20px;
            margin-bottom: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Message styles */
        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 10px;
            max-width: 80%;
        }
        
        .user-message {
            background-color: #DCF8C6;
            margin-left: auto;
            margin-right: 20px;
        }
        
        .ai-message {
            background-color: #E8E8E8;
            margin-right: auto;
            margin-left: 20px;
        }
        
        /* Input container styles */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: white;
            padding: 20px;
            box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
            z-index: 1000;
        }
        
        /* Button styles */
        .stButton>button {
            border-radius: 20px;
            height: 3em;
            background-color: #FF4B4B;
            color: white;
            border: none;
        }
        
        .stButton>button:hover {
            background-color: #FF2B2B;
        }
        
        /* Audio recorder styles */
        .recorder-button {
            background-color: #FF4B4B !important;
            color: white !important;
            border-radius: 20px !important;
            padding: 10px 20px !important;
        }
        
        /* Status box styles */
        .status-box {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .success-box {
            background-color: #D1F2EB;
            border: 1px solid #48C9B0;
        }
        
        .error-box {
            background-color: #FADBD8;
            border: 1px solid #E74C3C;
        }
        
        .info-box {
            background-color: #D4E6F1;
            border: 1px solid #3498DB;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Timestamp style */
        .timestamp {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        
        /* Audio player styles */
        audio {
            width: 100%;
            margin-top: 10px;
        }
        
        /* Loading animation */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

def initialize_lepton_client():
    api_token = os.getenv('LEPTON_API_TOKEN')
    if not api_token:
        raise ValueError("LEPTON_API_TOKEN not found in environment variables")
        
    return openai.OpenAI(
        base_url="https://llama3-1-405b.lepton.run/api/v1/",
        api_key=api_token
    )

def add_message(content, role, audio_path=None):
    st.session_state.messages.append({
        'content': content,
        'role': role,
        'timestamp': datetime.now().strftime("%H:%M"),
        'audio_path': audio_path
    })

def render_message(message):
    role_class = "user-message" if message['role'] == 'user' else 'ai-message'
    
    message_html = f"""
        <div class="message {role_class}">
            <strong>{'You' if message['role'] == 'user' else '🤖 AI'}</strong>
            <p>{message['content']}</p>
            <div class="timestamp">{message['timestamp']}</div>
        </div>
    """
    st.markdown(message_html, unsafe_allow_html=True)
    
    if message.get('audio_path'):
        with open(message['audio_path'], 'rb') as audio_file:
            st.audio(audio_file.read(), format='audio/mp3')

def render_chat_history():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        render_message(message)
    st.markdown('</div>', unsafe_allow_html=True)

def process_audio_file(audio_file) -> str:
    audio_bytes = audio_file.read()
    return base64.b64encode(audio_bytes).decode()

def generate_response(client, 
                     prompt: str, 
                     audio_data: Optional[str] = None,
                     generate_audio: bool = False) -> tuple[str, List[str]]:
    messages = []
    if audio_data:
        messages.append({"role": "user", "content": [{"type": "audio", "data": audio_data}]})
    else:
        messages.append({"role": "user", "content": prompt})

    extra_body = {}
    if generate_audio:
        extra_body.update({
            "tts_audio_format": "mp3",
            "tts_audio_bitrate": 16,
            "require_audio": True,
            "tts_preset_id": st.session_state.get('voice_preset', 'jessica'),
        })

    completion = client.chat.completions.create(
        model="llama3.1-405b",
        messages=messages,
        max_tokens=st.session_state.get('max_tokens', 128),
        stream=True,
        extra_body=extra_body if extra_body else None
    )

    full_response = ""
    audio_chunks = []

    response_placeholder = st.empty()
    
    for chunk in completion:
        if not chunk.choices:
            continue
        
        content = chunk.choices[0].delta.content
        audio = getattr(chunk.choices[0], 'audio', [])
        
        if content:
            full_response += content
            response_placeholder.markdown(f"```\n{full_response}\n```")
            
        if audio:
            audio_chunks.extend(audio)

    return full_response, audio_chunks

def save_audio(audio_chunks: List[str]) -> str:
    if not audio_chunks:
        return ""
        
    audio_data = b''.join([base64.b64decode(chunk) for chunk in audio_chunks])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(audio_data)
        return tmp_file.name

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        st.markdown("---")
        
        st.subheader("Voice Settings")
        st.session_state.voice_preset = st.selectbox(
            "Voice Preset",
            ["jessica", "josh", "emma", "michael"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("Model Settings")
        st.session_state.max_tokens = st.slider(
            "Max Response Length",
            min_value=50,
            max_value=500,
            value=128,
            step=10
        )
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()
        
        st.markdown("---")
        st.markdown("""
        ### About
        This app uses Lepton AI to:
        - Convert speech to text
        - Generate AI responses
        - Convert text to speech
        """)

def render_input_container():
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_input("Type your message:", key="text_input")
            generate_audio = st.checkbox("Enable voice response", value=True)
        
        with col2:
            audio_bytes = audio_recorder(
                pause_threshold=2.0,
                sample_rate=44100,
                text="",
                recording_color="#FF4B4B",
                neutral_color="#6B7280",
                icon_name="microphone",
                icon_size="2x"
            )
            
            if st.button("Send", key="send_button"):
                process_input(user_input, audio_bytes, generate_audio)
        
        st.markdown('</div>', unsafe_allow_html=True)

def process_input(text_input, audio_bytes, generate_audio):
    client = initialize_lepton_client()
    
    if text_input:
        add_message(text_input, 'user')
        
        with st.spinner("🤖 AI is thinking..."):
            response_text, audio_chunks = generate_response(
                client, 
                text_input,
                generate_audio=generate_audio
            )
            
            audio_path = None
            if generate_audio and audio_chunks:
                audio_path = save_audio(audio_chunks)
            
            add_message(response_text, 'assistant', audio_path)
    
    elif audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            add_message("🎤 [Voice Message]", 'user')
            
            with st.spinner("🎯 Processing your voice..."):
                with open(tmp_file.name, 'rb') as audio_file:
                    audio_data = process_audio_file(audio_file)
                
                response_text, audio_chunks = generate_response(
                    client,
                    "",
                    audio_data=audio_data,
                    generate_audio=generate_audio
                )
                
                audio_path = None
                if generate_audio and audio_chunks:
                    audio_path = save_audio(audio_chunks)
                
                add_message(response_text, 'assistant', audio_path)
            
            os.unlink(tmp_file.name)
    
    st.experimental_rerun()

def main():
    render_sidebar()
    
    st.title("🎙️ AI Voice & Text Assistant")
    
    try:
        client = initialize_lepton_client()
    except Exception as e:
        st.error(f"Failed to connect: {str(e)}")
        return
    
    render_chat_history()
    render_input_container()

if __name__ == "__main__":
    main()