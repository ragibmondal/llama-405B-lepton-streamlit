import streamlit as st
import os
import base64
import openai
from typing import List, Optional
import tempfile
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Voice Chat Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
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
        /* Message Bubbles */
        .message-container {
            display: flex;
            margin-bottom: 10px;
            animation: fadeIn 0.5s ease-in;
        }
        .message-bubble {
            padding: 12px 20px;
            border-radius: 20px;
            max-width: 70%;
            margin: 5px;
            position: relative;
            word-wrap: break-word;
        }
        .user-message {
            background-color: #DCF8C6;
            margin-left: auto;
            border-bottom-right-radius: 5px;
        }
        .ai-message {
            background-color: white;
            margin-right: auto;
            border-bottom-left-radius: 5px;
        }
        .message-time {
            font-size: 0.7em;
            color: #666;
            margin-top: 5px;
        }
        .audio-player {
            margin-top: 10px;
            width: 100%;
        }
        .recording-indicator {
            color: #FF4B4B;
            animation: pulse 1.5s infinite;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .chat-container {
            height: 600px;
            overflow-y: auto;
            padding: 20px;
            background-color: #E8E8E8;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        /* Hide Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* Recorder Button Styling */
        .recorder-container {
            display: flex;
            justify-content: center;
            padding: 10px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

def initialize_lepton_client():
    api_token = os.getenv('LEPTON_API_TOKEN')
    if not api_token:
        raise ValueError("LEPTON_API_TOKEN not found in environment variables")
    return openai.OpenAI(
        base_url="https://llama3-1-405b.lepton.run/api/v1/",
        api_key=api_token
    )

def process_audio_file(audio_file) -> str:
    audio_bytes = audio_file.read()
    return base64.b64encode(audio_bytes).decode()

def display_message(content, is_user=True, audio_path=None):
    message_class = "user-message" if is_user else "ai-message"
    current_time = datetime.now().strftime("%H:%M")
    
    st.markdown(f"""
        <div class="message-container">
            <div class="message-bubble {message_class}">
                {content}
                <div class="message-time">{current_time}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if audio_path:
        with open(audio_path, 'rb') as audio_file:
            st.audio(audio_file.read(), format='audio/mp3')

def generate_response(client, prompt: str, audio_data: Optional[str] = None) -> tuple[str, List[str]]:
    messages = []
    if audio_data:
        messages.append({"role": "user", "content": [{"type": "audio", "data": audio_data}]})
    else:
        messages.append({"role": "user", "content": prompt})

    extra_body = {
        "tts_audio_format": "mp3",
        "tts_audio_bitrate": 16,
        "require_audio": True,
        "tts_preset_id": "jessica",
    }

    completion = client.chat.completions.create(
        model="llama3.1-405b",
        messages=messages,
        max_tokens=128,
        stream=True,
        extra_body=extra_body
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
            response_placeholder.markdown(full_response)
            
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

def main():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.title("🎙️ Voice Chat Assistant")
    
    try:
        client = initialize_lepton_client()
    except Exception as e:
        st.error(f"Failed to connect: {str(e)}")
        return

    # Chat container
    chat_container = st.container()
    
    # Input section at the bottom
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_input("Type a message:", key="text_input")
        
        with col2:
            if st.button("Send", key="send_button"):
                if user_input:
                    # Add user message to chat
                    st.session_state.messages.append(("user", user_input, None))
                    
                    # Generate AI response
                    response_text, audio_chunks = generate_response(client, user_input)
                    audio_path = save_audio(audio_chunks)
                    
                    # Add AI response to chat
                    st.session_state.messages.append(("ai", response_text, audio_path))
                    st.experimental_rerun()

    # Voice recording section
    st.markdown("<div class='recorder-container'>", unsafe_allow_html=True)
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        sample_rate=44100,
        recording_color="#FF4B4B",
        neutral_color="#6B7280",
        icon_name="microphone",
        icon_size="2x"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if audio_bytes:
        # Save the recorded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            
        # Process audio automatically
        with open(tmp_file.name, 'rb') as audio_file:
            audio_data = process_audio_file(audio_file)
        
        # Add user audio message to chat
        st.session_state.messages.append(("user", "🎤 Voice message", tmp_file.name))
        
        # Generate AI response
        response_text, audio_chunks = generate_response(client, "", audio_data=audio_data)
        audio_path = save_audio(audio_chunks)
        
        # Add AI response to chat
        st.session_state.messages.append(("ai", response_text, audio_path))
        st.experimental_rerun()

    # Display chat messages
    with chat_container:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for msg_type, content, audio_path in st.session_state.messages:
            display_message(content, is_user=(msg_type == "user"), audio_path=audio_path)
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()