import streamlit as st
import os
import base64
import openai
from typing import List, Optional, Dict
import tempfile
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like interface
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #ffffff;
        }
        
        /* Chat container */
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        /* Message styling */
        .user-message {
            background-color: #f7f7f8;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        
        .assistant-message {
            background-color: #ffffff;
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        
        /* Input box styling */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #ffffff;
            padding: 1rem;
            border-top: 1px solid #e5e5e5;
        }
        
        .stTextInput {
            border: 1px solid #e5e5e5;
            border-radius: 0.5rem;
            padding: 0.5rem;
        }
        
        /* Audio recorder styling */
        .recorder-container {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        
        .audio-recorder {
            border-radius: 50%;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: transparent;
            border: 1px solid #e5e5e5;
            cursor: pointer;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #202123;
        }
        
        .css-1d391kg .stMarkdown {
            color: #ffffff;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #19c37d;
            color: white;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            border: none;
            transition: background-color 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #0f9c63;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Add spacing for fixed input box */
        .main-content {
            margin-bottom: 100px;
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
    """Convert audio file to base64 string"""
    audio_bytes = audio_file.read()
    return base64.b64encode(audio_bytes).decode()

def generate_response(client, 
                     messages: List[Dict],
                     audio_data: Optional[str] = None,
                     generate_audio: bool = False) -> tuple[str, List[str]]:
    """Generate response from LLM with optional audio input/output"""
    
    if audio_data:
        messages.append({"role": "user", "content": [{"type": "audio", "data": audio_data}]})

    extra_body = {}
    if generate_audio:
        extra_body.update({
            "tts_audio_format": "mp3",
            "tts_audio_bitrate": 16,
            "require_audio": True,
            "tts_preset_id": st.session_state.voice_preset,
        })

    completion = client.chat.completions.create(
        model="llama3.1-405b",
        messages=messages,
        max_tokens=st.session_state.max_tokens,
        stream=True,
        extra_body=extra_body if extra_body else None
    )

    full_response = ""
    audio_chunks = []

    message_placeholder = st.empty()
    
    for chunk in completion:
        if not chunk.choices:
            continue
        
        content = chunk.choices[0].delta.content
        audio = getattr(chunk.choices[0], 'audio', [])
        
        if content:
            full_response += content
            message_placeholder.markdown(f"{full_response}", unsafe_allow_html=True)
            
        if audio:
            audio_chunks.extend(audio)

    return full_response, audio_chunks

def save_audio(audio_chunks: List[str]) -> str:
    """Save audio chunks to a temporary file and return the path"""
    if not audio_chunks:
        return ""
        
    audio_data = b''.join([base64.b64decode(chunk) for chunk in audio_chunks])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(audio_data)
        return tmp_file.name

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'voice_preset' not in st.session_state:
        st.session_state.voice_preset = "jessica"
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 128
    if 'enable_voice' not in st.session_state:
        st.session_state.enable_voice = False

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.session_state.voice_preset = st.selectbox(
            "Voice Preset",
            ["jessica", "josh", "emma", "michael"],
            index=0
        )
        
        st.session_state.max_tokens = st.slider(
            "Max Response Length",
            min_value=50,
            max_value=500,
            value=128,
            step=10
        )
        
        st.session_state.enable_voice = st.toggle("Enable Voice Features", value=False)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.experimental_rerun()
        
        st.markdown("---")
        st.markdown("""
        ### About
        This is a ChatGPT-style interface powered by Lepton AI that supports:
        - Text chat
        - Voice input (optional)
        - Voice output (optional)
        """)

def main():
    initialize_session_state()
    render_sidebar()
    
    # Initialize client
    try:
        client = initialize_lepton_client()
    except Exception as e:
        st.error(f"Failed to connect: {str(e)}\nPlease check your LEPTON_API_TOKEN in .env file")
        return

    # Chat messages container
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(f'<div class="{role_class}">{message["content"]}</div>', unsafe_allow_html=True)
        
        if "audio_path" in message and message["audio_path"]:
            with open(message["audio_path"], 'rb') as audio_file:
                st.audio(audio_file.read(), format='audio/mp3')
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input container
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Text input
        user_input = st.text_input("Send a message:", key="user_input")
        
        # Voice input
        if st.session_state.enable_voice:
            audio_bytes = audio_recorder(
                pause_threshold=2.0,
                sample_rate=44100,
                text="",
                recording_color="#19c37d",
                neutral_color="#e5e5e5",
                icon_name="microphone",
                icon_size="2x"
            )
            
            if audio_bytes:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                    with open(tmp_file.name, 'rb') as audio_file:
                        audio_data = process_audio_file(audio_file)
                    
                    response_text, audio_chunks = generate_response(
                        client,
                        st.session_state.messages,
                        audio_data=audio_data,
                        generate_audio=st.session_state.enable_voice
                    )
                    
                    # Add messages to chat history
                    st.session_state.messages.append({"role": "user", "content": "🎤 Voice message"})
                    
                    audio_path = save_audio(audio_chunks) if audio_chunks else ""
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "audio_path": audio_path
                    })
                    
                    os.unlink(tmp_file.name)
                    st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle text input
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate response
            response_text, audio_chunks = generate_response(
                client,
                st.session_state.messages,
                generate_audio=st.session_state.enable_voice
            )
            
            # Add assistant response to chat
            audio_path = save_audio(audio_chunks) if audio_chunks else ""
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "audio_path": audio_path
            })
            
            # Clear input
            st.session_state.user_input = ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()