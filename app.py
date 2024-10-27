import streamlit as st
import os
import base64
import openai
from typing import List, Optional
import tempfile
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Voice & Text Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS for OpenWebUI-like interface
st.markdown("""
    <style>
        /* Global styles */
        .main {
            background-color: #ffffff;
        }
        
        [data-testid="stSidebar"] {
            background-color: #f9fafb;
        }
        
        /* Chat container */
        .chat-container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .message-container {
            display: flex;
            padding: 1rem !important;
            margin: 0.5rem 0;
            border-radius: 0.5rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
            margin-bottom: 1rem !important;
        }
        
        .user-message {
            background-color: #e7f5ff !important;
            border: 1px solid #d0ebff !important;
        }
        
        .assistant-message {
            background-color: #f8f9fa !important;
            border: 1px solid #e9ecef !important;
        }
        
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        
        .message-content {
            flex-grow: 1;
        }
        
        /* Input area */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: white;
            padding: 1rem;
            border-top: 1px solid #e5e7eb;
            z-index: 1000;
        }
        
        .stTextArea textarea {
            border-radius: 0.5rem;
            border: 1px solid #e5e7eb;
            padding: 0.75rem;
            font-size: 1rem;
            resize: none;
            height: 60px !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        }
        
        .stTextArea textarea:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.2);
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #2563eb;
            color: white;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            border: none;
            height: 40px;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            background-color: #1d4ed8;
        }
        
        /* Voice recorder */
        .recorder-button {
            background-color: transparent !important;
            border: 1px solid #e5e7eb !important;
            border-radius: 0.5rem !important;
            color: #374151 !important;
            padding: 0.5rem !important;
        }
        
        /* Hide unnecessary elements */
        #MainMenu, footer {
            visibility: hidden;
        }
        
        /* Audio player */
        audio {
            width: 100%;
            border-radius: 0.5rem;
            margin-top: 0.5rem;
        }
        
        /* Status messages */
        .status-message {
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            font-size: 0.875rem;
        }
        
        .info {
            background-color: #eff6ff;
            color: #1e40af;
            border: 1px solid #bfdbfe;
        }
        
        .success {
            background-color: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
        }
        
        .error {
            background-color: #fef2f2;
            color: #991b1b;
            border: 1px solid #fecaca;
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
            "tts_preset_id": "jessica",
        })

    completion = client.chat.completions.create(
        model="llama3.1-405b",
        messages=messages,
        max_tokens=128,
        stream=True,
        extra_body=extra_body if extra_body else None
    )

    full_response = ""
    audio_chunks = []

    # Get the message container from session state
    message_container = st.session_state.get('message_container')
    if message_container:
        with message_container:
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

def render_message(role: str, content: str, audio_path: Optional[str] = None):
    """Render a chat message with OpenWebUI-like styling"""
    avatar = "🤖" if role == "assistant" else "👤"
    bg_class = "assistant-message" if role == "assistant" else "user-message"
    
    st.markdown(f"""
        <div class="message-container {bg_class}">
            <div class="message-avatar">{avatar}</div>
            <div class="message-content">
                {content}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if audio_path:
        st.audio(audio_path)

def render_chat_interface():
    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        render_message(
            role=message["role"],
            content=message["content"],
            audio_path=message.get("audio_path")
        )
    
    # Store message container in session state for streaming
    message_container = st.container()
    st.session_state.message_container = message_container
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return message_container

def render_input_area(client):
    """Render the input area with text and voice options"""
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Initialize session state for input reset
    if "reset_input" not in st.session_state:
        st.session_state.reset_input = False
        
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        # Set empty default value when reset is triggered
        default_value = "" if st.session_state.reset_input else st.session_state.get("current_input", "")
        user_input = st.text_area(
            "Message", 
            value=default_value,
            key="input_area",
            label_visibility="collapsed"
        )
        # Store current input value
        st.session_state.current_input = user_input
    
    with col2:
        audio_bytes = audio_recorder(
            pause_threshold=2.0,
            sample_rate=44100,
            text="",
            recording_color="#2563eb",
            neutral_color="#6B7280",
            icon_name="microphone",
            icon_size="lg"
        )
    
    with col3:
        send_button = st.button("Send")
    
    # Handle text input
    if send_button and user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        response_text, audio_chunks = generate_response(
            client, 
            user_input,
            generate_audio=True
        )
        
        # Save audio if generated
        audio_path = None
        if audio_chunks:
            audio_path = save_audio(audio_chunks)
        
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "audio_path": audio_path
        })
        
        # Trigger input reset for next rerun
        st.session_state.reset_input = True
        st.session_state.current_input = ""
        st.rerun()
    
    # Reset the reset flag after rerun
    st.session_state.reset_input = False
    
    # Handle voice input
    if audio_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            
            # Add user audio message
            st.session_state.messages.append({
                "role": "user",
                "content": "🎤 Voice message",
                "audio_path": tmp_file.name
            })
            
            # Process voice input
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = process_audio_file(audio_file)
            
            # Generate response
            response_text, audio_chunks = generate_response(
                client,
                "",
                audio_data=audio_data,
                generate_audio=True
            )
            
            # Save response audio
            audio_path = None
            if audio_chunks:
                audio_path = save_audio(audio_chunks)
            
            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "audio_path": audio_path
            })
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Initialize client
    try:
        client = initialize_lepton_client()
    except Exception as e:
        st.error(f"Failed to initialize client: {str(e)}")
        return
    
    # Render chat interface
    message_container = render_chat_interface()
    
    # Render input area
    render_input_area(client)

if __name__ == "__main__":
    main()
