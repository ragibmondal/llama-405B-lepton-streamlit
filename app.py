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
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
            background-color: #FF4B4B;
            color: white;
            border: none;
        }
        .stButton>button:hover {
            background-color: #FF2B2B;
            color: white;
            border: none;
        }
        .recorder-button {
            background-color: #FF4B4B !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 10px 20px !important;
        }
        .css-1v0mbdj.etr89bj1 {
            margin-top: 20px;
        }
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
                     prompt: str, 
                     audio_data: Optional[str] = None,
                     generate_audio: bool = False) -> tuple[str, List[str]]:
    """Generate response from LLM with optional audio input/output"""
    
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
    """Save audio chunks to a temporary file and return the path"""
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
        voice_preset = st.selectbox(
            "Voice Preset",
            ["jessica", "josh", "emma", "michael"],
            index=0
        )
        
        st.markdown("---")
        st.subheader("Model Settings")
        max_tokens = st.slider(
            "Max Response Length",
            min_value=50,
            max_value=500,
            value=128,
            step=10
        )
        
        st.markdown("---")
        st.markdown("""
        ### About
        This app uses Lepton AI to:
        - Convert speech to text
        - Generate AI responses
        - Convert text to speech
        """)

def main():
    render_sidebar()
    
    # Main content
    st.title("🎙️ AI Voice & Text Assistant")
    st.markdown("Interact with AI using voice or text input")
    
    # Initialize the client
    try:
        client = initialize_lepton_client()
        with st.container():
            st.markdown("""
            <div class="status-box success-box">
                ✅ Connected to Lepton AI API
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        with st.container():
            st.markdown(f"""
            <div class="status-box error-box">
                ❌ Failed to connect: {str(e)}<br>
                Please check your LEPTON_API_TOKEN in .env file
            </div>
            """, unsafe_allow_html=True)
        return

    # Create two columns for input methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📝 Text Input")
        user_input = st.text_area("Type your message:", height=150)
        generate_audio = st.checkbox("Enable voice response", value=True)
        
        if st.button("Send Message"):
            if user_input:
                with st.spinner("🤖 AI is thinking..."):
                    response_text, audio_chunks = generate_response(
                        client, 
                        user_input,
                        generate_audio=generate_audio
                    )
                    
                    if generate_audio:
                        audio_path = save_audio(audio_chunks)
                        if audio_path:
                            st.markdown("### 🔊 AI Voice Response")
                            with open(audio_path, 'rb') as audio_file:
                                st.audio(audio_file.read(), format='audio/mp3')
                            os.unlink(audio_path)
            else:
                st.warning("⚠️ Please enter a message.")
    
    with col2:
        st.markdown("### 🎤 Voice Input")
        st.markdown("Click the button below and start speaking:")
        
        # Add a placeholder for the recording status
        status_placeholder = st.empty()
        
        audio_bytes = audio_recorder(
            pause_threshold=2.0,
            sample_rate=44100,
            text="",
            recording_color="#FF4B4B",
            neutral_color="#6B7280",
            icon_name="microphone",
            icon_size="2x"
        )
        
        if audio_bytes:
            status_placeholder.markdown("""
                <div class="status-box info-box">
                    🎵 Audio recorded successfully!
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📢 Your Recording")
            st.audio(audio_bytes, format="audio/wav")
            
            # Save the recorded audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                
            generate_audio = st.checkbox("Enable voice response", value=True, key="voice_input_audio")
            
            if st.button("Process Recording"):
                with open(tmp_file.name, 'rb') as audio_file:
                    audio_data = process_audio_file(audio_file)
                
                with st.spinner("🎯 Processing your voice..."):
                    response_text, audio_chunks = generate_response(
                        client,
                        "",
                        audio_data=audio_data,
                        generate_audio=generate_audio
                    )
                    
                    if generate_audio:
                        audio_path = save_audio(audio_chunks)
                        if audio_path:
                            st.markdown("### 🔊 AI Voice Response")
                            with open(audio_path, 'rb') as audio_file:
                                st.audio(audio_file.read(), format='audio/mp3')
                            os.unlink(audio_path)
                
                os.unlink(tmp_file.name)
        else:
            status_placeholder.markdown("""
                <div class="status-box info-box">
                    🎤 Click the microphone to start recording
                </div>
            """, unsafe_allow_html=True)

    # Chat history section
    st.markdown("---")
    st.markdown("### 💬 Chat History")
    st.info("Chat history is not yet implemented. Coming soon!")

if __name__ == "__main__":
    main()