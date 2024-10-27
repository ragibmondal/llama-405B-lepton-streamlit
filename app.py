import streamlit as st
import os
import base64
import openai
from typing import List, Optional
import tempfile
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="AI Voice & Text Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design elements
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #fafafa;
            padding: 2rem;
        }
        
        /* Custom container styling */
        .custom-container {
            background-color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3em;
            background-color: #7C3AED;
            color: white;
            border: none;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #6D28D9;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
        }
        
        /* Text area styling */
        .stTextArea textarea {
            border-radius: 10px;
            border: 2px solid #E5E7EB;
            padding: 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextArea textarea:focus {
            border-color: #7C3AED;
            box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2);
        }
        
        /* Status boxes */
        .status-box {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .success-box {
            background-color: #ECFDF5;
            border: 1px solid #059669;
            color: #065F46;
        }
        
        .error-box {
            background-color: #FEF2F2;
            border: 1px solid #DC2626;
            color: #991B1B;
        }
        
        .info-box {
            background-color: #EFF6FF;
            border: 1px solid #3B82F6;
            color: #1E40AF;
        }
        
        /* Audio recorder styling */
        .recorder-container {
            display: flex;
            justify-content: center;
            margin: 2rem 0;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #F3F4F6;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #1F2937;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #F9FAFB;
            border-radius: 10px;
            color: #4B5563;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #7C3AED;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

def initialize_lepton_client():
    """Initialize the Lepton AI client with error handling"""
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
                     generate_audio: bool = False,
                     max_tokens: int = 128,
                     voice_preset: str = "jessica") -> tuple[str, List[str]]:
    """Enhanced response generation with progress tracking"""
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
            "tts_preset_id": voice_preset,
        })

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    completion = client.chat.completions.create(
        model="llama3.1-405b",
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        extra_body=extra_body if extra_body else None
    )

    full_response = ""
    audio_chunks = []
    response_placeholder = st.empty()
    
    for i, chunk in enumerate(completion):
        if not chunk.choices:
            continue
        
        content = chunk.choices[0].delta.content
        audio = getattr(chunk.choices[0], 'audio', [])
        
        if content:
            full_response += content
            response_placeholder.markdown(f"""
            <div class="custom-container">
                <h4>AI Response:</h4>
                {full_response}
            </div>
            """, unsafe_allow_html=True)
            
        if audio:
            audio_chunks.extend(audio)
        
        # Update progress
        progress = min(100, int((i + 1) / max_tokens * 100))
        progress_bar.progress(progress)
        status_text.text(f"Generating response... {progress}%")

    progress_bar.empty()
    status_text.empty()
    
    return full_response, audio_chunks

def save_audio(audio_chunks: List[str]) -> str:
    """Save audio chunks to a temporary file"""
    if not audio_chunks:
        return ""
    
    audio_data = b''.join([base64.b64decode(chunk) for chunk in audio_chunks])
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
        tmp_file.write(audio_data)
        return tmp_file.name

def render_sidebar():
    """Render an enhanced sidebar with settings"""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        with st.expander("🎤 Voice Settings", expanded=True):
            voice_preset = st.selectbox(
                "Voice Preset",
                ["jessica", "josh", "emma", "michael"],
                index=0,
                help="Select the AI voice personality"
            )
        
        with st.expander("🤖 Model Settings", expanded=True):
            max_tokens = st.slider(
                "Max Response Length",
                min_value=50,
                max_value=500,
                value=128,
                step=10,
                help="Control the length of AI responses"
            )
        
        st.markdown("---")
        
        with st.expander("ℹ️ About", expanded=True):
            st.markdown("""
            ### AI Voice & Text Assistant
            
            This application leverages Lepton AI to provide:
            - 🎤 Speech-to-Text Conversion
            - 🤖 AI-Powered Responses
            - 🔊 Text-to-Speech Generation
            
            Built with Streamlit and ❤️
            """)
        
        return voice_preset, max_tokens

def main():
    voice_preset, max_tokens = render_sidebar()
    
    # Main content
    st.title("🎙️ AI Voice & Text Assistant")
    st.markdown("Interact naturally with AI using voice or text")
    
    # Initialize client with error handling
    try:
        client = initialize_lepton_client()
        st.markdown("""
            <div class="status-box success-box">
                ✅ Connected to Lepton AI API successfully
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"""
            <div class="status-box error-box">
                ❌ Connection Error: {str(e)}<br>
                Please verify your LEPTON_API_TOKEN in the .env file
            </div>
        """, unsafe_allow_html=True)
        return

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Text Input", "🎤 Voice Input"])
    
    with tab1:
        with st.container():
            st.markdown("""
            <div class="custom-container">
                <h3>💬 Text Conversation</h3>
            """, unsafe_allow_html=True)
            
            user_input = st.text_area("Type your message:", height=150, 
                                    placeholder="Enter your message here...")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                generate_audio = st.checkbox("Enable voice response", value=True)
            
            with col2:
                if st.button("Send Message", key="text_send"):
                    if user_input:
                        with st.spinner("🤖 Processing..."):
                            response_text, audio_chunks = generate_response(
                                client, 
                                user_input,
                                generate_audio=generate_audio,
                                max_tokens=max_tokens,
                                voice_preset=voice_preset
                            )
                            
                            if generate_audio and audio_chunks:
                                audio_path = save_audio(audio_chunks)
                                if audio_path:
                                    st.markdown("### 🔊 AI Voice Response")
                                    with open(audio_path, 'rb') as audio_file:
                                        st.audio(audio_file.read(), format='audio/mp3')
                                    os.unlink(audio_path)
                    else:
                        st.warning("⚠️ Please enter a message.")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        with st.container():
            st.markdown("""
            <div class="custom-container">
                <h3>🎤 Voice Interaction</h3>
                <p>Click the microphone button and start speaking:</p>
            </div>
            """, unsafe_allow_html=True)
            
            status_placeholder = st.empty()
            
            with st.container():
                audio_bytes = audio_recorder(
                    pause_threshold=2.0,
                    sample_rate=44100,
                    text="",
                    recording_color="#7C3AED",
                    neutral_color="#6B7280",
                    icon_name="microphone",
                    icon_size="2x"
                )
            
            if audio_bytes:
                status_placeholder.markdown("""
                    <div class="status-box success-box">
                        ✅ Audio recorded successfully!
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 📢 Your Recording")
                st.audio(audio_bytes, format="audio/wav")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(audio_bytes)
                
                generate_audio = st.checkbox("Enable voice response", value=True, key="voice_input_audio")
                
                if st.button("Process Recording", key="voice_send"):
                    with st.spinner("🎯 Processing voice input..."):
                        with open(tmp_file.name, 'rb') as audio_file:
                            audio_data = process_audio_file(audio_file)
                        
                        response_text, audio_chunks = generate_response(
                            client,
                            "",
                            audio_data=audio_data,
                            generate_audio=generate_audio,
                            max_tokens=max_tokens,
                            voice_preset=voice_preset
                        )
                        
                        if generate_audio and audio_chunks:
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
                        🎤 Click the microphone icon to start recording
                    </div>
                """, unsafe_allow_html=True)

    # Persistent chat history section
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("---")
    with st.expander("💬 Chat History", expanded=False):
        for message in st.session_state.messages:
            st.markdown(f"""
            <div class="custom-container">
                <strong>{'🤖 AI' if message['role'] == 'assistant' else '👤 You'}:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()