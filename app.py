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

    for chunk in completion:
        if not chunk.choices:
            continue
        
        content = chunk.choices[0].delta.content
        audio = getattr(chunk.choices[0], 'audio', [])
        
        if content:
            full_response += content
            st.write(content, end="")
            
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

def main():
    st.title("Voice and Text Generation App")
    
    # Initialize the client
    try:
        client = initialize_lepton_client()
    except Exception as e:
        st.error(f"Failed to initialize Lepton client: {str(e)}")
        st.error("Please make sure LEPTON_API_TOKEN is set in your .env file or environment variables")
        return

    # Input method selection
    input_method = st.radio("Choose input method:", ["Text", "Voice"])
    
    if input_method == "Text":
        user_input = st.text_area("Enter your prompt:", height=100)
        generate_audio = st.checkbox("Generate audio response")
        
        if st.button("Generate"):
            if user_input:
                with st.spinner("Generating response..."):
                    response_text, audio_chunks = generate_response(
                        client, 
                        user_input,
                        generate_audio=generate_audio
                    )
                    
                    if generate_audio:
                        audio_path = save_audio(audio_chunks)
                        if audio_path:
                            with open(audio_path, 'rb') as audio_file:
                                st.audio(audio_file.read(), format='audio/mp3')
                            os.unlink(audio_path)  # Clean up temporary file
            else:
                st.warning("Please enter a prompt.")
                
    else:  # Voice input
        st.write("Record your voice:")
        audio_bytes = audio_recorder()
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            # Save the recorded audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                
            generate_audio = st.checkbox("Generate audio response")
            
            if st.button("Process"):
                with open(tmp_file.name, 'rb') as audio_file:
                    audio_data = process_audio_file(audio_file)
                
                with st.spinner("Processing audio..."):
                    response_text, audio_chunks = generate_response(
                        client,
                        "",
                        audio_data=audio_data,
                        generate_audio=generate_audio
                    )
                    
                    if generate_audio:
                        audio_path = save_audio(audio_chunks)
                        if audio_path:
                            with open(audio_path, 'rb') as audio_file:
                                st.audio(audio_file.read(), format='audio/mp3')
                            os.unlink(audio_path)  # Clean up temporary file
                
                os.unlink(tmp_file.name)  # Clean up temporary file

if __name__ == "__main__":
    main()