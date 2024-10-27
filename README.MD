# Lepton AI Voice and Text Generation App

A Streamlit-based web application that enables both text and voice interaction with Lepton AI's language models. The app supports text-to-text, speech-to-text, and text-to-speech capabilities using the Lepton AI API.

## Features

- 🎙️ Voice Input Recording
- ⌨️ Text Input Support
- 🔊 Audio Response Generation
- 💬 Real-time Text Streaming
- 🎯 Multi-modal Interaction
- 🔄 Voice-to-Text Conversion
- 🗣️ Text-to-Speech Synthesis

## Prerequisites

- Python 3.8 or higher
- Lepton AI API Token
- Microphone (for voice input)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lepton-voice-app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Lepton AI API token:
```bash
LEPTON_API_TOKEN=your_token_here
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically `http://localhost:8501`)

3. Choose your input method:
   - **Text Input**: Type your prompt and optionally enable audio response
   - **Voice Input**: Record your voice and get both text and optional audio responses

## Project Structure

```
lepton-voice-app/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
└── README.md          # Project documentation
```

## Dependencies

- streamlit==1.32.0
- openai==1.12.0
- audio-recorder-streamlit==0.0.8
- python-dotenv==1.0.1

## Configuration

The application uses the following environment variables:

- `LEPTON_API_TOKEN`: Your Lepton AI API authentication token

## API Reference

The application uses the Lepton AI API with the following endpoints:

- Base URL: `https://llama3-1-405b.lepton.run/api/v1/`
- Model: `llama3.1-405b`

## Features in Detail

### Text Generation
- Input text prompts
- Real-time streaming responses
- Configurable token limits

### Voice Input
- Browser-based audio recording
- WAV format support
- Real-time audio preview

### Audio Generation
- Text-to-Speech synthesis
- MP3 format output
- Multiple voice presets available

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

Common issues and solutions:

1. **API Token Error**
   - Ensure your `.env` file exists and contains the correct API token
   - Check if python-dotenv is properly installed

2. **Audio Recording Issues**
   - Grant microphone permissions in your browser
   - Check if your microphone is properly connected
   - Try using a different browser if issues persist

3. **Streamlit Connection Error**
   - Verify that port 8501 is not in use
   - Check your internet connection
   - Ensure all dependencies are correctly installed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Lepton AI](https://www.lepton.ai/) for providing the API
- [Streamlit](https://streamlit.io/) for the web framework
- [OpenAI](https://github.com/openai/openai-python) for the Python client library

## Contact

For any questions or support, please open an issue in the repository or contact the maintainers.