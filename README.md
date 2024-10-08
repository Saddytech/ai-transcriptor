# AI Video Transcriber

AI Video Transcriber is a powerful and user-friendly application that allows you to transcribe audio from video files using state-of-the-art speech recognition technologies. It supports both Whisper and Vosk models for accurate transcription across multiple languages.

## Description

This application provides a graphical user interface for transcribing video files. It extracts the audio from video files, applies noise reduction, and then uses either the Whisper or Vosk speech recognition model to generate accurate transcriptions. The tool also offers features like subtitle generation, multiple output formats, and advanced settings for fine-tuning the transcription process.

## Features

- Support for multiple video file formats (mp4, avi, mov, mkv, flv, wmv)
- Audio extraction from video files
- Noise reduction for improved transcription accuracy
- Choice between Whisper and Vosk speech recognition engines
- Multiple Whisper model sizes (tiny, base, small, medium, large, large-v2)
- Support for custom Vosk model directories
- Automatic language detection or manual language selection
- Multiple output formats (Text, JSON, CSV, DOCX)
- Subtitle (.srt) file generation
- Advanced settings for fine-tuning transcription parameters
- Progress tracking and resource usage monitoring
- Logging of transcription process

## Requirements

- Python 3.6+
- FFmpeg
- PyTorch
- Whisper
- Vosk
- Other dependencies listed in `requirements.txt` (to be created)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Saddytech/ai-video-transcriber.git
   cd ai-video-transcriber
   ```

2. Install FFmpeg:
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - On macOS: `brew install ffmpeg`
   - On Linux: `sudo apt-get install ffmpeg`

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download Whisper models (optional):
   The application will download models automatically, but you can pre-download them from [OpenAI's Whisper repository](https://github.com/openai/whisper).

5. Download Vosk models (optional):
   If you plan to use Vosk, download models from the [Vosk website](https://alphacephei.com/vosk/models).

## Usage

1. Run the application:
   ```
   python transcript.py
   ```

2. Use the GUI to:
   - Select video file(s) for transcription
   - Choose an output directory
   - Select the speech recognition engine (Whisper or Vosk)
   - Choose the model size or path
   - Set the transcription language (or use automatic detection)
   - Adjust advanced settings if needed
   - Start the transcription process

3. Monitor the progress and resource usage in the application window.

4. Find the transcription results and subtitle files in your chosen output directory.

## Advanced Settings

- **Beam Size**: For Whisper model, sets the beam size for beam search decoding.
- **Best Of**: For Whisper model, sets the number of candidates when sampling with non-zero temperature.
- **Temperature**: For Whisper model, sets the temperature for sampling. Lower values make output more deterministic.
- **Segment Length**: Sets the length of audio segments for processing. Shorter segments use less memory.

## Contributing

Contributions to the AI Video Transcriber are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
