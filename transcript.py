import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import ffmpeg
import whisper
import numpy as np
import torch
import pysrt
import psutil
import configparser
import logging
import noisereduce as nr
from vosk import Model as VoskModel, KaldiRecognizer

import json
import wave

# Initialize logging
logging.basicConfig(filename='transcription_app.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Video Transcriber")
        self.root.geometry("750x800")
        self.root.resizable(False, False)

        # Load settings
        self.load_settings()

        self.create_widgets()
        self.log_file = "transcription_log.txt"

        # Load model cache
        self.model_cache = {}

        # Start resource monitoring
        self.update_resource_usage()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')

        # Video File Selection
        ttk.Label(self.root, text="Select Video Files:").pack(pady=5)
        video_frame = ttk.Frame(self.root)
        video_frame.pack(pady=5)
        ttk.Entry(video_frame, textvariable=self.video_paths_var, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(video_frame, text="Browse", command=self.browse_videos).pack(side=tk.LEFT)
        self.create_tooltip(video_frame, "Select one or more video files you want to transcribe.")

        # Output Directory Selection
        ttk.Label(self.root, text="Select Output Directory:").pack(pady=5)
        output_frame = ttk.Frame(self.root)
        output_frame.pack(pady=5)
        ttk.Entry(output_frame, textvariable=self.output_dir, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="Browse", command=self.browse_output).pack(side=tk.LEFT)
        self.create_tooltip(output_frame, "Choose where to save the transcription and subtitles.")

        # Model Selection
        ttk.Label(self.root, text="Select Speech Recognition Engine:").pack(pady=5)
        model_engine_frame = ttk.Frame(self.root)
        model_engine_frame.pack(pady=5)
        self.model_engine = tk.StringVar(value='Whisper')
        model_engine_options = ['Whisper', 'Vosk']
        ttk.OptionMenu(model_engine_frame, self.model_engine, self.model_engine.get(), *model_engine_options, command=self.update_model_options).pack()
        self.create_tooltip(model_engine_frame, "Select the speech recognition engine.")

        # Whisper Model Selection
        self.whisper_model_frame = ttk.Frame(self.root)
        self.whisper_model_frame.pack(pady=5)
        ttk.Label(self.whisper_model_frame, text="Select Whisper Model Size:").pack(pady=5)
        self.model_size = tk.StringVar(value='base')
        model_options = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2']
        ttk.OptionMenu(self.whisper_model_frame, self.model_size, self.model_size.get(), *model_options).pack()
        self.create_tooltip(self.whisper_model_frame, "Select the Whisper model size. Larger models are more accurate but require more resources.")

        # Vosk Model Path
        self.vosk_model_frame = ttk.Frame(self.root)
        ttk.Label(self.vosk_model_frame, text="Select Vosk Model Directory:").pack(pady=5)
        vosk_model_dir_frame = ttk.Frame(self.vosk_model_frame)
        vosk_model_dir_frame.pack(pady=5)
        self.vosk_model_path = tk.StringVar()
        ttk.Entry(vosk_model_dir_frame, textvariable=self.vosk_model_path, width=60).pack(side=tk.LEFT, padx=5)
        ttk.Button(vosk_model_dir_frame, text="Browse", command=self.browse_vosk_model).pack(side=tk.LEFT)
        self.create_tooltip(vosk_model_dir_frame, "Select the directory containing the Vosk model.")

        # Language Selection
        ttk.Label(self.root, text="Select Transcription Language:").pack(pady=5)
        language_frame = ttk.Frame(self.root)
        language_frame.pack(pady=5)
        self.transcription_language = tk.StringVar(value="Automatic Detection")
        language_options = sorted(whisper.tokenizer.LANGUAGES.keys())
        language_options.insert(0, "Automatic Detection")
        ttk.OptionMenu(language_frame, self.transcription_language, self.transcription_language.get(), *language_options).pack()
        self.create_tooltip(language_frame, "Select the language spoken in the video or choose automatic detection.")

        # Audio Stream Index Selection
        ttk.Label(self.root, text="Audio Stream Index (if multiple audio tracks):").pack(pady=5)
        stream_frame = ttk.Frame(self.root)
        stream_frame.pack(pady=5)
        self.audio_stream_index = tk.IntVar(value=0)
        ttk.Entry(stream_frame, textvariable=self.audio_stream_index, width=5).pack(side=tk.LEFT)
        ttk.Label(stream_frame, text="(Default is 0)").pack(side=tk.LEFT)
        self.create_tooltip(stream_frame, "Specify the audio stream index if your video has multiple audio tracks.")

        # Save Subtitles Option
        self.save_subtitles = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.root, text="Save as Subtitles (.srt)", variable=self.save_subtitles).pack(pady=5)
        self.create_tooltip(self.root, "Check this box to save the transcription as a subtitle file.")

        # Output Format Selection
        ttk.Label(self.root, text="Select Output Format:").pack(pady=5)
        format_frame = ttk.Frame(self.root)
        format_frame.pack(pady=5)
        self.output_format = tk.StringVar(value='Text')
        format_options = ['Text', 'JSON', 'CSV', 'DOCX']
        ttk.OptionMenu(format_frame, self.output_format, self.output_format.get(), *format_options).pack()
        self.create_tooltip(format_frame, "Choose the format for saving the transcription.")

        # Advanced Settings Button
        ttk.Button(self.root, text="Advanced Settings", command=self.open_advanced_settings).pack(pady=5)
        self.create_tooltip(self.root, "Click to adjust advanced transcription settings.")

        # Transcribe Button
        ttk.Button(self.root, text="Transcribe", command=self.start_transcription).pack(pady=15)
        self.create_tooltip(self.root, "Click to start the transcription process.")

        # Progress Bar and Logging
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=600, mode='determinate')
        self.progress.pack(pady=5)

        self.progress_label = ttk.Label(self.root, text="Progress: 0%")
        self.progress_label.pack()

        self.status_label = ttk.Label(self.root, text="Status: Ready")
        self.status_label.pack()

        self.resource_label = ttk.Label(self.root, text="")
        self.resource_label.pack()

        self.log_text = tk.Text(self.root, height=15, width=85)
        self.log_text.pack(pady=10)

        # Update model options visibility based on the selected engine
        self.update_model_options(self.model_engine.get())

    def create_tooltip(self, widget, text):
        tool_tip = ToolTip(widget, text)

    def browse_videos(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv")]
        )
        if file_paths:
            self.video_paths = list(file_paths)
            self.video_paths_var.set('; '.join(self.video_paths))

    def browse_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)

    def browse_vosk_model(self):
        directory = filedialog.askdirectory(title="Select Vosk Model Directory")
        if directory:
            self.vosk_model_path.set(directory)

    def open_advanced_settings(self):
        # Create a new window for advanced settings
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Advanced Settings")
        settings_window.geometry("300x250")
        settings_window.resizable(False, False)

        # Beam Size
        ttk.Label(settings_window, text="Beam Size:").pack(pady=5)
        ttk.Entry(settings_window, textvariable=self.beam_size).pack()
        self.create_tooltip(settings_window, "Beam size for beam search decoding. Higher values may improve accuracy at the cost of speed.")

        # Best Of
        ttk.Label(settings_window, text="Best Of:").pack(pady=5)
        ttk.Entry(settings_window, textvariable=self.best_of).pack()
        self.create_tooltip(settings_window, "Number of candidates when sampling with non-zero temperature.")

        # Temperature
        ttk.Label(settings_window, text="Temperature:").pack(pady=5)
        ttk.Entry(settings_window, textvariable=self.temperature).pack()
        self.create_tooltip(settings_window, "Temperature for sampling. Lower values make the output more deterministic.")

        # Segment Length
        ttk.Label(settings_window, text="Segment Length (s):").pack(pady=5)
        ttk.Entry(settings_window, textvariable=self.segment_length).pack()
        self.create_tooltip(settings_window, "Length of audio segments for processing. Shorter segments use less memory.")

        # Close Button
        ttk.Button(settings_window, text="Close", command=settings_window.destroy).pack(pady=10)

    def update_model_options(self, engine):
        if engine == 'Whisper':
            self.whisper_model_frame.pack(pady=5)
            self.vosk_model_frame.pack_forget()
        elif engine == 'Vosk':
            self.vosk_model_frame.pack(pady=5)
            self.whisper_model_frame.pack_forget()

    def start_transcription(self):
        if not self.video_paths:
            messagebox.showwarning("Warning", "Please select video files.")
            return
        if not self.output_dir.get():
            messagebox.showwarning("Warning", "Please select an output directory.")
            return
        if self.model_engine.get() == 'Vosk' and not self.vosk_model_path.get():
            messagebox.showwarning("Warning", "Please select the Vosk model directory.")
            return

        self.progress['value'] = 0
        self.progress_label.config(text="Progress: 0%")
        self.status_label.config(text="Status: Starting transcription...")
        self.log_text.delete(1.0, tk.END)

        # Save settings
        self.save_settings()

        threading.Thread(target=self.process_videos).start()

    def process_videos(self):
        total_videos = len(self.video_paths)
        for idx, video_path in enumerate(self.video_paths):
            self.current_video_index = idx
            self.total_videos = total_videos
            self.video_path = video_path
            try:
                self.transcribe_video()
            except Exception as e:
                self.log(f"Error processing {video_path}: {e}")
                continue
        self.status_label.config(text="Status: All videos processed.")
        messagebox.showinfo("Success", "All videos have been processed.")

    def transcribe_video(self):
        try:
            self.start_time = time.time()
            self.log(f"Processing video {self.current_video_index + 1}/{self.total_videos}: {self.video_path}")

            self.log("Extracting audio from video...")
            audio_file, duration = self.extract_audio(self.video_path)

            self.log(f"Audio extracted. Duration: {duration:.2f} seconds.")

            if self.model_engine.get() == 'Whisper':
                self.load_whisper_model()
                self.log("Starting transcription with Whisper...")
                transcription, segments = self.transcribe_audio_whisper(audio_file, duration)
            elif self.model_engine.get() == 'Vosk':
                self.load_vosk_model()
                self.log("Starting transcription with Vosk...")
                transcription, segments = self.transcribe_audio_vosk(audio_file, duration)
            else:
                raise Exception("Unknown speech recognition engine selected.")

            self.save_transcription(transcription, self.video_path)
            if self.save_subtitles.get():
                self.save_subtitles_file(segments, self.video_path)

            end_time = time.time()
            total_time = end_time - self.start_time
            self.log(f"Transcription completed in {total_time:.2f} seconds.")
            self.status_label.config(text="Status: Transcription completed for current video.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log(f"Error: {e}")
            self.status_label.config(text="Status: Error occurred.")

    def extract_audio(self, video_path):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(self.output_dir.get(), f"{base_name}_audio.wav")
        stream_index = self.audio_stream_index.get()

        # Extract the specified audio stream
        try:
            (
                ffmpeg
                .input(video_path)
                .output(
                    audio_path,
                    format='wav',
                    acodec='pcm_s16le',
                    ac=1,
                    ar='16000',
                    audio_bitrate='192k',
                    **{'map': f'0:a:{stream_index}'}
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            self.log(f"FFmpeg error: {e.stderr.decode()}")
            raise Exception("Failed to extract audio. Please check the audio stream index.")

        # Get audio duration
        try:
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['format']['duration'])
        except Exception as e:
            self.log(f"FFmpeg probe error: {e}")
            raise Exception("Failed to get audio duration.")

        return audio_path, duration

    def load_whisper_model(self):
        model_size = self.model_size.get()
        if model_size not in self.model_cache:
            self.log(f"Loading Whisper {model_size} model...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model_cache[model_size] = whisper.load_model(model_size, device=device)
        self.model = self.model_cache[model_size]

    def load_vosk_model(self):
        model_path = self.vosk_model_path.get()
        if model_path not in self.model_cache:
            self.log(f"Loading Vosk model from {model_path}...")
            self.model_cache[model_path] = VoskModel(model_path)
        self.model = self.model_cache[model_path]

    def transcribe_audio_whisper(self, audio_path, duration):
        self.log("Transcribing the audio file with Whisper...")

        # Prepare options
        language_option = None if self.transcription_language.get() == "Automatic Detection" else self.transcription_language.get()
        options = {
            'language': language_option,
            'temperature': self.temperature.get(),
            'fp16': torch.cuda.is_available(),
            'task': 'transcribe'
        }

        # Choose decoding method
        if self.beam_size.get() > 0 and self.best_of.get() > 0:
            messagebox.showwarning("Warning", "Cannot set both Beam Size and Best Of. Please choose one.")
            return "", []
        elif self.beam_size.get() > 0:
            options['beam_size'] = self.beam_size.get()
        elif self.best_of.get() > 0:
            options['best_of'] = self.best_of.get()

        # Load and preprocess the audio
        audio = whisper.load_audio(audio_path)
        audio_length = audio.shape[0] / whisper.audio.SAMPLE_RATE
        self.log(f"Audio length from array: {audio_length:.2f} seconds")
        self.log(f"Audio duration from metadata: {duration:.2f} seconds")

        # Apply noise reduction
        self.log("Applying noise reduction...")
        audio = nr.reduce_noise(y=audio, sr=whisper.audio.SAMPLE_RATE)

        # Segment the audio for processing
        segment_length = self.segment_length.get()  # seconds
        total_segments = int(np.ceil(audio_length / segment_length))
        self.log(f"Total segments: {total_segments}")
        segments = []

        # Process segments sequentially to avoid threading issues
        for i in range(total_segments):
            start_time = i * segment_length
            end_time = min(start_time + segment_length, audio_length)
            if start_time >= end_time:
                self.log(f"Skipping empty segment {start_time}-{end_time}")
                continue

            self.status_label.config(text=f"Status: Transcribing segment {start_time}-{end_time:.2f}s")
            start_index = int(start_time * whisper.audio.SAMPLE_RATE)
            end_index = int(end_time * whisper.audio.SAMPLE_RATE)

            if start_index >= audio.shape[0]:
                self.log(f"Start index {start_index} exceeds audio length {audio.shape[0]}")
                continue
            end_index = min(end_index, audio.shape[0])

            segment_audio = audio[start_index:end_index]
            if segment_audio.size == 0:
                self.log(f"Skipping empty segment {start_time}-{end_time}")
                continue

            # Adjust segment length to match model expectations
            segment_audio = whisper.pad_or_trim(segment_audio)

            # Convert to log-Mel spectrogram
            mel = whisper.log_mel_spectrogram(segment_audio).to(self.model.device)

            # Detect language if not set
            if options['language'] is None:
                _, probs = self.model.detect_language(mel)
                options['language'] = max(probs, key=probs.get)
                self.log(f"Detected language: {options['language']}")

            # Decode the audio
            decode_options = whisper.DecodingOptions(**options)
            result = whisper.decode(self.model, mel, decode_options)
            segments.append({
                'start': start_time,
                'end': end_time,
                'text': result.text
            })
            # Update progress
            progress = (i + 1) / total_segments * 100
            self.update_progress(progress)
            self.progress_label.config(text=f"Progress: {progress:.2f}%")

        # Combine transcriptions
        transcription = ' '.join([s['text'] for s in segments])

        # Save logs to file
        with open(os.path.join(self.output_dir.get(), self.log_file), 'w', encoding='utf-8') as f:
            f.write(self.log_text.get(1.0, tk.END))

        return transcription, segments

    def transcribe_audio_vosk(self, audio_path, duration):
        self.log("Transcribing the audio file with Vosk...")
        segments = []
        transcription = ""

        # Open the audio file
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            self.log("Audio file must be WAV format mono PCM.")
            raise Exception("Audio file must be WAV format mono PCM.")

        rec = KaldiRecognizer(self.model, wf.getframerate())
        rec.SetWords(True)

        # Read the audio in chunks
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if 'text' in res:
                    transcription += res['text'] + ' '
                    segments.append({
                        'start': res.get('start', 0),
                        'end': res.get('end', 0),
                        'text': res['text']
                    })
            else:
                pass  # Partial results can be processed if needed

        # Get the final bits of transcription
        res = json.loads(rec.FinalResult())
        if 'text' in res:
            transcription += res['text']
            segments.append({
                'start': res.get('start', 0),
                'end': res.get('end', 0),
                'text': res['text']
            })

        wf.close()

        # Update progress to 100%
        self.update_progress(100)
        self.progress_label.config(text="Progress: 100%")

        return transcription.strip(), segments

    def update_progress(self, progress):
        self.progress['value'] = progress

    def save_transcription(self, transcription, video_path):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_file = os.path.join(self.output_dir.get(), f"{base_name}_transcription.{self.output_format.get().lower()}")
        if self.output_format.get() == 'Text':
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcription)
        elif self.output_format.get() == 'JSON':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({'transcription': transcription}, f, ensure_ascii=False)
        elif self.output_format.get() == 'CSV':
            import csv
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Transcription'])
                writer.writerow([transcription])
        elif self.output_format.get() == 'DOCX':
            import docx
            doc = docx.Document()
            doc.add_paragraph(transcription)
            doc.save(output_file)
        self.log(f"Transcription saved to {output_file}")

    def save_subtitles_file(self, segments, video_path):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        srt_file = os.path.join(self.output_dir.get(), f"{base_name}_transcription.srt")
        subs = pysrt.SubRipFile()
        for i, segment in enumerate(segments):
            # Create a subtitle item
            item = pysrt.SubRipItem()
            item.index = i + 1
            start_sec = segment.get('start', 0)
            end_sec = segment.get('end', start_sec + 1)
            item.start = pysrt.SubRipTime(seconds=start_sec)
            item.end = pysrt.SubRipTime(seconds=end_sec)
            item.text = segment['text'].strip()
            subs.append(item)
        subs.save(srt_file, encoding='utf-8')
        self.log(f"Subtitles saved to {srt_file}")

    def log(self, message):
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        self.log_text.insert(tk.END, f"{timestamp} {message}\n")
        self.log_text.see(tk.END)
        logging.info(message)

    def update_resource_usage(self):
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        self.resource_label.config(text=f"CPU: {cpu}%  Memory: {memory}%")
        self.root.after(1000, self.update_resource_usage)

    def save_settings(self):
        config = configparser.ConfigParser()
        config['Settings'] = {
            'model_engine': self.model_engine.get(),
            'model_size': self.model_size.get(),
            'vosk_model_path': self.vosk_model_path.get(),
            'language': self.transcription_language.get(),
            'beam_size': str(self.beam_size.get()),
            'best_of': str(self.best_of.get()),
            'temperature': str(self.temperature.get()),
            'segment_length': str(self.segment_length.get()),
            'output_format': self.output_format.get(),
        }
        with open('settings.ini', 'w') as configfile:
            config.write(configfile)

    def load_settings(self):
        self.video_paths = []
        self.video_paths_var = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_engine = tk.StringVar(value='Whisper')
        self.model_size = tk.StringVar(value='base')
        self.vosk_model_path = tk.StringVar()
        self.audio_stream_index = tk.IntVar(value=0)
        self.transcription_language = tk.StringVar(value="Automatic Detection")
        self.save_subtitles = tk.BooleanVar(value=True)
        self.output_format = tk.StringVar(value='Text')
        self.beam_size = tk.IntVar(value=5)
        self.best_of = tk.IntVar(value=5)
        self.temperature = tk.DoubleVar(value=0.0)
        self.segment_length = tk.IntVar(value=30)

        config = configparser.ConfigParser()
        if os.path.exists('settings.ini'):
            config.read('settings.ini')
            self.model_engine.set(config.get('Settings', 'model_engine', fallback='Whisper'))
            self.model_size.set(config.get('Settings', 'model_size', fallback='base'))
            self.vosk_model_path.set(config.get('Settings', 'vosk_model_path', fallback=''))
            self.transcription_language.set(config.get('Settings', 'language', fallback="Automatic Detection"))
            self.beam_size.set(config.getint('Settings', 'beam_size', fallback=5))
            self.best_of.set(config.getint('Settings', 'best_of', fallback=5))
            self.temperature.set(config.getfloat('Settings', 'temperature', fallback=0.0))
            self.segment_length.set(config.getint('Settings', 'segment_length', fallback=30))
            self.output_format.set(config.get('Settings', 'output_format', fallback='Text'))

class ToolTip:
    def __init__(self, widget, text):
        self.waittime = 500     # milliseconds
        self.wraplength = 300   # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.id = None
        self.top = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hide_tooltip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.show_tooltip)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def show_tooltip(self, event=None):
        x = y = 0
        x = self.widget.winfo_pointerx() + 10
        y = self.widget.winfo_pointery() + 10
        self.top = tk.Toplevel(self.widget)
        self.top.wm_overrideredirect(True)
        self.top.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.top, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hide_tooltip(self):
        top = self.top
        self.top = None
        if top:
            top.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
