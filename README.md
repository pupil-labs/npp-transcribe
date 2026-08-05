# Audio Transcription & Subtitles Plugin

Audio Transcription & Subtitles plugin. Leverages Whisper and pyannote.audio for speech recognition and speaker diarization.

This plugin also allows event creation on word boundaries (similar to the Audio Event Annotations) but operates fully locally on your machine and directly within Neon Player. It uses Whisper for speech recognition and pyannote.audio for speaker diarization, enabling you to transcribe audio and identify different speakers in your recordings. The plugin generates a subtitle file that can be used for video playback or further analysis.

## Installation
For instructions on how to install and manage Neon Player plugins, please refer to the [Neon Player Plugin Documentation](https://docs.pupil-labs.com/neon/neon-player/plugin-api/#adding-a-plugin).

## Usage
1. **Transcribe**: Click the **Run Transcription** action to process the recording's audio. The plugin will use local hardware-accelerated Whisper models to transcribe speech and generate subtitle timings.
2. **Subtitles**: Once completed, the plugin will overlay subtitles onto the scene video automatically. You can toggle them, adjust font size, and change colors in the settings.
3. **Diarization**: If you want to identify different speakers, ensure you set your HuggingFace Token (with Pyannote agreement) via the plugin's **Set HF Token** action, enable diarization in the properties, and re-run the transcription.
4. **Events Generation**: Click **Create Word Events** to automatically inject Neon Player timeline events for every single spoken word.
5. **Export**: Click **Export to CSV** to save the transcription, including start/end times and speaker IDs, to a `.csv` file.
