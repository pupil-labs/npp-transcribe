# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mlx-whisper; sys_platform == 'darwin'",
#     "openai-whisper; sys_platform != 'darwin'",
#     "librosa",
#     "numpy",
#     "soundfile",
#     "pyannote-audio",
# ]
#
# [tool.uv.sources]
# pyannote-audio = { git = "https://github.com/pyannote/pyannote-audio.git" }
# ///

"""Audio Transcription & Subtitles Plugin

Algorithms:
- Speech Recognition: mlx-whisper (Whisper large-v3-turbo) on Apple Metal via MLX
- Word Timestamps: Whisper DTW-based forced alignment (word_timestamps=True)
- Speaker Diarization: pyannote.audio speaker-diarization-3.1 pipeline
  (requires a HuggingFace access token with pyannote model agreement)
- Resampling: librosa Kaiser-best resampler (orig_sr -> 16 kHz)
- Timestamp Normalization: first-frame UNIX epoch offset vs recording.start_time
"""

from __future__ import annotations

import enum
import json
import logging
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import keyring
import numpy as np
import soundfile as sf
from pupil_labs.neon_player import Plugin, ProgressUpdate, action
from pupil_labs.neon_player.plugins.events import EventsPlugin
from pyannote.audio import Pipeline
from PySide6.QtCore import QObject, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)
from qt_property_widgets.utilities import (
    PersistentPropertiesMixin,
    action_params,
    property_params,
)
from qt_property_widgets.widgets import (
    BoolWidget,
    ColorWidget,
    Expander,
    PropertyForm,
    PropertyWidget,
)

logger = logging.getLogger(__name__)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("pyannote").setLevel(logging.WARNING)


class WhisperLanguage(enum.Enum):
    Auto = "auto"
    English = "en"
    Spanish = "es"
    French = "fr"
    German = "de"
    Italian = "it"
    Japanese = "ja"
    Chinese = "zh"
    Portuguese = "pt"
    Dutch = "nl"
    Russian = "ru"
    Korean = "ko"
    Arabic = "ar"


class CaptionColors(PersistentPropertiesMixin, QObject):
    """Pair of text and highlight colors."""

    changed = Signal()

    def __init__(
        self,
        text_color: QColor | str = "#FFFFFF",
        highlight_color: QColor | str = "#FFD600",
    ) -> None:
        super().__init__()
        self._text_color = QColor(text_color)
        self._highlight_color = QColor(highlight_color)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CaptionColors):
            return False
        return (
            self._text_color == other._text_color
            and self._highlight_color == other._highlight_color
        )

    @property
    def text_color(self) -> QColor:
        return self._text_color

    @text_color.setter
    def text_color(self, v: QColor | str) -> None:
        self._text_color = QColor(v)
        self.changed.emit()

    @property
    def highlight_color(self) -> QColor:
        return self._highlight_color

    @highlight_color.setter
    def highlight_color(self, v: QColor | str) -> None:
        self._highlight_color = QColor(v)
        self.changed.emit()


class CaptionColorsWidget(PropertyWidget):
    """Places Text Color and Highlight Color widgets side-by-side on the same row."""

    value_changed = Signal(object)

    @staticmethod
    def from_property_impl(prop: property) -> "CaptionColorsWidget":
        return CaptionColorsWidget()

    def __init__(self) -> None:
        super().__init__()
        self._value = CaptionColors()
        self._updating = False

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addStretch(1)
        self.label_text = QLabel("Color")
        self.widget_text = ColorWidget()
        self.widget_text.spacer.setVisible(False)
        self.label_hl = QLabel("Highlight")
        self.widget_hl = ColorWidget()
        self.widget_hl.spacer.setVisible(False)

        layout.addWidget(self.label_text)
        layout.addWidget(self.widget_text)
        layout.addSpacing(10)
        layout.addWidget(self.label_hl)
        layout.addWidget(self.widget_hl)

        self.grid_layout.addLayout(layout, 0, 0)

        self.widget_text.value_changed.connect(self._on_text_changed)
        self.widget_hl.value_changed.connect(self._on_hl_changed)

    def _on_text_changed(self, col: QColor) -> None:
        if self._updating:
            return
        new_val = CaptionColors(col, self._value.highlight_color)
        self._value = new_val
        self.value_changed.emit(new_val)

    def _on_hl_changed(self, col: QColor) -> None:
        if self._updating:
            return
        new_val = CaptionColors(self._value.text_color, col)
        self._value = new_val
        self.value_changed.emit(new_val)

    @property
    def value(self) -> CaptionColors:
        return self._value

    @value.setter
    def value(self, val: CaptionColors | dict) -> None:
        self._updating = True
        try:
            if isinstance(val, dict):
                self._value = CaptionColors(
                    val.get("text_color", "#FFFFFF"),
                    val.get("highlight_color", "#FFD600"),
                )
            elif isinstance(val, CaptionColors):
                self._value = CaptionColors(val.text_color, val.highlight_color)
            self.widget_text.value = self._value.text_color
            self.widget_hl.value = self._value.highlight_color
        finally:
            self._updating = False


class CaptionStyle(PersistentPropertiesMixin, QObject):
    """Caption appearance and style settings."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._font_size = 32
        self._colors = CaptionColors()
        self._colors.changed.connect(self.changed.emit)

    @property
    @property_params(label="Font Size", min=10, max=100)
    def font_size(self) -> int:
        """Subtitle Font Size"""
        return self._font_size

    @font_size.setter
    def font_size(self, v: int) -> None:
        self._font_size = int(v)
        self.changed.emit()

    @property
    @property_params(label="Color", widget=CaptionColorsWidget)
    def colors(self) -> CaptionColors:
        return self._colors

    @colors.setter
    def colors(self, v: CaptionColors | dict) -> None:
        if isinstance(v, dict):
            self._colors.__setstate__(v)
        elif isinstance(v, CaptionColors):
            self._colors = v
            self._colors.changed.connect(self.changed.emit)
        self.changed.emit()

    # Backward compatibility proxies
    @property
    @property_params(widget=None, dont_encode=True)
    def text_color(self) -> QColor:
        return self._colors.text_color

    @text_color.setter
    def text_color(self, v: QColor | str) -> None:
        self._colors.text_color = v

    @property
    @property_params(widget=None, dont_encode=True)
    def highlight_color(self) -> QColor:
        return self._colors.highlight_color

    @highlight_color.setter
    def highlight_color(self, v: QColor | str) -> None:
        self._colors.highlight_color = v


class CollapsibleGroupWidget(PropertyWidget):
    """Renders a custom type inside a collapsible Expander section."""

    value_changed = Signal(object)

    @staticmethod
    def from_property_impl(prop: property) -> "CollapsibleGroupWidget":
        target_type = None
        if hasattr(prop, "fget") and prop.fget:
            try:
                import typing as T
                target_type = T.get_type_hints(prop.fget).get("return")
            except Exception:
                pass
            if target_type is None or isinstance(target_type, str):
                hints = getattr(prop.fget, "__annotations__", {})
                ret = hints.get("return")
                if isinstance(ret, type):
                    target_type = ret
                elif isinstance(ret, str) and hasattr(prop.fget, "__globals__"):
                    target_type = prop.fget.__globals__.get(ret, object)
        if target_type is None or not isinstance(target_type, type):
            target_type = object

        params = getattr(prop.fget, "parameters", {}) if hasattr(prop, "fget") and prop.fget else {}
        title = params.get("label") or (prop.fget.__name__.replace("_", " ").title() if hasattr(prop, "fget") and prop.fget else "")
        return CollapsibleGroupWidget(target_type, title=title)

    def __init__(self, target_type: type, title: str = "", expanded: bool = False) -> None:
        super().__init__()
        self.form = PropertyForm.from_type(target_type)
        self.expander = Expander(title=title, content_widget=self.form, expanded=expanded)
        self.expander.layout().setContentsMargins(0, 0, 0, 0)
        self.setObjectName("CollapsibleGroupWidget")
        self.setStyleSheet("""
            CollapsibleGroupWidget Expander {
                padding-top: 0px; 
                padding-bottom: 0px;
            }

            CollapsibleGroupWidget Expander Expander {
                padding-top: 0px; 
                padding-bottom: 0px;
            }

            CollapsibleGroupWidget QLabel#ExpanderName {
                color: #a09fa6;
            }
        """)
        self.grid_layout.addWidget(self.expander, 0, 0)
        self.form.value_changed.connect(self.value_changed.emit)
        self.form.property_changed.connect(lambda n, v: self.value_changed.emit(self.value))

    def on_bind(self, instance: Any) -> None:
        super().on_bind(instance)
        enable_prop = self.source_params.get("enable_prop")
        if enable_prop and hasattr(instance, enable_prop):
            self.bool_widget = BoolWidget()
            self.bool_widget.spacer.setVisible(False)
            self.bool_widget.value = bool(getattr(instance, enable_prop))
            self.bool_widget.value_changed.connect(
                lambda val: setattr(instance, enable_prop, val)
            )
            if hasattr(instance, "changed") and hasattr(instance.changed, "connect"):
                instance.changed.connect(
                    lambda: setattr(
                        self.bool_widget,
                        "value",
                        bool(getattr(instance, enable_prop)),
                    )
                    if hasattr(self, "bool_widget")
                    and self.bool_widget.value
                    != bool(getattr(instance, enable_prop))
                    else None
                )
            self.expander.controls_layout.addWidget(self.bool_widget)

    @property
    def value(self) -> object:
        return self.form.value

    @value.setter
    def value(self, v: object) -> None:
        self.form.value = v


_SPEAKER_COLORS = [
    QColor("#4FC3F7"),
    QColor("#81C784"),
    QColor("#FFB74D"),
    QColor("#F06292"),
    QColor("#CE93D8"),
    QColor("#80DEEA"),
]


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def _compute_audio_time_offset(first_ts: float, rec_start_ns: int) -> float:
    """Normalize a raw frame timestamp to seconds relative to the recording start."""
    rec_start_s = rec_start_ns / 1e9
    if first_ts > 1e17:
        return (first_ts - rec_start_ns) / 1e9
    elif first_ts > 1e8:
        return first_ts - rec_start_s
    else:
        return float(first_ts)


class TranscribePlugin(Plugin):
    label = "Audio Transcription & Subtitles"

    def __init__(self) -> None:
        super().__init__()
        self.render_layer = 100
        self._words: list[dict[str, Any]] = []
        self._segments: list[dict[str, Any]] = []
        self._model = "mlx-community/whisper-large-v3-turbo"
        self._audio_stream_idx = 0
        self._language = WhisperLanguage.Auto
        self._diarize = False
        self._enabled = True
        self._caption_style = CaptionStyle()
        self._caption_style.changed.connect(self.changed.emit)
        self._bg_color = QColor(0, 0, 0, 150)

    def on_recording_loaded(self, recording: Any) -> None:
        self._words = []
        self._segments = []
        data = self.load_cached_json("transcription.json")
        if data and isinstance(data, dict):
            self._words = data.get("words", [])
            self._segments = data.get("segments", [])

    @property
    @property_params(min=0, max=7)
    def audio_stream_idx(self) -> int:
        """Audio Channel Index"""
        return self._audio_stream_idx

    @audio_stream_idx.setter
    def audio_stream_idx(self, v: int):
        self._audio_stream_idx = v
        self.changed.emit()

    @property
    def language(self) -> WhisperLanguage:
        """Transcription Language"""
        return self._language

    @language.setter
    def language(self, v: WhisperLanguage):
        self._language = v
        self.changed.emit()

    @property
    def diarize(self) -> bool:
        """Speaker Diarization (requires HF token)"""
        return self._diarize

    @diarize.setter
    def diarize(self, v: bool) -> None:
        self._diarize = v
        self.changed.emit()

    @property
    @property_params(widget=None)
    def enabled(self) -> bool:
        """Display Captions"""
        return self._enabled

    @enabled.setter
    def enabled(self, v: bool):
        self._enabled = v
        self.changed.emit()

    @property
    @property_params(
        primary=True,
        widget=CollapsibleGroupWidget,
        label="Display Captions",
        enable_prop="enabled",
    )
    def caption_style(self) -> CaptionStyle:
        return self._caption_style

    @caption_style.setter
    def caption_style(self, value: CaptionStyle | dict) -> None:
        if isinstance(value, dict):
            self._caption_style.__setstate__(value)
        elif isinstance(value, CaptionStyle):
            self._caption_style = value
            self._caption_style.changed.connect(self.changed.emit)
        self.changed.emit()

    # Backward compatibility proxies
    @property
    @property_params(widget=None, dont_encode=True)
    def font_size(self) -> int:
        return self._caption_style.font_size

    @font_size.setter
    def font_size(self, v: int) -> None:
        self._caption_style.font_size = int(v)

    @property
    @property_params(widget=None, dont_encode=True)
    def text_color(self) -> QColor:
        return self._caption_style.text_color

    @text_color.setter
    def text_color(self, v: QColor) -> None:
        self._caption_style.text_color = QColor(v)

    @property
    @property_params(widget=None, dont_encode=True)
    def highlight_color(self) -> QColor:
        return self._caption_style.highlight_color

    @highlight_color.setter
    def highlight_color(self, v: QColor) -> None:
        self._caption_style.highlight_color = QColor(v)

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("dialog-password"))
    def set_hf_token(self) -> None:
        """Set HuggingFace Token"""
        token, ok = QInputDialog.getText(
            None,
            "Set HuggingFace Token",
            "Enter your HuggingFace Token (for Pyannote Diarization):",
            QLineEdit.EchoMode.Password,
        )
        if ok and token:
            keyring.set_password("neon_player", "HF_TOKEN", token.strip())
            self.app.show_notification(
                "Success", "HuggingFace token saved securely to keyring."
            )

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("media-playback-start"))
    def transcribe_audio(self) -> None:
        """Transcribe Audio (Whisper Turbo + Word Timestamps)"""
        cache_path = self.get_cache_path()
        if cache_path is not None:
            (cache_path / "transcription.json").unlink(missing_ok=True)

        self.job = self.job_manager.run_background_action(
            "Transcribing Audio", "TranscribePlugin.bg_transcribe"
        )
        self.job.finished.connect(self._on_transcribed)

    def _on_transcribed(self) -> None:
        self.on_recording_loaded(self.recording)
        self.changed.emit()
        if self._words:
            self.app.show_notification(
                "Transcription", f"Transcribed {len(self._words)} words."
            )

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("edit-find"))
    def create_events_from_keyword(self) -> None:
        """Create Events from Keyword"""
        if not self._words:
            self.app.show_notification("Notice", "Please transcribe the audio first.")
            return

        keyword, ok = QInputDialog.getText(
            None, "Keyword Search", "Enter keyword to search for in transcription:"
        )
        if not ok or not keyword:
            return

        keyword = keyword.lower().strip()
        events_plugin: EventsPlugin = Plugin.get_instance_by_name("EventsPlugin")
        if not events_plugin:
            self.app.show_notification("Error", "EventsPlugin not loaded.")
            return

        event_name = f"Spoke: {keyword}"

        rec_start = self.recording.start_time
        timestamps = []
        for w in self._words:
            if keyword in w.get("word", "").lower():
                ts_ns = rec_start + int(w["start"] * 1e9)
                timestamps.append(ts_ns)

        if timestamps:
            events_plugin.add_events({event_name: timestamps})

        self.app.show_notification(
            "Events Created", f"Added {len(timestamps)} events for '{keyword}'."
        )

    @action
    @action_params(compact=True, icon=QIcon.fromTheme("text-x-generic"))
    def view_transcript(self) -> None:
        """View Full Transcript"""
        if not self._words:
            self.app.show_notification("Notice", "Please transcribe the audio first.")
            return

        dialog = QDialog()
        dialog.setWindowTitle("Full Transcript")
        dialog.resize(700, 500)

        layout = QVBoxLayout()
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setFontFamily("Courier New")

        lines = []
        current_speaker = None
        line_words = []

        for w in self._words:
            speaker = w.get("speaker")
            start = w["start"]
            word = w.get("word", "").strip()

            if speaker != current_speaker:
                if line_words:
                    lines.append(" ".join(line_words))
                current_speaker = speaker
                prefix = f"\n[{_format_time(start)}]"
                if speaker:
                    prefix += f" {speaker}:"
                line_words = [prefix, word]
            else:
                line_words.append(word)

        if line_words:
            lines.append(" ".join(line_words))

        text_edit.setPlainText("\n".join(lines).strip())
        layout.addWidget(text_edit)

        btn = QPushButton("Close")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)

        dialog.setLayout(layout)
        dialog.exec()

    def _extract_audio_to_file(self) -> tuple[str, float, float]:
        """Extract audio stream, convert to mono float32, compute offset, and save to a temporary WAV file to save RAM."""
        if not hasattr(self.recording, "audio") or self.recording.audio is None:
            raise RuntimeError("No audio stream found in recording")

        orig_sr = 48000.0
        if hasattr(self.recording.audio, "rate"):
            orig_sr = float(self.recording.audio.rate)
        elif hasattr(self.recording.audio, "average_rate"):
            orig_sr = float(self.recording.audio.average_rate)

        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        first_ts = None
        sf_writer = None

        try:
            for frame in self.recording.audio:
                if first_ts is None:
                    first_ts = getattr(frame, "time", getattr(frame, "ts", 0.0))

                frame_np = frame.to_ndarray()

                if len(frame_np.shape) > 1:
                    if frame_np.shape[0] < frame_np.shape[1]:
                        # (channels, samples) -> (samples, channels)
                        frame_np = frame_np.T

                    if self._audio_stream_idx < frame_np.shape[1]:
                        mono_chunk = frame_np[:, self._audio_stream_idx]
                    else:
                        mono_chunk = frame_np[:, 0]
                else:
                    mono_chunk = frame_np

                mono_chunk = mono_chunk.astype(np.float32)
                max_val = np.max(np.abs(mono_chunk))
                if max_val > 1.0:
                    mono_chunk /= max_val

                if sf_writer is None:
                    sf_writer = sf.SoundFile(
                        tmp_path, mode="w", samplerate=int(orig_sr), channels=1
                    )

                sf_writer.write(mono_chunk)

        finally:
            if sf_writer is not None:
                sf_writer.close()

        if first_ts is None:
            raise RuntimeError("Failed to decode audio (no frames)")

        audio_time_offset = _compute_audio_time_offset(
            first_ts, self.recording.start_time
        )

        return tmp_path, orig_sr, audio_time_offset

    def bg_transcribe(self) -> Generator[ProgressUpdate, None, None]:
        yield ProgressUpdate(0.0, "Extracting audio via pl-neon-recording...")

        audio_path = None
        try:
            audio_path, orig_sr, audio_time_offset = self._extract_audio_to_file()

            lang_val = (
                None if self._language == WhisperLanguage.Auto else self._language.value
            )

            if sys.platform == "darwin":
                yield ProgressUpdate(0.2, "Transcribing with Whisper Turbo (Metal)...")
                import mlx_whisper

                result = mlx_whisper.transcribe(
                    audio_path,
                    path_or_hf_repo=self._model,
                    word_timestamps=True,
                    language=lang_val,
                )
            else:
                yield ProgressUpdate(0.2, "Loading Whisper Turbo model (PyTorch)...")
                import whisper

                model = whisper.load_model("turbo")
                yield ProgressUpdate(0.3, "Transcribing with standard Whisper...")
                result = model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    language=lang_val,
                )

            segments = result.get("segments", [])

            words: list[dict[str, Any]] = []
            for seg in segments:
                words.extend(
                    {
                        "word": w.get("word", ""),
                        "start": w["start"] + audio_time_offset,
                        "end": w["end"] + audio_time_offset,
                    }
                    for w in seg.get("words", [])
                )
                seg["start"] += audio_time_offset
                seg["end"] += audio_time_offset

            if self._diarize:
                hf_token = keyring.get_password("neon_player", "HF_TOKEN")
                if not hf_token:
                    logger.error(
                        "HuggingFace token not found in keyring. Skipping diarization."
                    )
                    self.app.show_notification(
                        "Notice", "HuggingFace token missing. Skipping diarization."
                    )
                else:
                    yield ProgressUpdate(0.7, "Running speaker diarization...")
                    words = self._run_diarization(audio_path, words, hf_token)

            yield ProgressUpdate(0.9, "Saving cache...")
            self.save_cached_json("transcription.json", {"words": words, "segments": segments})

            yield ProgressUpdate(1.0, "Done")

        except Exception:
            logger.exception("Failed during transcription process")
            raise
        finally:
            # Free large ML models and empty GPU memory caches
            if audio_path is not None:
                Path(audio_path).unlink(missing_ok=True)

            if "model" in locals():
                del model
            if "result" in locals():
                del result

            import gc

            gc.collect()

            if "torch" in sys.modules:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            if "mlx" in sys.modules:
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except Exception:
                    pass

    def _run_diarization(
        self, audio_path: str, words: list[dict[str, Any]], hf_token: str
    ) -> list[dict[str, Any]]:
        """Runs pyannote/speaker-diarization-3.1.

        Assigns a speaker label to each word by finding the diarization
        segment whose time range contains the word midpoint (greedy nearest match).
        """
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

        diarization = pipeline(audio_path)

        dia_segments = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        for w in words:
            mid = (w["start"] + w["end"]) / 2
            for seg_start, seg_end, speaker in dia_segments:
                if seg_start <= mid <= seg_end:
                    w["speaker"] = speaker
                    break

        del pipeline  # Explicitly release the pyannote model reference
        return words

    def render(self, painter: QPainter, time_ns: int) -> None:
        if not self._enabled or not self._words or not self.recording:
            return

        rec_start = self.recording.start_time
        time_s = (time_ns - rec_start) / 1e9

        CONTEXT_BEFORE = 3
        CONTEXT_AFTER = 5

        active_idx = None
        for i, w in enumerate(self._words):
            if w["start"] <= time_s <= w["end"]:
                active_idx = i
                break

        if active_idx is None:
            return

        window_start = max(0, active_idx - CONTEXT_BEFORE)
        window = self._words[window_start : active_idx + CONTEXT_AFTER]
        active_word = self._words[active_idx].get("word", "").strip()
        active_in_window = active_idx - window_start

        font = QFont("Arial", self.font_size, QFont.Weight.Normal)
        painter.setFont(font)
        fm = painter.fontMetrics()

        scene_width = self.recording.scene.width
        scene_height = self.recording.scene.height

        tokens = [w.get("word", "").strip() for w in window]
        display_text = " ".join(tokens)

        flags = (
            Qt.TextFlag.TextWordWrap
            | Qt.AlignmentFlag.AlignHCenter
            | Qt.AlignmentFlag.AlignBottom
        )

        rect_w = scene_width * 0.85
        rect_x = (scene_width - rect_w) / 2
        rect_y = scene_height * 0.75
        rect_h = scene_height * 0.20

        rect = QRectF(rect_x, rect_y, rect_w, rect_h)
        bounding_rect = fm.boundingRect(rect.toRect(), flags, display_text)

        bg_rect = QRectF(
            bounding_rect.x() - 15,
            bounding_rect.y() - 10,
            bounding_rect.width() + 30,
            bounding_rect.height() + 20,
        )
        painter.fillRect(bg_rect, self._bg_color)

        speaker = self._words[active_idx].get("speaker")
        if speaker:
            speaker_idx = (
                int(speaker.split("_")[-1]) if "_" in speaker else abs(hash(speaker))
            )
            text_col = _SPEAKER_COLORS[speaker_idx % len(_SPEAKER_COLORS)]
        else:
            text_col = self.text_color

        painter.setPen(QPen(text_col))
        painter.drawText(bounding_rect, flags, display_text)

        before_text = " ".join(tokens[:active_in_window])
        if before_text:
            before_text += " "
        before_w = fm.horizontalAdvance(before_text)
        word_w = fm.horizontalAdvance(active_word)

        highlight_rect = QRectF(
            bounding_rect.x() + before_w,
            bounding_rect.y(),
            word_w,
            bounding_rect.height(),
        )
        painter.setPen(QPen(self.highlight_color))
        painter.drawText(
            highlight_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            active_word,
        )
