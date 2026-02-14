"""
M4B Wizard GUI: combine MP3s, set chapter breaks (Whisper or one-per-file), edit names and metadata, publish.
Run: python m4b_wizard.py
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QSettings, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QGroupBox,
    QButtonGroup,
    QScrollArea,
    QFrame,
)
from PySide6.QtGui import QTextCursor, QTextDocument

# Backend
from combine_to_m4b import (
    concat_mp3s_from_list,
    transcribe_with_whisper,
    propose_chapter_breaks,
    chapters_from_breaks,
    get_audio_duration_sec,
    write_ffmetadata,
    create_m4b,
)


def load_metadata_from_mp3(path: Path) -> dict:
    """Read ID3 tags from an MP3; return dict with title, artist, album, genre, comment."""
    result = {"title": "", "artist": "", "album": "", "genre": "", "comment": ""}
    try:
        from mutagen.mp3 import MP3
        audio = MP3(str(path))
        if audio.tags is None:
            return result
        tags = audio.tags
        def text(frame_id: str) -> str:
            frames = tags.getall(frame_id)
            if not frames:
                return ""
            v = frames[0]
            if hasattr(v, "text") and v.text:
                return str(v.text[0]).strip()
            return ""
        def comment_text() -> str:
            frames = tags.getall("COMM")
            if not frames:
                return ""
            v = frames[0]
            if hasattr(v, "text") and v.text:
                return str(v.text[0]).strip()
            return ""
        result["title"] = text("TIT2")
        result["artist"] = text("TPE1")
        result["album"] = text("TALB")
        result["genre"] = text("TCON")
        result["comment"] = comment_text()
    except Exception:
        pass
    return result


@dataclass
class ProjectState:
    """In-memory project for the wizard."""
    file_paths: list[Path] = field(default_factory=list)
    chapter_mode: str = "one_per_file"  # "one_per_file" | "whisper"
    total_dur_sec: float = 0.0
    segments: list[dict] = field(default_factory=list)  # Whisper segments
    break_times: list[float] = field(default_factory=list)  # sorted chapter break times (start of each new chapter)
    proposed_titles: list[str] = field(default_factory=list)  # from Whisper or file names
    chapter_titles: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=lambda: {"title": "", "artist": "", "album": "", "genre": "", "comment": ""})
    combined_m4a_path: Path | None = None
    combined_wav_path: Path | None = None
    temp_dir: tempfile.TemporaryDirectory | None = None


def format_time(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}:{s:02d}"


# ---- Worker threads ----
class ConcatWorker(QThread):
    finished = Signal(bool)
    error = Signal(str)

    def __init__(self, file_paths: list[Path], out_m4a: Path, out_wav: Path):
        super().__init__()
        self.file_paths = file_paths
        self.out_m4a = out_m4a
        self.out_wav = out_wav

    def run(self):
        try:
            ok = concat_mp3s_from_list(self.file_paths, self.out_m4a, self.out_wav)
            self.finished.emit(ok)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False)


class TranscribeWorker(QThread):
    finished = Signal(list)  # segments
    error = Signal(str)

    def __init__(self, wav_path: Path, model_name: str = "base", language: str = "en", device: str | None = None):
        super().__init__()
        self.wav_path = wav_path
        self.model_name = model_name
        self.language = language
        self.device = device

    def run(self):
        try:
            segments = transcribe_with_whisper(
                self.wav_path,
                model_name=self.model_name,
                language=self.language,
                device=self.device,
            )
            self.finished.emit(segments)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit([])


class PublishWorker(QThread):
    progress = Signal(str)
    finished = Signal(bool)
    error = Signal(str)

    def __init__(
        self,
        file_paths: list[Path],
        break_times: list[float],
        total_dur_sec: float,
        chapter_titles: list[str],
        metadata: dict,
        out_path: Path,
    ):
        super().__init__()
        self.file_paths = file_paths
        self.break_times = break_times
        self.total_dur_sec = total_dur_sec
        self.chapter_titles = chapter_titles
        self.metadata = metadata
        self.out_path = out_path
        self._temp_dir = None

    def run(self):
        try:
            self._temp_dir = tempfile.TemporaryDirectory()
            tmp = Path(self._temp_dir.name)
            out_m4a = tmp / "combined.m4a"
            out_wav = tmp / "combined.wav"
            meta_path = tmp / "chapters.txt"

            self.progress.emit("Concatenating audio...")
            if not concat_mp3s_from_list(self.file_paths, out_m4a, out_wav):
                self.error.emit("FFmpeg concat failed")
                self.finished.emit(False)
                return

            self.progress.emit("Writing chapters and metadata...")
            meta = dict(self.metadata)
            if not meta.get("title"):
                meta["title"] = "Audiobook"
            if not meta.get("artist"):
                meta["artist"] = ""
            chapters = chapters_from_breaks(
                self.break_times,
                self.total_dur_sec,
                titles=self.chapter_titles,
            )
            write_ffmetadata(
                chapters,
                meta.get("title", ""),
                meta.get("artist", ""),
                meta_path,
                metadata=meta,
            )

            self.progress.emit("Creating M4B...")
            if not create_m4b(out_m4a, meta_path, self.out_path, metadata=meta):
                self.error.emit("FFmpeg M4B creation failed")
                self.finished.emit(False)
                return

            self.finished.emit(True)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False)


# ---- Step widgets ----
class Step1Files(QWidget):
    """Select files and chapter mode."""
    def __init__(self, state: ProjectState, parent=None, on_file_list_changed=None):
        super().__init__(parent)
        self.state = state
        self.on_file_list_changed = on_file_list_changed
        layout = QVBoxLayout(self)

        list_label = QLabel("MP3 files (order = playback order):")
        layout.addWidget(list_label)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.file_list)

        btn_layout = QHBoxLayout()
        add_files_btn = QPushButton("Add files...")
        add_files_btn.clicked.connect(self._add_files)
        add_folder_btn = QPushButton("Add folder...")
        add_folder_btn.clicked.connect(self._add_folder)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove)
        move_up_btn = QPushButton("Move up")
        move_up_btn.clicked.connect(self._move_up)
        move_down_btn = QPushButton("Move down")
        move_down_btn.clicked.connect(self._move_down)
        btn_layout.addWidget(add_files_btn)
        btn_layout.addWidget(add_folder_btn)
        btn_layout.addWidget(remove_btn)
        btn_layout.addWidget(move_up_btn)
        btn_layout.addWidget(move_down_btn)
        layout.addLayout(btn_layout)

        mode_group = QGroupBox("Chapter mode")
        mode_layout = QVBoxLayout(mode_group)
        self.radio_one_per_file = QRadioButton("One chapter per file")
        self.radio_one_per_file.setChecked(True)
        self.radio_whisper = QRadioButton("Use Whisper to find chapter breaks")
        mode_layout.addWidget(self.radio_one_per_file)
        mode_layout.addWidget(self.radio_whisper)
        layout.addWidget(mode_group)

        self._refresh_list()

    def _refresh_list(self):
        self.file_list.clear()
        for p in self.state.file_paths:
            self.file_list.addItem(p.name)

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select MP3 files", "", "MP3 (*.mp3);;All (*)"
        )
        for p in paths:
            path = Path(p)
            if path not in self.state.file_paths:
                self.state.file_paths.append(path)
        self._refresh_list()
        if self.on_file_list_changed:
            self.on_file_list_changed()

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with MP3s")
        if not folder:
            return
        folder_path = Path(folder)
        for f in sorted(folder_path.glob("*.mp3")):
            if f not in self.state.file_paths:
                self.state.file_paths.append(f)
        self._refresh_list()
        if self.on_file_list_changed:
            self.on_file_list_changed()

    def _remove(self):
        rows = sorted({i.row() for i in self.file_list.selectedIndexes()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.state.file_paths):
                self.state.file_paths.pop(r)
        self._refresh_list()
        if self.on_file_list_changed:
            self.on_file_list_changed()

    def _move_up(self):
        rows = sorted({i.row() for i in self.file_list.selectedIndexes()})
        if not rows or rows[0] == 0:
            return
        for r in rows:
            if r > 0:
                self.state.file_paths[r], self.state.file_paths[r - 1] = (
                    self.state.file_paths[r - 1],
                    self.state.file_paths[r],
                )
        self._refresh_list()
        for r in rows:
            if r > 0:
                self.file_list.setCurrentRow(r - 1)

    def _move_down(self):
        rows = sorted({i.row() for i in self.file_list.selectedIndexes()}, reverse=True)
        n = len(self.state.file_paths)
        if not rows or rows[0] >= n - 1:
            return
        for r in rows:
            if r < n - 1:
                self.state.file_paths[r], self.state.file_paths[r + 1] = (
                    self.state.file_paths[r + 1],
                    self.state.file_paths[r],
                )
        self._refresh_list()
        for r in rows:
            if r < n - 1:
                self.file_list.setCurrentRow(r + 1)

    def save_to_state(self):
        self.state.chapter_mode = "whisper" if self.radio_whisper.isChecked() else "one_per_file"

    def is_valid(self) -> bool:
        return len(self.state.file_paths) > 0


class Step2Breaks(QWidget):
    """Whisper transcript and chapter break editing; or one-per-file summary."""
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        self.layout = QVBoxLayout(self)
        self.status_label = QLabel("")
        self.layout.addWidget(self.status_label)
        self.stack = QStackedWidget()
        self.layout.addWidget(self.stack)

        self.one_per_file_page = QWidget()
        one_layout = QVBoxLayout(self.one_per_file_page)
        one_layout.addWidget(QLabel("Chapter mode: one per file. Break times will be set from file durations when you continue."))
        self.stack.addWidget(self.one_per_file_page)

        self.whisper_page = QWidget()
        wh_layout = QVBoxLayout(self.whisper_page)
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search transcript:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Type to search...")
        self.search_edit.textChanged.connect(self._on_search)
        self.search_edit.returnPressed.connect(self._find_next)
        search_layout.addWidget(self.search_edit)
        prev_btn = QPushButton("Previous")
        prev_btn.clicked.connect(self._find_previous)
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self._find_next)
        search_layout.addWidget(prev_btn)
        search_layout.addWidget(next_btn)
        wh_layout.addLayout(search_layout)
        self.transcript_edit = QPlainTextEdit()
        self.transcript_edit.setReadOnly(True)
        self.transcript_edit.setPlaceholderText("Transcript will appear after analysis.")
        wh_layout.addWidget(QLabel("Transcript (click a line and use buttons to add/remove chapter break):"))
        wh_layout.addWidget(self.transcript_edit)
        btn_row = QHBoxLayout()
        self.add_break_btn = QPushButton("Add break at selected segment")
        self.add_break_btn.clicked.connect(self._add_break_at_selection)
        self.remove_break_btn = QPushButton("Remove selected break")
        self.remove_break_btn.clicked.connect(self._remove_break)
        btn_row.addWidget(self.add_break_btn)
        btn_row.addWidget(self.remove_break_btn)
        wh_layout.addLayout(btn_row)
        wh_layout.addWidget(QLabel("Chapter breaks (time, title):"))
        self.breaks_list = QListWidget()
        wh_layout.addWidget(self.breaks_list)
        self.stack.addWidget(self.whisper_page)

    def _on_search(self):
        """On search box text change: jump to first match."""
        text = self.search_edit.text().strip()
        if not text:
            return
        cursor = self.transcript_edit.document().find(text)
        if not cursor.isNull():
            self.transcript_edit.setTextCursor(cursor)
            self.transcript_edit.centerCursor()

    def _find_next(self):
        """Move to next occurrence of search term (wraps from end to start)."""
        text = self.search_edit.text().strip()
        if not text:
            return
        doc = self.transcript_edit.document()
        cursor = self.transcript_edit.textCursor()
        start = cursor.position() + 1
        found = doc.find(text, start)
        if found.isNull():
            found = doc.find(text, 0)
        if not found.isNull():
            self.transcript_edit.setTextCursor(found)
            self.transcript_edit.centerCursor()

    def _find_previous(self):
        """Move to previous occurrence of search term (wraps from start to end)."""
        text = self.search_edit.text().strip()
        if not text:
            return
        doc = self.transcript_edit.document()
        cursor = self.transcript_edit.textCursor()
        start = max(0, cursor.selectionStart() - 1)
        found = doc.find(text, start, QTextDocument.FindFlag.FindBackward)
        if found.isNull():
            found = doc.find(text, doc.characterCount() - 1, QTextDocument.FindFlag.FindBackward)
        if not found.isNull():
            self.transcript_edit.setTextCursor(found)
            self.transcript_edit.centerCursor()

    def _segment_at_cursor(self) -> dict | None:
        """Return segment at current cursor position (by line)."""
        cursor = self.transcript_edit.textCursor()
        block = cursor.block()
        line_num = block.blockNumber()
        if 0 <= line_num < len(self.state.segments):
            return self.state.segments[line_num]
        return None

    def _add_break_at_selection(self):
        seg = self._segment_at_cursor()
        if seg is None:
            return
        t = float(seg.get("start", 0))
        if t in self.state.break_times:
            return
        self.state.break_times.append(t)
        self.state.break_times.sort()
        title = (seg.get("text") or "").strip()[:40] or f"{format_time(t)}"
        self.state.proposed_titles = self._titles_from_breaks()
        self._refresh_breaks_list()

    def _titles_from_breaks(self) -> list[str]:
        titles = []
        for i, t in enumerate(sorted(self.state.break_times)):
            # find segment that starts at t for title
            for seg in self.state.segments:
                if abs(float(seg.get("start", 0)) - t) < 0.5:
                    titles.append((seg.get("text") or "").strip()[:80] or f"Chapter {i + 1}")
                    break
            else:
                titles.append(f"Chapter {i + 1}")
        titles.append(f"Chapter {len(self.state.break_times) + 1}")
        return titles

    def _remove_break(self):
        row = self.breaks_list.currentRow()
        if 0 <= row < len(self.state.break_times):
            self.state.break_times.pop(row)
            self.state.break_times.sort()
            self.state.proposed_titles = self._titles_from_breaks()
            self._refresh_breaks_list()

    def _refresh_breaks_list(self):
        self.breaks_list.clear()
        for i, t in enumerate(sorted(self.state.break_times)):
            title = self.state.proposed_titles[i] if i < len(self.state.proposed_titles) else f"Chapter {i + 1}"
            self.breaks_list.addItem(f"{format_time(t)} — {title}")

    def show_whisper_ui(self, segments: list, total_dur: float):
        self.state.segments = segments
        self.state.total_dur_sec = total_dur
        proposed = propose_chapter_breaks(segments)
        self.state.break_times = [p[0] for p in proposed]
        self.state.proposed_titles = [p[1] for p in proposed]
        if self.state.break_times:
            self.state.proposed_titles.append(f"Chapter {len(self.state.break_times) + 1}")
        lines = []
        for s in segments:
            t = s.get("start", 0)
            text = (s.get("text") or "").strip()
            lines.append(f"[{format_time(float(t))}] {text}")
        self.transcript_edit.setPlainText("\n".join(lines))
        self._refresh_breaks_list()
        self.stack.setCurrentWidget(self.whisper_page)

    def show_one_per_file(self, break_times: list[float], total_dur: float, titles: list[str]):
        self.state.break_times = break_times
        self.state.total_dur_sec = total_dur
        self.state.proposed_titles = titles
        self.stack.setCurrentWidget(self.one_per_file_page)

    def is_valid(self) -> bool:
        return True


class Step3Names(QWidget):
    """Edit chapter names."""
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Set a name for each chapter:"))
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["#", "Start", "Title"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

    def refresh_from_state(self):
        times = sorted(self.state.break_times)
        total = self.state.total_dur_sec
        n_chapters = len(times) + 1
        titles = self.state.chapter_titles if self.state.chapter_titles else self.state.proposed_titles
        if len(titles) < n_chapters:
            titles = list(titles) + [f"Chapter {i + 1}" for i in range(len(titles), n_chapters)]
        self.table.setRowCount(n_chapters)
        prev = 0.0
        for i in range(n_chapters):
            end = times[i] if i < len(times) else total
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(format_time(prev)))
            title_item = QTableWidgetItem(titles[i] if i < len(titles) else f"Chapter {i + 1}")
            self.table.setItem(i, 2, title_item)
            prev = end
        self.state.chapter_titles = [self.table.item(i, 2).text() for i in range(n_chapters)]

    def save_to_state(self):
        self.state.chapter_titles = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 2)
            self.state.chapter_titles.append(item.text() if item else f"Chapter {i + 1}")

    def is_valid(self) -> bool:
        return True


class Step4Metadata(QWidget):
    """Title, author, album, genre, comment."""
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)
        self.fields = {}
        for key, label in [
            ("title", "Title (book):"),
            ("artist", "Author / Artist:"),
            ("album", "Album:"),
            ("genre", "Genre:"),
            ("comment", "Comment:"),
        ]:
            layout.addWidget(QLabel(label))
            edit = QLineEdit()
            edit.setText(self.state.metadata.get(key, ""))
            layout.addWidget(edit)
            self.fields[key] = edit
        load_btn = QPushButton("Load from first file")
        load_btn.clicked.connect(self._load_from_first_file)
        layout.addWidget(load_btn)
        layout.addStretch()

    def refresh_from_state(self):
        """Sync form fields from state.metadata."""
        for k, edit in self.fields.items():
            edit.setText(self.state.metadata.get(k, ""))

    def _load_from_first_file(self):
        if not self.state.file_paths:
            QMessageBox.information(self, "Metadata", "No files selected.")
            return
        meta = load_metadata_from_mp3(self.state.file_paths[0])
        for k, v in meta.items():
            if k in self.state.metadata:
                self.state.metadata[k] = v or self.state.metadata.get(k, "")
        self.refresh_from_state()

    def save_to_state(self):
        for k, edit in self.fields.items():
            self.state.metadata[k] = edit.text().strip()

    def is_valid(self) -> bool:
        return True


class Step5Publish(QWidget):
    """Destination and Publish button."""
    def __init__(self, state: ProjectState, parent=None):
        super().__init__(parent)
        self.state = state
        layout = QVBoxLayout(self)
        self.summary_label = QLabel("")
        layout.addWidget(self.summary_label)
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Save to:"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Choose destination...")
        path_layout.addWidget(self.path_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)
        self.publish_btn = QPushButton("Publish")
        self.publish_btn.clicked.connect(self._on_publish)
        layout.addWidget(self.publish_btn)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("")
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)

    def _browse(self):
        title = self.state.metadata.get("title") or "Audiobook"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save M4B as", f"{title}.m4b", "M4B (*.m4b);;All (*)"
        )
        if path:
            self.path_edit.setText(path)

    def refresh_summary(self):
        self.summary_label.setText(
            f"Files: {len(self.state.file_paths)} · Chapters: {len(self.state.break_times) + 1}\n"
            f"Title: {self.state.metadata.get('title') or '(not set)'}"
        )
        if not self.path_edit.text().strip() and self.state.metadata.get("title"):
            self.path_edit.setPlaceholderText(f"{self.state.metadata['title']}.m4b")

    def _on_publish(self):
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Publish", "Choose a destination path.")
            return
        path = Path(path)
        if path.suffix.lower() != ".m4b":
            path = path.with_suffix(".m4b")
        self.path_edit.setText(str(path))
        self.publish_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Starting...")
        mw = self.window()
        if hasattr(mw, "set_background_busy"):
            mw.set_background_busy(True, "Publishing M4B...")
        self._worker = PublishWorker(
            self.state.file_paths,
            list(self.state.break_times),
            self.state.total_dur_sec,
            list(self.state.chapter_titles),
            dict(self.state.metadata),
            path,
        )
        self._worker.progress.connect(self.progress_label.setText)
        self._worker.finished.connect(self._publish_finished)
        self._worker.error.connect(self._publish_error)
        self._worker.start()

    def _publish_finished(self, ok: bool):
        mw = self.window()
        if hasattr(mw, "set_background_busy"):
            mw.set_background_busy(False)
        self.publish_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_label.setVisible(False)
        if ok:
            QMessageBox.information(
                self, "Done", f"M4B saved to:\n{self.path_edit.text()}"
            )
        else:
            self.progress_label.setVisible(True)
            self.progress_label.setText("Publish failed.")

    def _publish_error(self, msg: str):
        mw = self.window()
        if hasattr(mw, "set_background_busy"):
            mw.set_background_busy(False)
        QMessageBox.critical(self, "Publish error", msg)

    def is_valid(self) -> bool:
        return bool(self.path_edit.text().strip())


# ---- Settings ----
SETTINGS_ORG = "M4BWizard"
SETTINGS_APP = "M4BWizard"


def _settings_bool(val) -> bool:
    """QSettings on Windows can return "true"/"false" strings; normalize to bool."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in ("true", "1", "yes")


def load_settings() -> dict:
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    raw_prefer = s.value("prefer_cpu", False)
    prefer_cpu = _settings_bool(raw_prefer)
    return {
        "whisper_model": s.value("whisper_model", "base", type=str) or "base",
        "whisper_language": s.value("whisper_language", "en", type=str) or "en",
        "prefer_cpu": prefer_cpu,
    }


def save_settings(settings: dict) -> None:
    s = QSettings(SETTINGS_ORG, SETTINGS_APP)
    s.setValue("whisper_model", settings.get("whisper_model", "base"))
    s.setValue("whisper_language", settings.get("whisper_language", "en"))
    s.setValue("prefer_cpu", settings.get("prefer_cpu", False))


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = dict(settings)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Whisper model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium"])
        idx = self.model_combo.findText(self.settings.get("whisper_model", "base"))
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 1)
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel("Whisper language (e.g. en, es):"))
        self.language_edit = QLineEdit()
        self.language_edit.setText(self.settings.get("whisper_language", "en"))
        self.language_edit.setPlaceholderText("e.g. en, es")
        layout.addWidget(self.language_edit)
        self.prefer_cpu_cb = QCheckBox("Prefer CPU (disable GPU for Whisper)")
        self.prefer_cpu_cb.setChecked(_settings_bool(self.settings.get("prefer_cpu", False)))
        layout.addWidget(self.prefer_cpu_cb)
        bbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bbox.accepted.connect(self.accept)
        bbox.rejected.connect(self.reject)
        layout.addWidget(bbox)

    def get_settings(self) -> dict:
        self.settings["whisper_model"] = self.model_combo.currentText()
        self.settings["whisper_language"] = self.language_edit.text().strip() or "en"
        self.settings["prefer_cpu"] = self.prefer_cpu_cb.isChecked()
        return self.settings


# ---- Main window ----
class WizardWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("M4B Wizard")
        self.resize(700, 550)
        self.state = ProjectState()
        self.settings = load_settings()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        top_row = QHBoxLayout()
        self.step_label = QLabel("Step 1 of 5: Select files")
        top_row.addWidget(self.step_label)
        top_row.addStretch()
        settings_btn = QPushButton("Settings")
        settings_btn.clicked.connect(self._open_settings)
        top_row.addWidget(settings_btn)
        layout.addLayout(top_row)

        self.stack = QStackedWidget()
        self.step1 = Step1Files(self.state, on_file_list_changed=self._update_buttons)
        self.step2 = Step2Breaks(self.state)
        self.step3 = Step3Names(self.state)
        self.step4 = Step4Metadata(self.state)
        self.step5 = Step5Publish(self.state)
        self.stack.addWidget(self.step1)
        self.stack.addWidget(self.step2)
        self.stack.addWidget(self.step3)
        self.stack.addWidget(self.step4)
        self.stack.addWidget(self.step5)
        layout.addWidget(self.stack)

        btn_layout = QHBoxLayout()
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self._back)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self._next)
        btn_layout.addWidget(self.back_btn)
        btn_layout.addWidget(self.next_btn)
        layout.addLayout(btn_layout)

        self._current_step = 0
        self._background_busy = False
        self._background_message = ""
        self.statusBar().showMessage("")
        self._update_buttons()

    def set_background_busy(self, busy: bool, message: str = "") -> None:
        """Track background work for status bar and close confirmation."""
        self._background_busy = busy
        self._background_message = message
        if busy:
            self.statusBar().showMessage("Working: " + (message or "Please wait..."))
        else:
            self.statusBar().showMessage("")

    def closeEvent(self, event):
        if self._background_busy:
            reply = QMessageBox.question(
                self,
                "Background task in progress",
                "A task is still running (e.g. transcribing or publishing). Close anyway? Progress will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()

    def _step_titles(self):
        return [
            "Step 1 of 5: Select files and chapter mode",
            "Step 2 of 5: Chapter breaks",
            "Step 3 of 5: Chapter names",
            "Step 4 of 5: Metadata",
            "Step 5 of 5: Publish",
        ]

    def _update_buttons(self):
        self.back_btn.setEnabled(self._current_step > 0)
        if self._current_step < 4:
            self.next_btn.setText("Next")
            self.next_btn.setEnabled(self._current_step_widget().is_valid())
        else:
            self.next_btn.setText("Publish")
            self.next_btn.setVisible(False)
        self.step_label.setText(self._step_titles()[self._current_step])

    def _current_step_widget(self) -> QWidget:
        return self.stack.widget(self._current_step)

    def _back(self):
        if self._current_step <= 0:
            return
        self._current_step -= 1
        self.stack.setCurrentIndex(self._current_step)
        self._update_buttons()

    def _next(self):
        w = self._current_step_widget()
        if hasattr(w, "save_to_state"):
            w.save_to_state()
        if not w.is_valid():
            return

        if self._current_step == 0:
            self._go_step2()
            return
        if self._current_step == 1:
            self.step3.refresh_from_state()
        if self._current_step == 2:
            self.step3.save_to_state()
        if self._current_step == 3:
            if self.state.file_paths and all(not (self.state.metadata.get(k) or "").strip() for k in ("title", "artist", "album", "genre", "comment")):
                meta = load_metadata_from_mp3(self.state.file_paths[0])
                for k, v in meta.items():
                    if v and k in self.state.metadata:
                        self.state.metadata[k] = v
                self.step4.refresh_from_state()
            self.step5.refresh_summary()

        if self._current_step >= 4:
            return
        self._current_step += 1
        self.stack.setCurrentIndex(self._current_step)
        self._update_buttons()

    def _go_step2(self):
        """From step 1: either run Whisper or compute one-per-file breaks."""
        self.step1.save_to_state()
        if self.state.chapter_mode == "one_per_file":
            total = 0.0
            break_times = []
            titles = []
            for i, p in enumerate(self.state.file_paths):
                d = get_audio_duration_sec(p)
                if d <= 0:
                    d = 0.0
                total += d
                if i < len(self.state.file_paths) - 1:
                    break_times.append(total)
                titles.append(p.stem or f"Chapter {len(titles) + 1}")
            if not titles:
                titles = [f"Chapter {i+1}" for i in range(len(self.state.file_paths))]
            self.state.break_times = break_times
            self.state.total_dur_sec = total
            self.state.proposed_titles = titles
            self.step2.show_one_per_file(
                self.state.break_times,
                self.state.total_dur_sec,
                self.state.proposed_titles,
            )
            self._current_step = 1
            self.stack.setCurrentIndex(1)
            self._update_buttons()
            return

        # Whisper: concat then transcribe in background — show step 2 so Next later goes to step 3
        self._current_step = 1
        self.stack.setCurrentIndex(1)
        self.step2.status_label.setText("Preparing... Concatenating and transcribing (this may take a while).")
        self.step2.stack.setCurrentWidget(self.step2.whisper_page)
        self.step2.transcript_edit.setPlainText("Transcribing with Whisper... Please wait.")
        self.next_btn.setEnabled(False)
        self.back_btn.setEnabled(False)
        self._update_buttons()

        self._temp_dir = tempfile.TemporaryDirectory()
        tmp = Path(self._temp_dir.name)
        out_m4a = tmp / "combined.m4a"
        out_wav = tmp / "combined.wav"

        self._concat_worker = ConcatWorker(
            self.state.file_paths, out_m4a, out_wav
        )
        self._concat_worker.finished.connect(self._on_concat_done)
        self._concat_worker.error.connect(self._on_worker_error)
        self._concat_worker.start()

        self._out_m4a = out_m4a
        self._out_wav = out_wav

    def _open_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() == QDialog.Accepted:
            self.settings = dlg.get_settings()
            save_settings(self.settings)

    def _on_concat_done(self, ok: bool):
        if not ok:
            self.set_background_busy(False)
            self.next_btn.setEnabled(True)
            self.back_btn.setEnabled(True)
            QMessageBox.critical(self, "Error", "Concatenation failed.")
            return
        self.state.combined_m4a_path = self._out_m4a
        self.state.combined_wav_path = self._out_wav
        import soundfile as sf
        import torch
        info = sf.info(str(self._out_wav))
        total_dur = info.frames / info.samplerate if info.samplerate else 0.0
        prefer_bool = _settings_bool(self.settings.get("prefer_cpu", False))
        device = "cpu" if prefer_bool else None
        cuda_available = torch.cuda.is_available()
        if device is None:
            device = "cuda" if cuda_available else "cpu"
        status_msg = f"Transcribing with Whisper ({device.upper()})..."
        if device == "cpu" and not prefer_bool:
            status_msg += " (PyTorch did not detect a CUDA GPU; install PyTorch with CUDA or check drivers)"
        self.set_background_busy(True, status_msg)
        self.step2.status_label.setText(status_msg)
        self._transcribe_worker = TranscribeWorker(
            self._out_wav,
            model_name=self.settings.get("whisper_model", "base"),
            language=self.settings.get("whisper_language", "en"),
            device=device,
        )
        self._transcribe_worker.finished.connect(self._on_transcribe_done)
        self._transcribe_worker.error.connect(self._on_worker_error)
        self._transcribe_worker.start()

    def _on_transcribe_done(self, segments: list):
        self.set_background_busy(False)
        self.next_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        import soundfile as sf
        info = sf.info(str(self.state.combined_wav_path))
        total_dur = info.frames / info.samplerate if info.samplerate else 0.0
        self.step2.status_label.setText("")
        self.step2.show_whisper_ui(segments, total_dur)

    def _on_worker_error(self, msg: str):
        self.set_background_busy(False)
        self.next_btn.setEnabled(True)
        self.back_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)


def main():
    app = QApplication([])
    win = WizardWindow()
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
