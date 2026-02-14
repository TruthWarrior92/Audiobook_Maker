# Combine Audio to M4B

Combine MP3 files into a single M4B audiobook with chapters. Use **Whisper** to detect spoken "Chapter N" cues (and edit breaks in a searchable transcript) or **one chapter per file**. GUI wizard or CLI.

## Requirements

- Python 3.10+
- FFmpeg on PATH
- GPU optional (VAD and Whisper run on CPU; GPU speeds up Whisper). For the GUI wizard, Whisper uses **base** by default; runs well on RTX 2070 (8GB) and GTX 1080 Ti (11GB) with CUDA and FP16.

## Setup

```bash
pip install -r requirements.txt
```

## GUI Wizard

Run the step-by-step wizard to select files, choose chapter mode (one per file or Whisper), edit chapter breaks in the transcript, set chapter names and metadata, then publish to a chosen path:

```bash
python m4b_wizard.py
```

**Wizard steps:** 1) Add MP3 files (and reorder); choose "One chapter per file" or "Use Whisper to find chapter breaks". 2) If Whisper: transcript appears with proposed breaks; search and add/remove breaks at segment boundaries. If one per file: break times are set from file durations. 3) Edit chapter names. 4) Set title, author, album, genre, comment. 5) Choose destination and click **Publish**.

## CLI Usage

```bash
# Process all books (Whisper for chapter cues, fallback to VAD)
python combine_to_m4b.py

# One book, target chapter count (used only when falling back to VAD)
python combine_to_m4b.py --book "Ender's Game" --chapters 15

# Use a larger Whisper model for better accuracy (slower)
python combine_to_m4b.py --whisper-model small

# Skip Whisper; use VAD-only chapter detection (generic "Chapter 1", "Chapter 2", ...)
python combine_to_m4b.py --no-whisper
```

**Options**

- `--whisper-model` — `tiny` | `base` | `small` | `medium` | `large` (default: `base`). Larger models are more accurate but slower and use more memory.
- `--no-whisper` — Disable Whisper; use only Silero VAD for chapter boundaries (no spoken chapter titles).

**Note:** Full run can take 15–30+ minutes for a long audiobook (concat + VAD + encode). With Whisper, transcription adds significant time; long books may take much longer. In the GUI, Whisper uses the **base** model and CUDA + FP16 when available (8GB+ VRAM).

## Output

- **GUI:** You choose the output path when you click Publish.
- **CLI:** M4B files are written as `books/<BookName>/<BookName>.m4b`. Chapter titles come from Whisper when detected, or generic "Chapter 1", "Chapter 2", etc.

## Project layout

- `combine_to_m4b.py` — CLI: processes all books in `books/` (one subfolder per book, each with numbered MP3s).
- `m4b_wizard.py` — GUI: wizard to pick files, set chapter mode, edit breaks, metadata, and publish.
- `requirements.txt` — Python dependencies (PyTorch, Whisper, PySide6, FFmpeg, etc.).
- `books/` — For CLI only: add a folder per book with MP3s inside; GUI can use any folder.
