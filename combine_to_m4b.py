"""
Combine MP3 files per book, detect chapter breaks via Whisper cues or Silero VAD, output M4B.
Requires FFmpeg on PATH.

Usage: python combine_to_m4b.py [--chapters N] [--book "Book Name"] [--whisper-model base] [--no-whisper]
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import whisper


def get_silero_vad():
    """Load Silero VAD model and utils."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return model, utils


def concat_mp3s_from_list(
    file_paths: list[Path], out_m4a: Path, out_wav: Path
) -> bool:
    """Concatenate an ordered list of MP3s to M4A and 16kHz mono WAV."""
    if not file_paths:
        return False
    list_path = out_m4a.parent / "_concat_list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in file_paths:
            path_str = Path(p).resolve().as_posix().replace("'", "'\\''")
            f.write(f"file '{path_str}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c:a", "aac", "-b:a", "192k",
        str(out_m4a),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_path.unlink(missing_ok=True)
    if result.returncode != 0:
        return False
    cmd_wav = [
        "ffmpeg", "-y", "-i", str(out_m4a),
        "-ar", "16000", "-ac", "1",
        str(out_wav),
    ]
    result = subprocess.run(cmd_wav, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return True


def concat_mp3s(book_dir: Path, out_m4a: Path, out_wav: Path) -> bool:
    """Concatenate MP3s in a directory to M4A (for output) and WAV (for VAD analysis)."""
    mp3s = sorted(book_dir.glob("*.mp3"), key=lambda p: _file_sort_key(p.name))
    if not mp3s:
        print(f"  No MP3 files in {book_dir}")
        return False
    return concat_mp3s_from_list(mp3s, out_m4a, out_wav)


def _file_sort_key(name: str) -> tuple:
    """Sort by leading numeric prefix."""
    m = re.match(r"(\d+)", name)
    return (int(m.group(1)), name) if m else (0, name)


def detect_chapters(
    wav_path: Path,
    model,
    get_speech_timestamps,
    target_chapters: int = 15,
    min_silence_sec: float = 1.5,
    min_chapter_sec: float = 45.0,
    sample_rate: int = 16000,
) -> list[tuple[float, float]]:
    """
    Return list of (start_sec, end_sec) for each chapter.
    Uses longest silences between speech to approximate target_chapters.
    """
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    waveform = torch.from_numpy(data).float().unsqueeze(0)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        sr = sample_rate
    wav = waveform.squeeze(0)

    timestamps = get_speech_timestamps(wav, model, sampling_rate=sr, return_seconds=True)
    total_dur = wav.shape[0] / sr
    if not timestamps:
        return [(0.0, total_dur)]

    gaps = []
    for i in range(1, len(timestamps)):
        gap_start = timestamps[i - 1]["end"]
        gap_end = timestamps[i]["start"]
        gap_dur = gap_end - gap_start
        if gap_dur >= min_silence_sec:
            gaps.append((gap_start, gap_end, gap_dur))

    if not gaps:
        return [(0.0, total_dur)]

    # Take top (target_chapters - 1) longest silences as chapter breaks
    gaps_sorted = sorted(gaps, key=lambda g: g[2], reverse=True)
    n_breaks = min(target_chapters - 1, len(gaps_sorted))
    break_positions = sorted([(g[0] + g[2] / 2) for g in gaps_sorted[:n_breaks]])

    chapters = []
    prev_end = 0.0
    for pos in break_positions:
        if pos - prev_end >= min_chapter_sec:
            chapters.append((prev_end, pos))
            prev_end = pos
    chapters.append((prev_end, total_dur))

    return chapters


# Match "Chapter N", "Chapter N:", "Chapter Twenty-Three", "Part 2", "Part Two", "Section 1", "Book 1"
CHAPTER_CUE_PATTERN = re.compile(
    r"^\s*(?:Chapter|Part|Section|Book)\s+(\d+|[A-Za-z\-]+(?:\s+[A-Za-z\-]+)*)\s*:?\s*(.*)$",
    re.IGNORECASE,
)


def parse_chapter_cues(segments: list[dict]) -> list[tuple[float, float, str]]:
    """
    From Whisper segments, return list of (start_sec, end_sec, title) for segments
    that look like chapter announcements. end_sec is the segment end; callers use
    segment start as next chapter start.
    """
    result = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        m = CHAPTER_CUE_PATTERN.match(text)
        if m:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", start))
            title = text.strip()
            result.append((start, end, title))
    return result


def transcribe_with_whisper(
    wav_path: Path,
    model_name: str = "base",
    language: str = "en",
    device: str | None = None,
) -> list[dict]:
    """
    Transcribe with Whisper; return segments as list of {start, end, text}.
    Uses CUDA + FP16 when device is cuda. model_name: tiny, base, small, medium, large.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=False,
        verbose=False,
        fp16=use_fp16,
    )
    return result.get("segments") or []


def propose_chapter_breaks(segments: list[dict]) -> list[tuple[float, str]]:
    """
    From Whisper segments, suggest chapter breaks from "Chapter N" etc.
    Returns list of (time_sec, title) for each suggested break (segment start).
    """
    cues = parse_chapter_cues(segments)
    return [(float(c[0]), c[2]) for c in sorted(cues, key=lambda x: x[0])]


def chapters_from_breaks(
    break_times: list[float],
    total_dur_sec: float,
    titles: list[str] | None = None,
) -> list[tuple[float, float, str]]:
    """
    Build (start, end, title) chapters from sorted break times.
    Breaks define start of each new chapter; first chapter starts at 0, last ends at total_dur_sec.
    """
    times = sorted(set(break_times))
    times = [t for t in times if 0 < t < total_dur_sec]
    if not times:
        title = (titles[0] if titles and len(titles) > 0 else "Chapter 1").strip() or "Chapter 1"
        return [(0.0, total_dur_sec, title)]
    chapters = []
    prev = 0.0
    for i, t in enumerate(times):
        title = (
            titles[i].strip() or f"Chapter {i + 1}"
            if titles and i < len(titles)
            else f"Chapter {i + 1}"
        )
        chapters.append((prev, t, title))
        prev = t
    last_title = (
        titles[len(times)].strip() or f"Chapter {len(times) + 1}"
        if titles and len(titles) > len(times)
        else f"Chapter {len(times) + 1}"
    )
    chapters.append((prev, total_dur_sec, last_title))
    return chapters


def get_audio_duration_sec(path: Path) -> float:
    """Get duration in seconds via ffprobe. Works for MP3, M4A, WAV."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def detect_chapters_whisper(
    wav_path: Path,
    whisper_model,
    total_dur_sec: float,
    language: str | None = "en",
) -> list[tuple[float, float, str]] | None:
    """
    Transcribe with Whisper, parse chapter cues. Returns list of (start, end, title)
    or None on failure or if fewer than 2 cues found.
    """
    try:
        result = whisper_model.transcribe(
            str(wav_path),
            language=language,
            word_timestamps=False,
            verbose=False,
        )
    except Exception as e:
        print(f"  Whisper failed: {e}")
        return None
    segments = result.get("segments") or []
    cues = parse_chapter_cues(segments)
    if len(cues) < 2:
        return None
    cues_sorted = sorted(cues, key=lambda c: c[0])
    chapters = []
    # Content before first cue = intro
    first_start = cues_sorted[0][0]
    if first_start > 0:
        chapters.append((0.0, first_start, "Introduction"))
    for i in range(len(cues_sorted)):
        start = cues_sorted[i][0]
        end = cues_sorted[i + 1][0] if i + 1 < len(cues_sorted) else total_dur_sec
        title = cues_sorted[i][2]
        chapters.append((start, end, title))
    return chapters


def _escape_ffmeta(s: str) -> str:
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("=", "\\=")


def write_ffmetadata(
    chapters: list[tuple[float, float]] | list[tuple[float, float, str]],
    title: str,
    artist: str,
    path: Path,
    metadata: dict | None = None,
) -> None:
    """Write FFMETADATA1 format for FFmpeg. Chapters are (start, end) or (start, end, chapter_title).
    If metadata dict is provided, write title, artist, album, genre, comment from it (overrides title/artist args if keys present).
    """
    meta = dict(metadata) if metadata else {}
    if "title" not in meta:
        meta["title"] = title
    if "artist" not in meta:
        meta["artist"] = artist
    with open(path, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        for key in ("title", "artist", "album", "genre", "comment", "date", "album_artist"):
            if key in meta and meta[key]:
                f.write(f"{key}={_escape_ffmeta(str(meta[key]))}\n")
        f.write("\n")
        for i, ch in enumerate(chapters):
            start, end = ch[0], ch[1]
            chapter_title = ch[2] if len(ch) >= 3 else f"Chapter {i + 1}"
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            f.write("[CHAPTER]\n")
            f.write("TIMEBASE=1/1000\n")
            f.write(f"START={start_ms}\n")
            f.write(f"END={end_ms}\n")
            f.write(f"title={_escape_ffmeta(chapter_title)}\n\n")


def create_m4b(
    audio_path: Path,
    metadata_path: Path,
    out_path: Path,
    metadata: dict | None = None,
) -> bool:
    """Create M4B with chapter metadata. Optional metadata dict adds -metadata k=v for M4B tags."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-i", str(metadata_path),
        "-map", "0:a",
        "-map_metadata", "1",
        "-map_chapters", "1",
        "-c:a", "copy",
    ]
    if metadata:
        for k, v in metadata.items():
            if v is not None and str(v).strip():
                cmd.extend(["-metadata", f"{k}={str(v).strip()}"])
    cmd.append(str(out_path))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FFmpeg M4B failed: {result.stderr[:500]}")
        return False
    return True


def process_book(
    book_dir: Path,
    vad_model,
    get_speech_timestamps,
    target_chapters: int = 15,
    whisper_model=None,
) -> bool:
    """Process one book: concat, detect chapters (Whisper or VAD), create M4B."""
    book_name = book_dir.name
    print(f"\nProcessing: {book_name}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        combined_m4a = tmp_path / "combined.m4a"
        combined_wav = tmp_path / "combined.wav"
        meta = tmp_path / "chapters.txt"

        if not concat_mp3s(book_dir, combined_m4a, combined_wav):
            return False
        print(f"  Concatenated to {combined_m4a.stat().st_size / 1e6:.1f} MB")

        info = sf.info(str(combined_wav))
        total_dur_sec = info.frames / info.samplerate if info.samplerate else 0.0
        if total_dur_sec <= 0:
            data, sr = sf.read(str(combined_wav))
            total_dur_sec = len(data) / sr

        chapters = None
        if whisper_model:
            print("  Transcribing with Whisper for chapter cues...")
            chapters = detect_chapters_whisper(
                combined_wav, whisper_model, total_dur_sec, language="en"
            )
        if chapters is None:
            print("  Analyzing speech for chapter boundaries (VAD)...")
            vad_chapters = detect_chapters(
                combined_wav,
                vad_model,
                get_speech_timestamps,
                target_chapters=target_chapters,
            )
            chapters = [(s, e) for s, e in vad_chapters]
        else:
            print(f"  Detected {len(chapters)} chapters from Whisper cues")

        if all(len(c) == 2 for c in chapters):
            print(f"  Detected {len(chapters)} chapters")
        write_ffmetadata(chapters, book_name, "", meta)

        out_path = book_dir / f"{book_name}.m4b"
        if not create_m4b(combined_m4a, meta, out_path):
            return False

    print(f"  Output: {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Combine MP3s to M4B with ML chapter detection")
    ap.add_argument("--chapters", type=int, default=15, help="Target chapter count (default: 15)")
    ap.add_argument("--book", type=str, help="Process only this book folder name")
    ap.add_argument(
        "--whisper-model",
        type=str,
        default="base",
        choices=("tiny", "base", "small", "medium", "large"),
        help="Whisper model for chapter cue detection (default: base)",
    )
    ap.add_argument("--no-whisper", action="store_true", help="Skip Whisper; use VAD-only chapter detection")
    args = ap.parse_args()

    books_dir = Path(__file__).resolve().parent / "books"
    if not books_dir.exists():
        print("books/ folder not found")
        sys.exit(1)

    print("Loading Silero VAD...")
    vad_model, utils = get_silero_vad()
    get_speech_timestamps = utils[0]

    whisper_model = None
    if not args.no_whisper:
        print(f"Loading Whisper ({args.whisper_model})...")
        whisper_model = whisper.load_model(args.whisper_model)

    book_dirs = [d for d in books_dir.iterdir() if d.is_dir()]
    if args.book:
        book_dirs = [d for d in book_dirs if d.name == args.book]
        if not book_dirs:
            print(f"Book '{args.book}' not found")
            sys.exit(1)
    elif not book_dirs:
        print("No book folders in books/")
        sys.exit(1)

    ok = 0
    for book_dir in book_dirs:
        if process_book(
            book_dir,
            vad_model,
            get_speech_timestamps,
            target_chapters=args.chapters,
            whisper_model=whisper_model,
        ):
            ok += 1

    print(f"\nDone: {ok}/{len(book_dirs)} books processed")
    input("\nPress Enter to close...")


if __name__ == "__main__":
    main()
