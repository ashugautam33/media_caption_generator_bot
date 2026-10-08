import os
import re
import json
import math
import asyncio
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import yt_dlp
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from google import genai
from google.genai import types

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

WHISPER_MODEL_NAME = os.getenv(
    "WHISPER_MODEL",
    "base",
).strip()

MAX_VIDEO_SECONDS = int(
    os.getenv(
        "MAX_VIDEO_SECONDS",
        "180",
    )
)

MAX_FILE_SIZE = 20 * 1024 * 1024

KALAKAR_URL = "https://app.kalakar.io/"


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


# ============================================================
# VALIDATION
# ============================================================

if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN is missing. Add BOT_TOKEN to Railway Variables."
    )

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add GEMINI_API_KEY to Railway Variables."
    )


# ============================================================
# GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

GEMINI_MODEL = None


PREFERRED_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def normalize_model_name(name: str) -> str:
    """
    Converts:
        models/gemini-xxx
    into:
        gemini-xxx
    """
    if not name:
        return ""

    return name.replace("models/", "").strip()


def find_gemini_model() -> Optional[str]:
    """
    Find a Gemini model available to the current API key.

    Only models supporting generateContent are considered.
    """

    logger.info(
        "Checking Gemini models available to the API key..."
    )

    try:
        models = list(
            gemini_client.models.list()
        )

    except Exception as exc:
        logger.exception(
            "Could not list Gemini models: %s",
            exc,
        )
        return None

    available = []

    for model in models:
        name = normalize_model_name(
            getattr(model, "name", "")
        )

        if not name:
            continue

        supported_actions = getattr(
            model,
            "supported_actions",
            None,
        )

        if supported_actions:
            supports_generate = (
                "generateContent"
                in supported_actions
            )

            if not supports_generate:
                continue

        available.append(name)

    logger.info(
        "Models available for generateContent:"
    )

    for name in available:
        logger.info("  %s", name)

    # First use preferred models.
    for preferred in PREFERRED_GEMINI_MODELS:
        if preferred in available:
            logger.info(
                "Automatically selected Gemini model: %s",
                preferred,
            )
            return preferred

    # Then use a generic flash model.
    for name in available:
        lowered = name.lower()

        if (
            "flash" in lowered
            and "embedding" not in lowered
            and "tts" not in lowered
            and "live" not in lowered
        ):
            logger.info(
                "Automatically selected fallback Gemini model: %s",
                name,
            )
            return name

    # Last fallback.
    if available:
        logger.info(
            "Using first available generateContent model: %s",
            available[0],
        )
        return available[0]

    logger.error(
        "No Gemini generateContent model is available."
    )

    return None


GEMINI_MODEL = find_gemini_model()

logger.info(
    "FINAL GEMINI MODEL: %s",
    GEMINI_MODEL,
)


# ============================================================
# WHISPER
# ============================================================

_whisper_model = None


def get_whisper_model():
    global _whisper_model

    if _whisper_model is None:
        logger.info(
            "Loading Whisper model: %s",
            WHISPER_MODEL_NAME,
        )

        _whisper_model = WhisperModel(
            WHISPER_MODEL_NAME,
            device="cpu",
            compute_type="int8",
        )

        logger.info(
            "Whisper model loaded."
        )

    return _whisper_model


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {
    "english": "English",
    "hindi": "Hindi",
    "punjabi": "Punjabi",
}


# ============================================================
# CAPTION STYLES
# ============================================================

CAPTION_STYLES = {
    "instagram": (
        "Create an engaging Instagram caption. "
        "Make it natural, modern and easy to read."
    ),

    "short": (
        "Create a short punchy Instagram caption. "
        "Keep it concise and impactful."
    ),

    "funny": (
        "Create a funny, witty and entertaining Instagram caption."
    ),

    "professional": (
        "Create a polished professional Instagram caption."
    ),

    "travel": (
        "Create an attractive travel-style Instagram caption."
    ),

    "romantic": (
        "Create a romantic and emotional Instagram caption."
    ),

    "viral": (
        "Create a highly engaging social-media caption "
        "with a strong hook."
    ),

    "aesthetic": (
        "Create a stylish aesthetic Instagram caption."
    ),

    "attitude": (
        "Create a confident attitude-style caption."
    ),

    "bollywood": (
        "Create a dramatic Bollywood-inspired caption."
    ),

    "podcast": (
        "Create a professional podcast social-media caption. "
        "Include a strong hook, concise summary, "
        "engagement line and relevant hashtags."
    ),

    "custom": (
        "Create the best possible social-media caption "
        "based on the supplied media."
    ),
}


# ============================================================
# KEYBOARDS
# ============================================================

def language_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "English",
                    callback_data="language:english",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Hindi",
                    callback_data="language:hindi",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Punjabi",
                    callback_data="language:punjabi",
                ),
            ],
        ]
    )


def style_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Instagram",
                    callback_data="style:instagram",
                ),
                InlineKeyboardButton(
                    "Short",
                    callback_data="style:short",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Funny",
                    callback_data="style:funny",
                ),
                InlineKeyboardButton(
                    "Professional",
                    callback_data="style:professional",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Travel",
                    callback_data="style:travel",
                ),
                InlineKeyboardButton(
                    "Romantic",
                    callback_data="style:romantic",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Viral",
                    callback_data="style:viral",
                ),
                InlineKeyboardButton(
                    "Aesthetic",
                    callback_data="style:aesthetic",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Attitude",
                    callback_data="style:attitude",
                ),
                InlineKeyboardButton(
                    "Bollywood",
                    callback_data="style:bollywood",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Podcast",
                    callback_data="style:podcast",
                ),
            ],
        ]
    )


def podcast_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Open Kalakar",
                    url=KALAKAR_URL,
                ),
            ],
            [
                InlineKeyboardButton(
                    "Generate Podcast Caption Here",
                    callback_data="podcast:generate",
                ),
            ],
        ]
    )


def after_caption_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Regenerate",
                    callback_data="regenerate",
                ),
            ],
            [
                InlineKeyboardButton(
                    "New Style",
                    callback_data="new_style",
                ),
                InlineKeyboardButton(
                    "New Language",
                    callback_data="new_language",
                ),
            ],
            [
                InlineKeyboardButton(
                    "New Media",
                    callback_data="new_media",
                ),
            ],
        ]
    )


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_filename(filename: str) -> str:
    filename = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        filename,
    )

    return filename[:150]


def run_subprocess(
    command: List[str],
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    logger.info(
        "Running command: %s",
        " ".join(command),
    )

    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


# ============================================================
# FFPROBE
# ============================================================

def probe_media(path: Path) -> Dict:
    """
    Returns ffprobe JSON information.
    """

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]

    result = run_subprocess(
        command,
        timeout=60,
    )

    if result.returncode != 0:
        logger.error(
            "ffprobe failed:\n%s",
            result.stderr,
        )

        raise RuntimeError(
            "Could not read the video file."
        )

    try:
        return json.loads(
            result.stdout
        )
    except json.JSONDecodeError:
        raise RuntimeError(
            "Invalid media information returned by ffprobe."
        )


def get_video_duration(path: Path) -> float:
    info = probe_media(path)

    duration = (
        info.get("format", {})
        .get("duration")
    )

    try:
        return float(duration)
    except Exception:
        return 0.0


def has_audio_stream(path: Path) -> bool:
    info = probe_media(path)

    streams = info.get(
        "streams",
        [],
    )

    for stream in streams:
        if stream.get("codec_type") == "audio":
            return True

    return False


# ============================================================
# AUDIO EXTRACTION
# ============================================================

def extract_audio(
    video_path: Path,
    audio_path: Path,
):
    """
    Extract podcast audio as:
        16 kHz
        mono
        PCM WAV

    This format works reliably with Whisper.
    """

    logger.info(
        "Checking audio stream..."
    )

    if not has_audio_stream(video_path):
        raise RuntimeError(
            "This video has no audio track."
        )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]

    result = run_subprocess(
        command,
        timeout=180,
    )

    if result.returncode != 0:
        logger.error(
            "FFmpeg audio extraction failed:\n%s",
            result.stderr,
        )

        raise RuntimeError(
            "FFmpeg could not extract the audio track."
        )

    if not audio_path.exists():
        raise RuntimeError(
            "Audio extraction completed but no WAV file was created."
        )

    if audio_path.stat().st_size < 1000:
        raise RuntimeError(
            "The extracted audio file is empty or invalid."
        )

    logger.info(
        "Audio extracted successfully: %s bytes",
        audio_path.stat().st_size,
    )


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(
    audio_path: Path,
    language: Optional[str] = None,
):
    """
    Returns:
        full_text
        segments
        words
    """

    model = get_whisper_model()

    whisper_language = None

    if language == "english":
        whisper_language = "en"

    elif language == "hindi":
        whisper_language = "hi"

    elif language == "punjabi":
        whisper_language = "pa"

    logger.info(
        "Starting Whisper transcription. Language=%s",
        whisper_language,
    )

    segments_generator, info = model.transcribe(
        str(audio_path),
        language=whisper_language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )

    segments = []
    all_words = []
    full_text_parts = []

    for segment in segments_generator:
        segment_words = []

        if segment.words:
            for word in segment.words:
                text_value = (
                    word.word or ""
                ).strip()

                if not text_value:
                    continue

                item = {
                    "word": text_value,
                    "start": float(word.start),
                    "end": float(word.end),
                }

                segment_words.append(item)
                all_words.append(item)

        segment_item = {
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            "words": segment_words,
        }

        segments.append(
            segment_item
        )

        if segment.text.strip():
            full_text_parts.append(
                segment.text.strip()
            )

    full_text = " ".join(
        full_text_parts
    ).strip()

    logger.info(
        "Whisper detected language: %s",
        getattr(info, "language", "unknown"),
    )

    logger.info(
        "Transcription words: %d",
        len(all_words),
    )

    return (
        full_text,
        segments,
        all_words,
    )


# ============================================================
# GEMINI ERROR
# ============================================================

def friendly_gemini_error(exc: Exception) -> str:
    text = str(exc)

    lowered = text.lower()

    if (
        "429" in lowered
        or "quota" in lowered
        or "resource exhausted" in lowered
    ):
        return (
            "Gemini quota is currently unavailable. "
            "Please try again later."
        )

    if (
        "api key" in lowered
        or "permission" in lowered
        or "unauthorized" in lowered
        or "401" in lowered
        or "403" in lowered
    ):
        return (
            "Gemini API authentication failed. "
            "Please check GEMINI_API_KEY in Railway."
        )

    if (
        "404" in lowered
        or "not found" in lowered
        or "model" in lowered
    ):
        return (
            "Gemini model is unavailable for this API key."
        )

    return (
        "Gemini could not generate the caption."
    )


# ============================================================
# GEMINI CAPTION
# ============================================================

async def generate_caption_from_text(
    transcript: str,
    language: str,
    style: str,
) -> str:

    if not GEMINI_MODEL:
        raise RuntimeError(
            "No Gemini model is available."
        )

    language_name = LANGUAGES.get(
        language,
        "English",
    )

    style_instruction = CAPTION_STYLES.get(
        style,
        CAPTION_STYLES["instagram"],
    )

    prompt = f"""
You are a professional social-media caption writer.

Create an Instagram/social-media caption based on this podcast/video transcript.

Target language:
{language_name}

Style:
{style_instruction}

Requirements:

1. Write naturally in the requested language.
2. Do not mention that AI was used.
3. Do not say "here is your caption".
4. Create a strong opening hook.
5. Keep the caption engaging.
6. Include a concise relevant description.
7. End with a natural engagement line.
8. Add relevant hashtags.
9. Do not invent facts that are not supported by the transcript.
10. Avoid excessive hashtags.

Return only the final caption.

TRANSCRIPT:
{transcript}
"""

    logger.info(
        "Generating Gemini caption..."
    )

    response = await asyncio.to_thread(
        gemini_client.models.generate_content,
        model=GEMINI_MODEL,
        contents=prompt,
    )

    text = getattr(
        response,
        "text",
        None,
    )

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return text.strip()


# ============================================================
# PHOTO CAPTION
# ============================================================

async def generate_photo_caption(
    message,
    context,
):

    language = context.user_data.get(
        "language",
        "english",
    )

    style = context.user_data.get(
        "style",
        "instagram",
    )

    file_id = context.user_data.get(
        "file_id"
    )

    if not file_id:
        await message.reply_text(
            "Please send the photo again."
        )
        return

    status = await message.reply_text(
        "✍️ Writing your caption"
    )

    try:
        telegram_file = await context.bot.get_file(
            file_id
        )

        with tempfile.TemporaryDirectory() as temp:
            temp_dir = Path(temp)

            photo_path = (
                temp_dir / "photo.jpg"
            )

            await telegram_file.download_to_drive(
                custom_path=str(photo_path)
            )

            with open(
                photo_path,
                "rb",
            ) as image_file:
                image_bytes = image_file.read()

            if not GEMINI_MODEL:
                raise RuntimeError(
                    "Gemini model unavailable."
                )

            language_name = LANGUAGES.get(
                language,
                "English",
            )

            style_instruction = CAPTION_STYLES.get(
                style,
                CAPTION_STYLES["instagram"],
            )

            prompt = f"""
Analyze this image and write an Instagram caption.

Language:
{language_name}

Style:
{style_instruction}

Requirements:
- Accurately describe what is visible.
- Do not invent names, locations or facts.
- Make it natural and engaging.
- Include relevant hashtags.
- Return only the final caption.
"""

            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    prompt,
                ],
            )

            caption = getattr(
                response,
                "text",
                None,
            )

            if not caption:
                raise RuntimeError(
                    "Gemini returned an empty caption."
                )

            await status.edit_text(
                caption.strip(),
                reply_markup=after_caption_keyboard(),
            )

    except Exception as exc:
        logger.exception(
            "Photo caption error: %s",
            exc,
        )

        await status.edit_text(
            f"❌ Could not generate caption.\n\n"
            f"{friendly_gemini_error(exc)}"
        )


# ============================================================
# ASS ESCAPING
# ============================================================

def ass_escape(text: str) -> str:
    text = text.replace(
        "\\",
        r"\\",
    )

    text = text.replace(
        "{",
        r"\{",
    )

    text = text.replace(
        "}",
        r"\}",
    )

    text = text.replace(
        "\n",
        r"\N",
    )

    return text


def ass_time(seconds: float) -> str:
    seconds = max(
        0.0,
        float(seconds),
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    remaining = seconds % 60

    centiseconds = int(
        round(
            (remaining - int(remaining))
            * 100
        )
    )

    whole_seconds = int(
        remaining
    )

    if centiseconds >= 100:
        whole_seconds += 1
        centiseconds = 0

    return (
        f"{hours}:{minutes:02d}:"
        f"{whole_seconds:02d}."
        f"{centiseconds:02d}"
    )


# ============================================================
# ASS CAPTION GENERATION
# ============================================================

def group_words(
    words: List[Dict],
    words_per_caption: int = 4,
) -> List[List[Dict]]:

    groups = []

    current = []

    for word in words:
        current.append(word)

        if len(current) >= words_per_caption:
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def create_ass_subtitles(
    words: List[Dict],
    ass_path: Path,
):
    """
    Create animated karaoke-style ASS subtitles.

    Each caption displays a small group of words,
    with word-by-word highlighting using ASS karaoke tags.
    """

    groups = group_words(
        words,
        words_per_caption=4,
    )

    header = r"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 2
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Podcast,Arial,72,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,2,5,50,50,500,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [
        header
    ]

    for group in groups:
        if not group:
            continue

        start = group[0]["start"]
        end = group[-1]["end"]

        if end <= start:
            end = start + 0.5

        parts = []

        for word in group:
            duration_cs = max(
                1,
                int(
                    round(
                        (
                            word["end"]
                            - word["start"]
                        )
                        * 100
                    )
                ),
            )

            clean_word = ass_escape(
                word["word"]
            )

            parts.append(
                r"{\k"
                + str(duration_cs)
                + "}"
                + clean_word
            )

        text = " ".join(parts)

        line = (
            "Dialogue: 0,"
            f"{ass_time(start)},"
            f"{ass_time(end)},"
            "Podcast,,0,0,0,,"
            f"{text}\n"
        )

        lines.append(line)

    ass_path.write_text(
        "".join(lines),
        encoding="utf-8",
    )


# ============================================================
# BURN CAPTIONS
# ============================================================

def burn_captions(
    video_path: Path,
    ass_path: Path,
    output_path: Path,
):
    """
    Burn ASS subtitles into the MP4.
    """

    subtitle_path = str(
        ass_path
    ).replace(
        "\\",
        "/",
    )

    # Escape Windows-style drive colon if needed.
    subtitle_filter = (
        "subtitles="
        + subtitle_path.replace(
            ":",
            r"\:",
        )
    )

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        subtitle_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    result = run_subprocess(
        command,
        timeout=600,
    )

    if result.returncode != 0:
        logger.error(
            "Caption rendering failed:\n%s",
            result.stderr,
        )

        raise RuntimeError(
            "FFmpeg could not render the captions."
        )

    if not output_path.exists():
        raise RuntimeError(
            "Captioned video was not created."
        )

    if output_path.stat().st_size < 1000:
        raise RuntimeError(
            "Captioned video is empty."
        )


# ============================================================
# VIDEO CAPTION
# ============================================================

async def generate_video_caption(
    message,
    context,
    podcast_mode: bool = False,
):
    language = context.user_data.get(
        "language",
        "english",
    )

    style = context.user_data.get(
        "style",
        "podcast" if podcast_mode else "instagram",
    )

    file_id = context.user_data.get(
        "file_id"
    )

    if not file_id:
        await message.reply_text(
            "Please send the video again."
        )
        return

    if podcast_mode:
        status_text = (
            "✍️ creating your caption."
        )
    else:
        status_text = (
            "✍️ creating your caption."
        )

    status = await message.reply_text(
        status_text
    )

    try:
        with tempfile.TemporaryDirectory() as temp:
            temp_dir = Path(temp)

            video_path = (
                temp_dir / "input.mp4"
            )

            audio_path = (
                temp_dir / "audio.wav"
            )

            ass_path = (
                temp_dir / "captions.ass"
            )

            output_path = (
                temp_dir / "captioned.mp4"
            )

            # ------------------------------------------------
            # DOWNLOAD TELEGRAM VIDEO
            # ------------------------------------------------

            telegram_file = await context.bot.get_file(
                file_id
            )

            await telegram_file.download_to_drive(
                custom_path=str(video_path)
            )

            if not video_path.exists():
                raise RuntimeError(
                    "Video could not be downloaded."
                )

            if (
                video_path.stat().st_size
                > MAX_FILE_SIZE
            ):
                raise RuntimeError(
                    "Video is larger than the allowed file size."
                )

            # ------------------------------------------------
            # CHECK VIDEO
            # ------------------------------------------------

            duration = get_video_duration(
                video_path
            )

            logger.info(
                "Video duration: %.2f seconds",
                duration,
            )

            if (
                duration > 0
                and duration > MAX_VIDEO_SECONDS
            ):
                raise RuntimeError(
                    f"Video is longer than the allowed "
                    f"{MAX_VIDEO_SECONDS} seconds."
                )

            # ------------------------------------------------
            # AUDIO EXTRACTION
            # ------------------------------------------------

            await status.edit_text(
                "✍️ creating your caption."
            )

            await asyncio.to_thread(
                extract_audio,
                video_path,
                audio_path,
            )

            # ------------------------------------------------
            # TRANSCRIPTION
            # ------------------------------------------------

            await status.edit_text(
                "✍️ creating your caption."
            )

            (
                transcript,
                segments,
                words,
            ) = await asyncio.to_thread(
                transcribe_audio,
                audio_path,
                language,
            )

            if not transcript:
                raise RuntimeError(
                    "No speech could be detected."
                )

            if not words:
                raise RuntimeError(
                    "No word timestamps were generated."
                )

            logger.info(
                "Transcript:\n%s",
                transcript[:2000],
            )

            # ------------------------------------------------
            # GEMINI CAPTION
            # ------------------------------------------------

            caption = await generate_caption_from_text(
                transcript=transcript,
                language=language,
                style=style,
            )

            # ------------------------------------------------
            # ASS SUBTITLES
            # ------------------------------------------------

            await status.edit_text(
                "✍️ creating your caption."
            )

            await asyncio.to_thread(
                create_ass_subtitles,
                words,
                ass_path,
            )

            # ------------------------------------------------
            # BURN INTO VIDEO
            # ------------------------------------------------

            await asyncio.to_thread(
                burn_captions,
                video_path,
                ass_path,
                output_path,
            )

            # ------------------------------------------------
            # SEND CAPTION
            # ------------------------------------------------

            await message.reply_text(
                "Caption:\n\n"
                + caption
            )

            # ------------------------------------------------
            # SEND VIDEO
            # ------------------------------------------------

            await message.reply_video(
                video=open(
                    output_path,
                    "rb",
                ),
                caption=(
                    "Captioned video ready."
                ),
                supports_streaming=True,
                read_timeout=300,
                write_timeout=300,
                connect_timeout=60,
                pool_timeout=60,
            )

            await status.delete()

    except Exception as exc:
        logger.exception(
            "Video processing failed: %s",
            exc,
        )

        error_text = str(exc)

        await status.edit_text(
            "❌ **Video processing failed.**\n\n"
            f"`{error_text}`",
            parse_mode="Markdown",
        )


# ============================================================
# START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Generate Caption",
                    callback_data="new_media",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Podcast Captions",
                    callback_data="podcast:start",
                ),
            ],
            [
                InlineKeyboardButton(
                    "Open Kalakar",
                    url=KALAKAR_URL,
                ),
            ],
        ]
    )

    await update.message.reply_text(
        "Welcome to Caption On The Way.\n\n"
        "Send me a photo or video and I will "
        "create an Instagram caption for you.\n\n"
        "For podcast videos, I can also create "
        "timed captions directly on the video.",
        reply_markup=keyboard,
    )


# ============================================================
# HELP
# ============================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    await update.message.reply_text(
        "How to use Caption On The Way:\n\n"
        "1. Send a photo or video.\n"
        "2. Select your caption style.\n"
        "3. Select your language.\n"
        "4. Wait for the caption.\n\n"
        "Podcast:\n"
        "Send a podcast video and choose Podcast "
        "to create timed video captions.\n\n"
        "Available languages:\n"
        "English\n"
        "Hindi\n"
        "Punjabi\n\n"
        "/podcast - Podcast caption workflow\n"
        "/kalakar - Open Kalakar\n"
        "/start - Start the bot",
    )


# ============================================================
# PODCAST COMMAND
# ============================================================

async def podcast_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    await update.message.reply_text(
        "Podcast Captioning\n\n"
        "Send your podcast video to create "
        "automatic timed captions.\n\n"
        "You can also open Kalakar for its "
        "caption editor.",
        reply_markup=podcast_keyboard(),
    )


# ============================================================
# KALAKAR COMMAND
# ============================================================

async def kalakar_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    await update.message.reply_text(
        "Kalakar Caption Editor\n\n"
        "Open Kalakar to edit and style captions "
        "for your video.",
        reply_markup=InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "Open Kalakar",
                        url=KALAKAR_URL,
                    )
                ]
            ]
        ),
    )


# ============================================================
# MEDIA HANDLER
# ============================================================

async def media_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    message = update.message

    context.user_data.clear()

    if message.photo:

        photo = message.photo[-1]

        context.user_data["media_type"] = "photo"
        context.user_data["file_id"] = (
            photo.file_id
        )

    elif message.video:

        video = message.video

        if (
            video.file_size
            and video.file_size > MAX_FILE_SIZE
        ):
            await message.reply_text(
                "❌ This video is too large."
            )
            return

        context.user_data["media_type"] = "video"
        context.user_data["file_id"] = (
            video.file_id
        )

    elif message.document:

        document = message.document

        mime = (
            document.mime_type
            or ""
        ).lower()

        if mime.startswith("video/"):

            if (
                document.file_size
                and document.file_size > MAX_FILE_SIZE
            ):
                await message.reply_text(
                    "❌ This video is too large."
                )
                return

            context.user_data[
                "media_type"
            ] = "video"

            context.user_data[
                "file_id"
            ] = document.file_id

        else:
            await message.reply_text(
                "Please send a photo or video."
            )
            return

    else:
        return

    context.user_data["stage"] = (
        "style"
    )

    await message.reply_text(
        "Media received.\n\n"
        "Choose your caption style:",
        reply_markup=style_keyboard(),
    )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    text = (
        update.message.text
        or ""
    ).strip()

    if not text:
        return

    if context.user_data.get(
        "stage"
    ) == "custom_style":

        context.user_data[
            "custom_style"
        ] = text

        context.user_data[
            "style"
        ] = "custom"

        context.user_data[
            "stage"
        ] = "language"

        await update.message.reply_text(
            "Choose your language:",
            reply_markup=language_keyboard(),
        )

        return

    await update.message.reply_text(
        "Please send a photo or video, "
        "or use /start."
    )


# ============================================================
# CALLBACK HANDLER
# ============================================================

async def callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    data = query.data

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    if data.startswith("style:"):

        style = data.split(
            ":",
            1,
        )[1]

        if style not in CAPTION_STYLES:
            return

        context.user_data[
            "style"
        ] = style

        context.user_data[
            "stage"
        ] = "language"

        if style == "custom":

            context.user_data[
                "stage"
            ] = "custom_style"

            await query.message.reply_text(
                "Write your custom caption style:"
            )

            return

        await query.message.reply_text(
            "Choose your language:",
            reply_markup=language_keyboard(),
        )

        return

    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

    if data.startswith("language:"):

        language = data.split(
            ":",
            1,
        )[1]

        if language not in LANGUAGES:
            return

        context.user_data[
            "language"
        ] = language

        context.user_data[
            "stage"
        ] = "generating"

        media_type = context.user_data.get(
            "media_type"
        )

        if media_type == "photo":

            await generate_photo_caption(
                query.message,
                context,
            )

        elif media_type == "video":

            podcast_mode = (
                context.user_data.get(
                    "podcast_mode",
                    False,
                )
            )

            await generate_video_caption(
                query.message,
                context,
                podcast_mode=podcast_mode,
            )

        else:

            await query.message.reply_text(
                "Media information was lost.\n\n"
                "Please send the photo or video again."
            )

        return

    # --------------------------------------------------------
    # PODCAST START
    # --------------------------------------------------------

    if data == "podcast:start":

        context.user_data.clear()

        context.user_data[
            "podcast_mode"
        ] = True

        context.user_data[
            "style"
        ] = "podcast"

        context.user_data[
            "stage"
        ] = "waiting_media"

        await query.message.reply_text(
            "Podcast mode enabled.\n\n"
            "Send your podcast video."
        )

        return

    # --------------------------------------------------------
    # PODCAST GENERATE
    # --------------------------------------------------------

    if data == "podcast:generate":

        context.user_data[
            "podcast_mode"
        ] = True

        context.user_data[
            "style"
        ] = "podcast"

        context.user_data[
            "stage"
        ] = "language"

        await query.message.reply_text(
            "Send your podcast video first."
        )

        return

    # --------------------------------------------------------
    # REGENERATE
    # --------------------------------------------------------

    if data == "regenerate":

        media_type = context.user_data.get(
            "media_type"
        )

        if media_type == "photo":

            await generate_photo_caption(
                query.message,
                context,
            )

        elif media_type == "video":

            await generate_video_caption(
                query.message,
                context,
                podcast_mode=context.user_data.get(
                    "podcast_mode",
                    False,
                ),
            )

        else:

            await query.message.reply_text(
                "Please send your media again."
            )

        return

    # --------------------------------------------------------
    # NEW STYLE
    # --------------------------------------------------------

    if data == "new_style":

        await query.message.reply_text(
            "Choose your caption style:",
            reply_markup=style_keyboard(),
        )

        return

    # --------------------------------------------------------
    # NEW LANGUAGE
    # --------------------------------------------------------

    if data == "new_language":

        await query.message.reply_text(
            "Choose your language:",
            reply_markup=language_keyboard(),
        )

        return

    # --------------------------------------------------------
    # NEW MEDIA
    # --------------------------------------------------------

    if data == "new_media":

        context.user_data.clear()

        await query.message.reply_text(
            "Send a new photo or video."
        )

        return


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):

    logger.exception(
        "Unhandled bot exception:",
        exc_info=context.error,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "Starting Caption On The Way bot..."
    )

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    # Commands
    application.add_handler(
        CommandHandler(
            "start",
            start_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "podcast",
            podcast_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "kalakar",
            kalakar_command,
        )
    )

    # Callbacks
    application.add_handler(
        CallbackQueryHandler(
            callback_handler
        )
    )

    # Photos/videos/documents
    application.add_handler(
        MessageHandler(
            filters.PHOTO
            | filters.VIDEO
            | filters.Document.VIDEO,
            media_handler,
        )
    )

    # Text
    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_handler,
        )
    )

    application.add_error_handler(
        error_handler
    )

    logger.info(
        "Bot is running."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
