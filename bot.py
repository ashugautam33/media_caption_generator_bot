import os
import re
import shutil
import asyncio
import logging
import mimetypes
import tempfile
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from google import genai
from google.genai import types

from faster_whisper import WhisperModel

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

BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    ""
).strip()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY",
    ""
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite"
).strip()

WHISPER_MODEL_NAME = os.getenv(
    "WHISPER_MODEL",
    "base"
).strip()

MAX_VIDEO_SECONDS = int(
    os.getenv(
        "MAX_VIDEO_SECONDS",
        "60"
    )
)

MAX_FILE_SIZE = 20 * 1024 * 1024

VIDEO_FRAME_COUNT = 6


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
    level=logging.INFO,
)

logger = logging.getLogger(
    "media-caption-generator"
)


# ============================================================
# VALIDATION
# ============================================================

if not BOT_TOKEN:

    raise RuntimeError(
        "BOT_TOKEN is missing."
    )


if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is missing."
    )


# ============================================================
# GEMINI
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# WHISPER
# ============================================================

logger.info(
    "Loading Whisper model: %s",
    WHISPER_MODEL_NAME
)

whisper_model = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type="int8",
)

logger.info(
    "Whisper model loaded."
)


# ============================================================
# STYLES
# ============================================================

CAPTION_STYLES = {

    "instagram": {
        "name": "Instagram",
        "instruction": (
            "Create a polished, natural and engaging "
            "Instagram caption."
        ),
    },

    "short": {
        "name": "Short",
        "instruction": (
            "Create a very short and punchy caption."
        ),
    },

    "funny": {
        "name": "Funny",
        "instruction": (
            "Create a funny and playful caption."
        ),
    },

    "professional": {
        "name": "Professional",
        "instruction": (
            "Create a professional and polished caption."
        ),
    },

    "travel": {
        "name": "Travel",
        "instruction": (
            "Create a travel-inspired caption focused "
            "on the journey, location, atmosphere or adventure."
        ),
    },

    "romantic": {
        "name": "Romantic",
        "instruction": (
            "Create a tasteful romantic and emotional caption."
        ),
    },

    "viral": {
        "name": "Viral",
        "instruction": (
            "Create an attention-grabbing caption with "
            "a strong social-media hook."
        ),
    },

    "aesthetic": {
        "name": "Aesthetic",
        "instruction": (
            "Create a stylish, poetic and aesthetically "
            "pleasing caption."
        ),
    },

    "attitude": {
        "name": "Attitude",
        "instruction": (
            "Create a confident, bold and stylish caption."
        ),
    },

    "bollywood": {
        "name": "Bollywood",
        "instruction": (
            "Create an original cinematic Bollywood-inspired "
            "caption. Do not copy song lyrics."
        ),
    },

    "custom": {
        "name": "Custom",
        "instruction": (
            "Follow the user's custom instructions."
        ),
    },
}


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {

    "english": {
        "name": "English",
        "whisper": "en",
        "instruction": (
            "Write the caption and subtitles in natural English."
        ),
    },

    "hindi": {
        "name": "Hindi",
        "whisper": "hi",
        "instruction": (
            "Write primarily in Hindi using Devanagari script."
        ),
    },

    "hinglish": {
        "name": "Hinglish",
        "whisper": None,
        "instruction": (
            "Write natural Hinglish using Roman script, "
            "mixing Hindi and English."
        ),
    },

    "punjabi": {
        "name": "Punjabi",
        "whisper": "pa",
        "instruction": (
            "Write primarily in Punjabi using Gurmukhi script."
        ),
    },
}


# ============================================================
# START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    context.user_data.clear()

    text = (
        "🎨 *MEDIA CAPTION GENERATOR*\n\n"
        "Welcome! 👋\n\n"
        "📸 Send a photo\n"
        "🎥 Send a video\n\n"
        "I can create:\n"
        "• Instagram captions\n"
        "• Hashtags\n"
        "• Hindi captions\n"
        "• Punjabi captions\n"
        "• Hinglish captions\n"
        "• Video subtitles\n"
        "• Word-synchronized captions\n"
        "• Captioned MP4 videos\n\n"
        "🤖 AI: Gemini\n"
        "🎙 Speech recognition: Whisper\n"
        "🎬 Video processing: FFmpeg\n\n"
        "📤 Send your media to begin."
    )

    await update.message.reply_text(
        text,
        parse_mode="Markdown",
    )


# ============================================================
# HELP
# ============================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    text = (
        "📖 *HOW TO USE*\n\n"

        "📸 *PHOTO*\n"
        "Send a photo → choose style → choose language "
        "→ receive caption + hashtags.\n\n"

        "🎥 *VIDEO*\n"
        "Send a video → choose style → choose language "
        "→ choose video processing option.\n\n"

        "🎬 *VIDEO FEATURES*\n"
        "🤖 AI Caption\n"
        "📝 Subtitles\n"
        "🎬 Caption + Subtitles\n\n"

        "🌐 Languages:\n"
        "🇬🇧 English\n"
        "🇮🇳 Hindi\n"
        "🗣 Hinglish\n"
        "🪯 Punjabi\n\n"

        "Commands:\n"
        "/start - Start bot\n"
        "/help - Help\n"
        "/cancel - Cancel"
    )

    await update.message.reply_text(
        text,
        parse_mode="Markdown",
    )


# ============================================================
# CANCEL
# ============================================================

async def cancel(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    context.user_data.clear()

    await update.message.reply_text(
        "✅ Request cancelled.\n\n"
        "📤 Send a new photo or video."
    )


# ============================================================
# STYLE KEYBOARD
# ============================================================

def style_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "📸 Instagram",
                callback_data="style:instagram",
            ),
            InlineKeyboardButton(
                "✨ Short",
                callback_data="style:short",
            ),
        ],

        [
            InlineKeyboardButton(
                "😂 Funny",
                callback_data="style:funny",
            ),
            InlineKeyboardButton(
                "💼 Professional",
                callback_data="style:professional",
            ),
        ],

        [
            InlineKeyboardButton(
                "✈️ Travel",
                callback_data="style:travel",
            ),
            InlineKeyboardButton(
                "❤️ Romantic",
                callback_data="style:romantic",
            ),
        ],

        [
            InlineKeyboardButton(
                "🔥 Viral",
                callback_data="style:viral",
            ),
            InlineKeyboardButton(
                "🌸 Aesthetic",
                callback_data="style:aesthetic",
            ),
        ],

        [
            InlineKeyboardButton(
                "😎 Attitude",
                callback_data="style:attitude",
            ),
            InlineKeyboardButton(
                "🎬 Bollywood",
                callback_data="style:bollywood",
            ),
        ],

        [
            InlineKeyboardButton(
                "✍️ Custom",
                callback_data="style:custom",
            ),
        ],
    ])


# ============================================================
# LANGUAGE KEYBOARD
# ============================================================

def language_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "🇬🇧 English",
                callback_data="language:english",
            ),
            InlineKeyboardButton(
                "🇮🇳 Hindi",
                callback_data="language:hindi",
            ),
        ],

        [
            InlineKeyboardButton(
                "🗣 Hinglish",
                callback_data="language:hinglish",
            ),
            InlineKeyboardButton(
                "🪯 Punjabi",
                callback_data="language:punjabi",
            ),
        ],
    ])


# ============================================================
# PHOTO RESULT KEYBOARD
# ============================================================

def photo_result_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "🔄 Regenerate",
                callback_data="action:regenerate",
            ),
        ],

        [
            InlineKeyboardButton(
                "🎨 New Style",
                callback_data="action:style",
            ),
            InlineKeyboardButton(
                "🌐 New Language",
                callback_data="action:language",
            ),
        ],

        [
            InlineKeyboardButton(
                "📤 New Media",
                callback_data="action:new",
            ),
        ],
    ])


# ============================================================
# VIDEO RESULT KEYBOARD
# ============================================================

def video_result_keyboard():

    return InlineKeyboardMarkup([

        [
            InlineKeyboardButton(
                "🔄 Regenerate Caption",
                callback_data="action:regenerate",
            ),
        ],

        [
            InlineKeyboardButton(
                "📝 Subtitles",
                callback_data="video:subtitles",
            ),
        ],

        [
            InlineKeyboardButton(
                "🎬 Caption + Subtitles",
                callback_data="video:both",
            ),
        ],

        [
            InlineKeyboardButton(
                "🎨 New Style",
                callback_data="action:style",
            ),
        ],

        [
            InlineKeyboardButton(
                "🌐 New Language",
                callback_data="action:language",
            ),
        ],

        [
            InlineKeyboardButton(
                "📤 New Media",
                callback_data="action:new",
            ),
        ],
    ])


# ============================================================
# MEDIA HANDLER
# ============================================================

async def media_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    message = update.effective_message

    media_type = None
    file_id = None
    file_size = None

    # --------------------------------------------------------
    # PHOTO
    # --------------------------------------------------------

    if message.photo:

        photo = message.photo[-1]

        media_type = "photo"

        file_id = photo.file_id

        file_size = photo.file_size

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    elif message.video:

        video = message.video

        media_type = "video"

        file_id = video.file_id

        file_size = video.file_size

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    elif message.document:

        document = message.document

        mime = (
            document.mime_type or ""
        ).lower()

        if mime.startswith("image/"):

            media_type = "photo"

        elif mime.startswith("video/"):

            media_type = "video"

        else:

            await message.reply_text(
                "❌ Unsupported file.\n\n"
                "Please send an image or video."
            )

            return

        file_id = document.file_id

        file_size = document.file_size

    else:

        return

    # --------------------------------------------------------
    # SIZE
    # --------------------------------------------------------

    if (
        file_size
        and file_size > MAX_FILE_SIZE
    ):

        await message.reply_text(
            "❌ File is too large.\n\n"
            "Please send a file smaller than 20 MB."
        )

        return

    # --------------------------------------------------------
    # SAVE STATE
    # --------------------------------------------------------

    context.user_data.clear()

    context.user_data["file_id"] = file_id

    context.user_data["media_type"] = media_type

    context.user_data["original_caption"] = (
        message.caption or ""
    )

    context.user_data["stage"] = "style"

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

    await message.reply_text(
        "✅ *Media received!*\n\n"
        "🎨 Choose your caption style:",
        reply_markup=style_keyboard(),
        parse_mode="Markdown",
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

    data = query.data or ""

    # ========================================================
    # STYLE
    # ========================================================

    if data.startswith("style:"):

        style = data.split(
            ":",
            1
        )[1]

        if style not in CAPTION_STYLES:

            return

        context.user_data[
            "style"
        ] = style

        if style == "custom":

            context.user_data[
                "stage"
            ] = "custom"

            await query.message.reply_text(
                "✍️ *Enter your custom caption style.*\n\n"
                "Example:\n"
                "Create a classy emotional Instagram caption "
                "with a poetic feel.",
                parse_mode="Markdown",
            )

            return

        context.user_data[
            "stage"
        ] = "language"

        await query.message.reply_text(
            "🌐 *Choose language:*",
            reply_markup=language_keyboard(),
            parse_mode="Markdown",
        )

        return

    # ========================================================
    # LANGUAGE
    # ========================================================

    if data.startswith("language:"):

        language = data.split(
            ":",
            1
        )[1]

        if language not in LANGUAGES:

            return

        context.user_data[
            "language"
        ] = language

        context.user_data[
            "stage"
        ] = "generating"

        await generate_caption(
            query.message,
            context,
        )

        return

    # ========================================================
    # REGENERATE
    # ========================================================

    if data == "action:regenerate":

        await generate_caption(
            query.message,
            context,
        )

        return

    # ========================================================
    # NEW STYLE
    # ========================================================

    if data == "action:style":

        context.user_data[
            "stage"
        ] = "style"

        await query.message.reply_text(
            "🎨 *Choose another style:*",
            reply_markup=style_keyboard(),
            parse_mode="Markdown",
        )

        return

    # ========================================================
    # NEW LANGUAGE
    # ========================================================

    if data == "action:language":

        context.user_data[
            "stage"
        ] = "language"

        await query.message.reply_text(
            "🌐 *Choose another language:*",
            reply_markup=language_keyboard(),
            parse_mode="Markdown",
        )

        return

    # ========================================================
    # NEW MEDIA
    # ========================================================

    if data == "action:new":

        context.user_data.clear()

        await query.message.reply_text(
            "📤 Send a new photo or video."
        )

        return

    # ========================================================
    # VIDEO SUBTITLES
    # ========================================================

    if data == "video:subtitles":

        await process_video_subtitles(
            query.message,
            context,
            include_caption=False,
        )

        return

    # ========================================================
    # VIDEO BOTH
    # ========================================================

    if data == "video:both":

        await process_video_subtitles(
            query.message,
            context,
            include_caption=True,
        )

        return


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    stage = context.user_data.get(
        "stage"
    )

    if stage == "custom":

        custom = (
            update.message.text or ""
        ).strip()

        if not custom:

            await update.message.reply_text(
                "Please enter a custom style."
            )

            return

        context.user_data[
            "custom_style"
        ] = custom

        context.user_data[
            "stage"
        ] = "language"

        await update.message.reply_text(
            "🌐 *Choose language:*",
            reply_markup=language_keyboard(),
            parse_mode="Markdown",
        )

        return

    await update.message.reply_text(
        "📤 Send a photo or video.\n\n"
        "Use /help for instructions."
    )


# ============================================================
# DOWNLOAD MEDIA
# ============================================================

async def download_media(
    message,
    context,
):

    file_id = context.user_data.get(
        "file_id"
    )

    media_type = context.user_data.get(
        "media_type"
    )

    if not file_id:

        raise RuntimeError(
            "No media file found."
        )

    telegram_file = await context.bot.get_file(
        file_id
    )

    suffix = (
        ".jpg"
        if media_type == "photo"
        else ".mp4"
    )

    temporary = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    )

    temporary.close()

    path = Path(
        temporary.name
    )

    await telegram_file.download_to_drive(
        custom_path=str(path)
    )

    if not path.exists():

        raise RuntimeError(
            "Media download failed."
        )

    if path.stat().st_size <= 0:

        raise RuntimeError(
            "Downloaded media is empty."
        )

    return path


# ============================================================
# MIME
# ============================================================

def image_mime_type(
    path: Path
):

    mime, _ = mimetypes.guess_type(
        str(path)
    )

    if mime and mime.startswith(
        "image/"
    ):

        return mime

    return "image/jpeg"


# ============================================================
# FFPROBE DURATION
# ============================================================

def video_duration(
    path: Path
):

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
        )

    except Exception:

        return 0

    if result.returncode != 0:

        return 0

    try:

        return float(
            result.stdout.strip()
        )

    except Exception:

        return 0


# ============================================================
# EXTRACT VIDEO FRAMES
# ============================================================

def extract_video_frames(
    video_path: Path,
    output_directory: Path,
):

    duration = video_duration(
        video_path
    )

    if duration <= 0:

        duration = MAX_VIDEO_SECONDS

    duration = min(
        duration,
        MAX_VIDEO_SECONDS,
    )

    if duration <= 2:

        timestamps = [0]

    else:

        timestamps = []

        for i in range(
            VIDEO_FRAME_COUNT
        ):

            fraction = (
                i /
                max(
                    VIDEO_FRAME_COUNT - 1,
                    1,
                )
            )

            timestamp = (
                duration * fraction
            )

            if timestamp >= duration:

                timestamp = max(
                    duration - 0.2,
                    0,
                )

            timestamps.append(
                timestamp
            )

    frames = []

    for index, timestamp in enumerate(
        timestamps
    ):

        output_file = (
            output_directory /
            f"frame_{index:02d}.jpg"
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale='min(1280,iw)':-2",
            "-q:v",
            "5",
            str(output_file),
        ]

        try:

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
            )

        except subprocess.TimeoutExpired:

            continue

        if (
            result.returncode == 0
            and output_file.exists()
            and output_file.stat().st_size > 0
        ):

            frames.append(
                output_file
            )

    return frames


# ============================================================
# EXTRACT AUDIO
# ============================================================

def extract_audio(
    video_path: Path,
    audio_path: Path,
):

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Could not extract audio."
        )

    if not audio_path.exists():

        raise RuntimeError(
            "Audio file was not created."
        )


# ============================================================
# WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(
    audio_path: Path,
    language_key: str,
):

    language_info = LANGUAGES.get(
        language_key,
        LANGUAGES["english"]
    )

    language_code = (
        language_info["whisper"]
    )

    logger.info(
        "Starting transcription. "
        "Requested language=%s",
        language_key,
    )

    segments, info = (
        whisper_model.transcribe(
            str(audio_path),
            language=language_code,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500
            },
        )
    )

    transcript_segments = []

    all_words = []

    detected_language = (
        getattr(
            info,
            "language",
            None
        )
    )

    for segment in segments:

        segment_text = (
            segment.text or ""
        ).strip()

        if not segment_text:

            continue

        words = []

        if segment.words:

            for word in segment.words:

                word_text = (
                    word.word or ""
                ).strip()

                if not word_text:

                    continue

                word_data = {
                    "text": word_text,
                    "start": float(
                        word.start or 0
                    ),
                    "end": float(
                        word.end or 0
                    ),
                }

                words.append(
                    word_data
                )

                all_words.append(
                    word_data
                )

        transcript_segments.append({
            "start": float(
                segment.start
            ),
            "end": float(
                segment.end
            ),
            "text": segment_text,
            "words": words,
        })

    return {
        "language": detected_language,
        "segments": transcript_segments,
        "words": all_words,
    }


# ============================================================
# BUILD CAPTION PROMPT
# ============================================================

def build_caption_prompt(
    context,
    transcript=""
):

    style_key = context.user_data.get(
        "style",
        "instagram",
    )

    language_key = context.user_data.get(
        "language",
        "english",
    )

    style = CAPTION_STYLES.get(
        style_key,
        CAPTION_STYLES["instagram"],
    )

    language = LANGUAGES.get(
        language_key,
        LANGUAGES["english"],
    )

    custom_style = context.user_data.get(
        "custom_style",
        "",
    )

    original_caption = context.user_data.get(
        "original_caption",
        "",
    )

    if style_key == "custom":

        style_instruction = (
            custom_style
            or "Create an engaging Instagram caption."
        )

    else:

        style_instruction = style[
            "instruction"
        ]

    prompt = f"""
You are an expert Instagram and social media
caption writer.

Analyze the supplied media carefully.

STYLE:
{style["name"]}

STYLE INSTRUCTION:
{style_instruction}

LANGUAGE:
{language["name"]}

LANGUAGE INSTRUCTION:
{language["instruction"]}

USER DESCRIPTION:
{original_caption or "No description provided."}

VIDEO TRANSCRIPT:
{transcript or "No speech detected."}

IMPORTANT:

- Accurately describe what is visible.
- Use the transcript only as supporting context.
- Do not invent people, places or events.
- Do not identify unknown people.
- Do not claim facts that are not supported.
- Make the caption natural.
- Use suitable emojis.
- Generate exactly 12 relevant hashtags.
- Avoid generic spam hashtags.
- Do not copy song lyrics.
- Bollywood style must be original.
- Do not mention AI.
- Do not mention Gemini.
- Do not explain your reasoning.

OUTPUT:

📸 CAPTION

[main caption]


#️⃣ HASHTAGS

[12 hashtags]


✨ SHORT CAPTION

[short caption]
"""

    return prompt


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_with_gemini(
    prompt,
    image_paths=None,
):

    parts = [
        types.Part.from_text(
            text=prompt
        )
    ]

    for image_path in (
        image_paths or []
    ):

        data = image_path.read_bytes()

        if not data:

            continue

        parts.append(
            types.Part.from_bytes(
                data=data,
                mime_type="image/jpeg",
            )
        )

    response = (
        gemini_client
        .models
        .generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=parts,
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.9,
                max_output_tokens=1200,
            ),
        )
    )

    if not response:

        raise RuntimeError(
            "Gemini returned no response."
        )

    text = response.text

    if not text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return clean_text(
        text
    )


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(
    text
):

    text = (
        text or ""
    ).strip()

    text = re.sub(
        r"^```(?:text)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    return text.strip()


# ============================================================
# GEMINI ERROR
# ============================================================

def friendly_gemini_error(
    error
):

    text = str(
        error
    )

    lower = text.lower()

    if (
        "quota" in lower
        or "resource_exhausted" in lower
        or "429" in lower
        or "rate limit" in lower
    ):

        return (
            "⏳ *Gemini free-tier limit reached.*\n\n"
            "Please wait for the quota to reset "
            "and try again."
        )

    if (
        "api key" in lower
        or "unauthenticated" in lower
        or "permission denied" in lower
    ):

        return (
            "🔐 *Gemini API key problem.*\n\n"
            "Check your `GEMINI_API_KEY` "
            "Railway variable."
        )

    if (
        "not_found" in lower
        or "model not found" in lower
    ):

        return (
            "⚠️ *Gemini model unavailable.*\n\n"
            "Check your `GEMINI_MODEL` variable."
        )

    return (
        "❌ *AI generation failed.*\n\n"
        "Please try again later."
    )


# ============================================================
# GENERATE PHOTO CAPTION
# ============================================================

async def generate_caption(
    message,
    context,
):

    media_type = context.user_data.get(
        "media_type"
    )

    if media_type == "video":

        await generate_video_caption(
            message,
            context,
        )

        return

    media_path = None
    processing_message = None

    try:

        processing_message = (
            await message.reply_text(
                "⏳ *Analyzing your photo...*\n\n"
                "🤖 Gemini is writing your caption.",
                parse_mode="Markdown",
            )
        )

        media_path = await download_media(
            message,
            context,
        )

        prompt = build_caption_prompt(
            context
        )

        result = await asyncio.to_thread(
            generate_with_gemini,
            prompt,
            [media_path],
        )

        context.user_data[
            "last_result"
        ] = result

        context.user_data[
            "stage"
        ] = "result"

        try:

            await processing_message.delete()

        except Exception:

            pass

        await send_long_message(
            message,
            result,
            photo_result_keyboard(),
        )

    except Exception as error:

        logger.exception(
            "Photo caption failed."
        )

        if processing_message:

            try:

                await processing_message.delete()

            except Exception:

                pass

        await message.reply_text(
            friendly_gemini_error(
                error
            ),
            parse_mode="Markdown",
        )

    finally:

        if media_path:

            try:

                media_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass


# ============================================================
# GENERATE VIDEO CAPTION
# ============================================================

async def generate_video_caption(
    message,
    context,
):

    video_path = None
    processing_message = None
    work_directory = None

    try:

        processing_message = (
            await message.reply_text(
                "⏳ *Analyzing video...*\n\n"
                "🎬 Extracting video frames...",
                parse_mode="Markdown",
            )
        )

        video_path = await download_media(
            message,
            context,
        )

        duration = video_duration(
            video_path
        )

        if (
            duration
            and duration > MAX_VIDEO_SECONDS
        ):

            await processing_message.edit_text(
                f"⚠️ This video is {duration:.0f} seconds long.\n\n"
                f"I currently process videos up to "
                f"{MAX_VIDEO_SECONDS} seconds.\n\n"
                "Please send a shorter clip."
            )

            return

        work_directory = Path(
            tempfile.mkdtemp(
                prefix="caption_video_"
            )
        )

        frames_directory = (
            work_directory /
            "frames"
        )

        frames_directory.mkdir()

        frames = extract_video_frames(
            video_path,
            frames_directory,
        )

        if not frames:

            raise RuntimeError(
                "Could not extract video frames."
            )

        await processing_message.edit_text(
            "🎙 *Listening to the video...*\n\n"
            "Whisper is generating the transcript.",
            parse_mode="Markdown",
        )

        audio_path = (
            work_directory /
            "audio.wav"
        )

        extract_audio(
            video_path,
            audio_path,
        )

        language = context.user_data.get(
            "language",
            "english",
        )

        transcript_data = await asyncio.to_thread(
            transcribe_audio,
            audio_path,
            language,
        )

        transcript_text = "\n".join(
            segment["text"]
            for segment in transcript_data[
                "segments"
            ]
        )

        logger.info(
            "Detected language: %s",
            transcript_data.get(
                "language"
            ),
        )

        await processing_message.edit_text(
            "🤖 *Creating caption...*\n\n"
            "Gemini is combining the video "
            "and transcript.",
            parse_mode="Markdown",
        )

        prompt = build_caption_prompt(
            context,
            transcript_text,
        )

        result = await asyncio.to_thread(
            generate_with_gemini,
            prompt,
            frames,
        )

        context.user_data[
            "last_result"
        ] = result

        context.user_data[
            "transcript"
        ] = transcript_data

        context.user_data[
            "video_path"
        ] = str(video_path)

        context.user_data[
            "stage"
        ] = "video_result"

        try:

            await processing_message.delete()

        except Exception:

            pass

        await send_long_message(
            message,
            result,
            video_result_keyboard(),
        )

    except Exception as error:

        logger.exception(
            "Video caption failed."
        )

        if processing_message:

            try:

                await processing_message.delete()

            except Exception:

                pass

        error_message = (
            "❌ *Video processing failed.*\n\n"
            f"`{str(error)[:800]}`"
        )

        await message.reply_text(
            error_message,
            parse_mode="Markdown",
        )

    finally:

        if video_path:

            try:

                video_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass

        if work_directory:

            try:

                shutil.rmtree(
                    work_directory,
                    ignore_errors=True,
                )

            except Exception:

                pass


# ============================================================
# VIDEO SUBTITLE PROCESSING
# ============================================================

async def process_video_subtitles(
    message,
    context,
    include_caption=False,
):

    file_id = context.user_data.get(
        "file_id"
    )

    if not file_id:

        await message.reply_text(
            "📤 Please send a video first."
        )

        return

    if (
        context.user_data.get(
            "media_type"
        )
        != "video"
    ):

        await message.reply_text(
            "❌ Subtitle processing requires a video."
        )

        return

    processing_message = None
    video_path = None
    work_directory = None

    try:

        processing_message = (
            await message.reply_text(
                "🎬 *Preparing your captioned video...*\n\n"
                "This can take some time on Railway.",
                parse_mode="Markdown",
            )
        )

        video_path = await download_media(
            message,
            context,
        )

        duration = video_duration(
            video_path
        )

        if (
            duration
            and duration > MAX_VIDEO_SECONDS
        ):

            await processing_message.edit_text(
                f"⚠️ Video is {duration:.0f} seconds.\n\n"
                f"Maximum supported duration is "
                f"{MAX_VIDEO_SECONDS} seconds."
            )

            return

        work_directory = Path(
            tempfile.mkdtemp(
                prefix="subtitle_job_"
            )
        )

        audio_path = (
            work_directory /
            "audio.wav"
        )

        extract_audio(
            video_path,
            audio_path,
        )

        await processing_message.edit_text(
            "🎙 *Generating subtitles...*\n\n"
            "Whisper is transcribing the speech.",
            parse_mode="Markdown",
        )

        language = context.user_data.get(
            "language",
            "english",
        )

        transcript_data = await asyncio.to_thread(
            transcribe_audio,
            audio_path,
            language,
        )

        if not transcript_data[
            "segments"
        ]:

            await processing_message.edit_text(
                "❌ No speech was detected in this video."
            )

            return

        subtitle_file = (
            work_directory /
            "captions.ass"
        )

        create_ass_subtitles(
            transcript_data,
            subtitle_file,
            language,
        )

        output_file = (
            work_directory /
            "captioned_video.mp4"
        )

        await processing_message.edit_text(
            "🎬 *Burning captions into video...*\n\n"
            "FFmpeg is rendering the final MP4.",
            parse_mode="Markdown",
        )

        burn_subtitles(
            video_path,
            subtitle_file,
            output_file,
        )

        if not output_file.exists():

            raise RuntimeError(
                "Captioned video was not created."
            )

        if (
            output_file.stat().st_size
            > 49 * 1024 * 1024
        ):

            await processing_message.edit_text(
                "⚠️ The captioned video is too large "
                "to send through Telegram.\n\n"
                "Try a shorter or lower-resolution video."
            )

            return

        if include_caption:

            caption = context.user_data.get(
                "last_result"
            )

            if not caption:

                prompt = build_caption_prompt(
                    context,
                    "\n".join(
                        segment["text"]
                        for segment
                        in transcript_data[
                            "segments"
                        ]
                    ),
                )

                caption = await asyncio.to_thread(
                    generate_with_gemini,
                    prompt,
                    [],
                )

                context.user_data[
                    "last_result"
                ] = caption

        try:

            await processing_message.delete()

        except Exception:

            pass

        await send_captioned_video(
            message,
            output_file,
            context.user_data.get(
                "last_result"
            )
            if include_caption
            else None,
        )

    except Exception as error:

        logger.exception(
            "Subtitle processing failed."
        )

        if processing_message:

            try:

                await processing_message.delete()

            except Exception:

                pass

        await message.reply_text(
            "❌ *Subtitle processing failed.*\n\n"
            f"`{str(error)[:800]}`",
            parse_mode="Markdown",
        )

    finally:

        if video_path:

            try:

                video_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass

        if work_directory:

            try:

                shutil.rmtree(
                    work_directory,
                    ignore_errors=True,
                )

            except Exception:

                pass


# ============================================================
# ASS TIME FORMAT
# ============================================================

def ass_time(
    seconds
):

    seconds = max(
        float(seconds),
        0,
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = (
        seconds
        % 60
    )

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{secs:05.2f}"
    )


# ============================================================
# ESCAPE ASS TEXT
# ============================================================

def escape_ass(
    text
):

    text = (
        text
        .replace(
            "\\",
            r"\\",
        )
        .replace(
            "{",
            r"\{",
        )
        .replace(
            "}",
            r"\}",
        )
    )

    return text


# ============================================================
# CREATE ASS SUBTITLES
# ============================================================

def create_ass_subtitles(
    transcript_data,
    output_file,
    language_key,
):

    # --------------------------------------------------------
    # Font
    # --------------------------------------------------------

    if language_key == "hindi":

        font_name = "Noto Sans Devanagari"

    elif language_key == "punjabi":

        font_name = "Noto Sans Gurmukhi"

    else:

        font_name = "Arial"

    # --------------------------------------------------------
    # ASS header
    # --------------------------------------------------------

    lines = [

        "[Script Info]",

        "ScriptType: v4.00+",

        "PlayResX: 1920",

        "PlayResY: 1080",

        "ScaledBorderAndShadow: yes",

        "",

        "[V4+ Styles]",

        "Format: Name, Fontname, Fontsize, "
        "PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",

        (
            f"Style: Default,"
            f"{font_name},"
            f"62,"
            f"&H00FFFFFF,"
            f"&H00FFFFFF,"
            f"&H00000000,"
            f"&H99000000,"
            f"-1,0,0,0,"
            f"100,100,0,0,"
            f"1,4,1,"
            f"2,100,100,80,1"
        ),

        "",

        "[Events]",

        "Format: Layer, Start, End, Style, "
        "Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]

    # --------------------------------------------------------
    # Build subtitles
    # --------------------------------------------------------

    for segment in transcript_data[
        "segments"
    ]:

        start = ass_time(
            segment["start"]
        )

        end = ass_time(
            segment["end"]
        )

        text = escape_ass(
            segment["text"]
        )

        # Word wrapping.
        text = (
            text.replace(
                "\n",
                r"\N"
            )
        )

        lines.append(
            "Dialogue: 0,"
            f"{start},"
            f"{end},"
            "Default,,0,0,0,,"
            f"{text}"
        )

    output_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# BURN SUBTITLES
# ============================================================

def burn_subtitles(
    video_path,
    subtitle_file,
    output_file,
):

    # FFmpeg subtitles filter expects a properly escaped path.
    subtitle_path = (
        str(subtitle_file)
        .replace(
            "\\",
            "/"
        )
        .replace(
            ":",
            "\\:"
        )
        .replace(
            "'",
            "\\'"
        )
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"subtitles='{subtitle_path}'",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_file),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:

        logger.error(
            "FFmpeg error: %s",
            result.stderr[-4000:],
        )

        raise RuntimeError(
            "FFmpeg could not burn subtitles."
        )


# ============================================================
# SEND VIDEO
# ============================================================

async def send_captioned_video(
    message,
    video_path,
    caption=None,
):

    if caption:

        # Telegram caption limit is much smaller than
        # a normal message, so only use the short part.
        short_caption = extract_short_caption(
            caption
        )

    else:

        short_caption = (
            "🎬 Captioned video generated successfully."
        )

    await message.reply_video(
        video=str(video_path),
        caption=short_caption,
        supports_streaming=True,
    )

    if caption:

        await send_long_message(
            message,
            caption,
            photo_result_keyboard(),
        )


# ============================================================
# EXTRACT SHORT CAPTION
# ============================================================

def extract_short_caption(
    text
):

    match = re.search(
        r"✨\s*SHORT CAPTION\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:

        result = (
            match.group(1)
            .strip()
        )

        if result:

            return result[:900]

    return (
        "🎬 Captioned video generated successfully."
    )


# ============================================================
# SEND LONG MESSAGE
# ============================================================

async def send_long_message(
    message,
    text,
    keyboard=None,
):

    max_length = 3900

    if len(text) <= max_length:

        await message.reply_text(
            text,
            reply_markup=keyboard,
        )

        return

    chunks = []

    current = ""

    for paragraph in text.split(
        "\n"
    ):

        proposed = (
            current
            + "\n"
            + paragraph
        ).strip()

        if len(proposed) > max_length:

            if current:

                chunks.append(
                    current
                )

            current = paragraph

        else:

            current = proposed

    if current:

        chunks.append(
            current
        )

    for index, chunk in enumerate(
        chunks
    ):

        if (
            index
            == len(chunks) - 1
        ):

            await message.reply_text(
                chunk,
                reply_markup=keyboard,
            )

        else:

            await message.reply_text(
                chunk
            )


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update,
    context: ContextTypes.DEFAULT_TYPE,
):

    logger.exception(
        "Unhandled application error",
        exc_info=context.error,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "========================================"
    )

    logger.info(
        "Media Caption Generator"
    )

    logger.info(
        "Gemini model: %s",
        GEMINI_MODEL,
    )

    logger.info(
        "Whisper model: %s",
        WHISPER_MODEL_NAME,
    )

    logger.info(
        "Maximum video duration: %s seconds",
        MAX_VIDEO_SECONDS,
    )

    logger.info(
        "========================================"
    )

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    # --------------------------------------------------------
    # COMMANDS
    # --------------------------------------------------------

    application.add_handler(
        CommandHandler(
            "start",
            start,
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
            "cancel",
            cancel,
        )
    )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    application.add_handler(
        CallbackQueryHandler(
            callback_handler,
        )
    )

    # --------------------------------------------------------
    # PHOTO
    # --------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            media_handler,
        )
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.VIDEO,
            media_handler,
        )
    )

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.Document.ALL,
            media_handler,
        )
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_handler,
        )
    )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    application.add_error_handler(
        error_handler
    )

    logger.info(
        "Bot is running."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
