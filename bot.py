import os
import re
import shutil
import asyncio
import json
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

REQUESTED_GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    ""
).strip()

WHISPER_MODEL_NAME = os.getenv(
    "WHISPER_MODEL",
    "base"
).strip()

MAX_VIDEO_SECONDS = int(
    os.getenv(
        "MAX_VIDEO_SECONDS",
        "180"
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
        "BOT_TOKEN is missing. "
        "Add BOT_TOKEN to Railway Variables."
    )


if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Add GEMINI_API_KEY to Railway Variables."
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

logger.info(
    "Initializing Gemini client..."
)

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# GEMINI MODEL DISCOVERY
# ============================================================

def normalize_model_name(
    name
):

    if not name:
        return ""

    name = str(name).strip()

    if name.startswith("models/"):

        name = name[
            len("models/"):
        ]

    return name


def model_supports_generate_content(
    model
):

    actions = getattr(
        model,
        "supported_actions",
        None
    )

    if actions:

        for action in actions:

            if str(action).lower() == (
                "generatecontent"
            ):

                return True

    # Some SDK versions expose the
    # information differently.
    return True


def find_gemini_model():

    logger.info(
        "Checking Gemini models available "
        "to the API key..."
    )

    try:

        available_models = []

        for model in gemini_client.models.list():

            model_name = normalize_model_name(
                getattr(
                    model,
                    "name",
                    ""
                )
            )

            if not model_name:

                continue

            if not model_supports_generate_content(
                model
            ):

                continue

            available_models.append(
                model_name
            )

        available_models = sorted(
            set(available_models)
        )

        logger.info(
            "Models available for generateContent:"
        )

        for model_name in available_models:

            logger.info(
                "  - %s",
                model_name
            )

        if not available_models:

            raise RuntimeError(
                "The Gemini API key does not have "
                "any generateContent models available."
            )

        # ----------------------------------------------------
        # If user explicitly specified a model,
        # try it first.
        # ----------------------------------------------------

        if REQUESTED_GEMINI_MODEL:

            requested = normalize_model_name(
                REQUESTED_GEMINI_MODEL
            )

            if requested in available_models:

                logger.info(
                    "Using requested Gemini model: %s",
                    requested
                )

                return requested

            logger.warning(
                "Requested model '%s' is not "
                "available. Searching automatically.",
                requested
            )

        # ----------------------------------------------------
        # Preferred models.
        #
        # These are ordered from newest/general purpose
        # toward older lightweight models.
        # ----------------------------------------------------

        preferred_models = [

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

        for preferred in preferred_models:

            if preferred in available_models:

                logger.info(
                    "Automatically selected Gemini model: %s",
                    preferred
                )

                return preferred

        # ----------------------------------------------------
        # Generic Flash fallback.
        # ----------------------------------------------------

        flash_models = []

        for model_name in available_models:

            lower = model_name.lower()

            if (
                "flash" in lower
                and "embedding" not in lower
                and "tts" not in lower
                and "image" not in lower
                and "live" not in lower
            ):

                flash_models.append(
                    model_name
                )

        if flash_models:

            # Prefer models that are not preview/experimental.
            stable = [
                model
                for model in flash_models
                if "preview" not in model.lower()
                and "exp" not in model.lower()
            ]

            if stable:

                flash_models = stable

            flash_models.sort(
                reverse=True
            )

            selected = flash_models[0]

            logger.info(
                "Selected Flash fallback: %s",
                selected
            )

            return selected

        # ----------------------------------------------------
        # Last resort: first generateContent model.
        # ----------------------------------------------------

        selected = available_models[0]

        logger.warning(
            "No preferred Flash model found. "
            "Using: %s",
            selected
        )

        return selected

    except Exception as error:

        logger.exception(
            "Gemini model discovery failed."
        )

        raise RuntimeError(
            "Unable to retrieve Gemini models. "
            "Please check GEMINI_API_KEY and "
            "your Google AI Studio project."
        ) from error


GEMINI_MODEL = find_gemini_model()


logger.info(
    "FINAL GEMINI MODEL: %s",
    GEMINI_MODEL
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
    "Whisper model loaded successfully."
)


# ============================================================
# CAPTION STYLES
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
            "on journey, atmosphere, adventure or location."
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
            "caption. Never copy song lyrics."
        ),
    },

    "podcast": {
        "name": "Podcast",
        "instruction": (
            "Create a strong podcast/reel caption from the spoken content. "
            "Start with a compelling hook, summarize the key idea, and "
            "include relevant hashtags. Keep it natural and social-media friendly."
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
            "Write the caption in natural English."
        ),
    },

    "hindi": {
        "name": "Hindi",
        "whisper": "hi",
        "instruction": (
            "Write primarily in Hindi using Devanagari script."
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
# SAFE TELEGRAM MESSAGE EDIT
# ============================================================

async def safe_edit_message(message, text, **kwargs):
    """
    Safely edit a Telegram message.

    Telegram raises BadRequest("Message is not modified") when
    edit_text() is called with exactly the same text and markup.
    Ignore only that specific condition; all other errors are
    re-raised so real problems are not hidden.
    """

    try:
        current_text = getattr(message, "text", None)
        current_markup = getattr(
            message,
            "reply_markup",
            None,
        )
        requested_markup = kwargs.get(
            "reply_markup",
            None,
        )

        if (
            current_text == text
            and current_markup == requested_markup
        ):
            return message

        return await message.edit_text(
            text,
            **kwargs,
        )

    except Exception as error:
        if "Message is not modified" in str(error):
            return message
        raise


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
        "• Video subtitles\n"
        "• Captioned videos\n\n"
        "🤖 Gemini AI\n"
        "🎙 Whisper speech recognition\n"
        "🎙 Podcast mode with word-by-word animated captions\n"
        "🎬 FFmpeg video processing\n\n"
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
        "Send photo → choose style → choose language "
        "→ receive caption + hashtags.\n\n"

        "🎥 *VIDEO*\n"
        "Send video → choose style → choose language.\n\n"

        "Then choose:\n"
        "🤖 AI Caption\n"
        "📝 Subtitles\n"
        "🎬 Caption + Subtitles\n\n"

        "🌐 Languages:\n"
        "English\n"
        "Hindi\n"
        "Punjabi\n\n"

        "Commands:\n"
        "/start\n"
        "/help\n"
        "/cancel"
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
                "🎙️ Podcast",
                callback_data="style:podcast",
            ),
            InlineKeyboardButton(
                "✍️ Custom",
                callback_data="style:custom",
            ),
        ],
    ])


# ============================================================
# VIDEO MODE KEYBOARD
# ============================================================

def video_mode_keyboard():

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🎥 Normal Video Caption",
                callback_data="video_mode:normal",
            ),
        ],
        [
            InlineKeyboardButton(
                "🎙 Podcast Caption + Subtitles",
                callback_data="video_mode:podcast",
            ),
        ],
        [
            InlineKeyboardButton(
                "❌ Cancel",
                callback_data="action:new",
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
                "English",
                callback_data="language:english",
            ),
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
    ]])


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
                "🎙️ Podcast Animated Captions",
                callback_data="video:podcast",
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

    if message.photo:

        photo = message.photo[-1]

        media_type = "photo"
        file_id = photo.file_id
        file_size = photo.file_size

    elif message.video:

        video = message.video

        media_type = "video"
        file_id = video.file_id
        file_size = video.file_size

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

    if (
        file_size
        and file_size > MAX_FILE_SIZE
    ):

        await message.reply_text(
            "❌ File is too large.\n\n"
            "Please send a file smaller than 20 MB."
        )

        return

    context.user_data.clear()

    context.user_data[
        "file_id"
    ] = file_id

    context.user_data[
        "media_type"
    ] = media_type

    context.user_data[
        "original_caption"
    ] = message.caption or ""

    if media_type == "video":

        context.user_data[
            "stage"
        ] = "video_mode"

        await message.reply_text(
            "🎥 *Video received!*\n\n"
            "How would you like to process this video?",
            reply_markup=video_mode_keyboard(),
            parse_mode="Markdown",
        )

    else:

        context.user_data[
            "stage"
        ] = "style"

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

    # --------------------------------------------------------
    # VIDEO MODE
    # --------------------------------------------------------

    if data.startswith("video_mode:"):

        mode = data.split(":", 1)[1]

        if mode == "normal":
            context.user_data["video_mode"] = "normal"
            context.user_data["stage"] = "style"

            await query.message.reply_text(
                "🎨 *Choose your video caption style:*",
                reply_markup=style_keyboard(),
                parse_mode="Markdown",
            )
            return

        if mode == "podcast":
            context.user_data["video_mode"] = "podcast"
            context.user_data["style"] = "podcast"
            context.user_data["stage"] = "language"

            await query.message.reply_text(
                "🎙 *Podcast mode selected.*\n\n"
                "Choose the caption language:",
                reply_markup=language_keyboard(),
                parse_mode="Markdown",
            )
            return

        return

    # --------------------------------------------------------
    # STYLE
    # --------------------------------------------------------

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
                "Create a classy emotional Instagram "
                "caption with a poetic feel.",
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

    # --------------------------------------------------------
    # LANGUAGE
    # --------------------------------------------------------

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

        media_type = context.user_data.get("media_type")

        if media_type == "photo":
            await generate_photo_caption(query.message, context)
        elif media_type == "video":
            if context.user_data.get("style") == "podcast":
                await process_video_subtitles(
                    query.message,
                    context,
                    include_caption=True,
                    podcast_mode=True,
                )
            else:
                await generate_video_caption(query.message, context)
        else:
            await query.message.reply_text(
                "❌ Media information was lost. Please send the media again."
            )

        return

    # --------------------------------------------------------
    # REGENERATE
    # --------------------------------------------------------

    if data == "action:regenerate":

        media_type = context.user_data.get("media_type")

        if media_type == "photo":
            await generate_photo_caption(query.message, context)
        elif media_type == "video":
            await generate_video_caption(query.message, context)
        else:
            await query.message.reply_text(
                "❌ Please send the media again."
            )

        return

    # --------------------------------------------------------
    # NEW STYLE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NEW LANGUAGE
    # --------------------------------------------------------

    if data == "action:language":

        context.user_data[
            "stage"
        ] = "language"

        await query.message.reply_text(
            "🌐 *Choose language:*",
            reply_markup=language_keyboard(),
            parse_mode="Markdown",
        )

        return

    # --------------------------------------------------------
    # NEW MEDIA
    # --------------------------------------------------------

    if data == "action:new":

        context.user_data.clear()

        await query.message.reply_text(
            "📤 Send a new photo or video."
        )

        return

    # --------------------------------------------------------
    # PODCAST CAPTIONED VIDEO
    # --------------------------------------------------------

    if data == "video:podcast":

        context.user_data["style"] = "podcast"

        await process_video_subtitles(
            query.message,
            context,
            include_caption=True,
            podcast_mode=True,
        )

        return

    # --------------------------------------------------------
    # VIDEO SUBTITLES
    # --------------------------------------------------------

    if data == "video:subtitles":

        await process_video_subtitles(
            query.message,
            context,
            include_caption=False,
        )

        return

    # --------------------------------------------------------
    # VIDEO BOTH
    # --------------------------------------------------------

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
# VIDEO DURATION
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
        MAX_VIDEO_SECONDS
    )

    if duration <= 2:

        timestamps = [0]

    else:

        timestamps = []

        for index in range(
            VIDEO_FRAME_COUNT
        ):

            fraction = (
                index
                / max(
                    VIDEO_FRAME_COUNT - 1,
                    1
                )
            )

            timestamp = (
                duration
                * fraction
            )

            if timestamp >= duration:

                timestamp = max(
                    duration - 0.2,
                    0
                )

            timestamps.append(
                timestamp
            )

    frames = []

    for index, timestamp in enumerate(
        timestamps
    ):

        output_file = (
            output_directory
            / f"frame_{index:02d}.jpg"
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

            logger.warning(
                "Frame extraction timeout."
            )

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
    """Extract the first decodable audio stream as 16 kHz mono WAV."""

    if not video_path.exists() or video_path.stat().st_size == 0:
        raise RuntimeError("The downloaded video is missing or empty.")

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    # FFprobe is used for diagnostics only. FFmpeg performs the actual
    # extraction because some MP4/MOV files have unusual stream metadata.
    probe_command = [
        "ffprobe", "-v", "error",
        "-show_entries",
        "stream=index,codec_type,codec_name:format=duration",
        "-of", "json",
        str(video_path),
    ]

    try:
        probe = subprocess.run(
            probe_command, capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        probe = None
        logger.warning("FFprobe timed out for %s", video_path)
    except FileNotFoundError:
        raise RuntimeError("FFmpeg/FFprobe is not installed on the server.")

    if probe is not None and probe.returncode == 0:
        try:
            probe_data = json.loads(probe.stdout or "{}")
            logger.info("Media streams: %s", probe_data.get("streams", []))
        except json.JSONDecodeError:
            logger.warning("Could not parse FFprobe output.")
    elif probe is not None:
        logger.warning("FFprobe diagnostic failed: %s", (probe.stderr or "")[-2000:])

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        "-map", "0:a:0?",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        str(audio_path),
    ]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out while extracting audio from the video.")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg is not installed on the Railway server.")

    details = (result.stderr or "").strip()

    if result.returncode != 0:
        logger.error("FFmpeg audio extraction failed:\n%s", details[-5000:])
        lower = details.lower()
        if "matches no streams" in lower or "does not contain any stream" in lower:
            raise RuntimeError(
                "This video does not contain a readable audio track. "
                "Please send the original video with its audio enabled."
            )
        raise RuntimeError(
            "FFmpeg could not decode the video's audio track. "
            "Please send the original MP4/MOV video."
        )

    if not audio_path.exists() or audio_path.stat().st_size < 1024:
        logger.error("No usable WAV created. FFmpeg: %s", details[-3000:])
        raise RuntimeError(
            "The video contains no readable audio track. "
            "Please send a video with speech/audio."
        )

    logger.info(
        "Audio extracted successfully: %s (%d bytes)",
        audio_path, audio_path.stat().st_size
    )


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

    whisper_language = (
        language_info["whisper"]
    )

    logger.info(
        "Transcribing audio. Language=%s",
        whisper_language or "auto"
    )

    segments, info = (
        whisper_model.transcribe(
            str(audio_path),
            language=whisper_language,
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

    detected_language = getattr(
        info,
        "language",
        None
    )

    for segment in segments:

        text = (
            segment.text or ""
        ).strip()

        if not text:

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
            "text": text,
            "words": words,
        })

    return {
        "language": detected_language,
        "segments": transcript_segments,
        "words": all_words,
    }


# ============================================================
# CAPTION PROMPT
# ============================================================

def build_caption_prompt(
    context,
    transcript=""
):

    style_key = context.user_data.get(
        "style",
        "instagram"
    )

    language_key = context.user_data.get(
        "language",
        "english"
    )

    style = CAPTION_STYLES.get(
        style_key,
        CAPTION_STYLES["instagram"]
    )

    language = LANGUAGES.get(
        language_key,
        LANGUAGES["english"]
    )

    custom_style = context.user_data.get(
        "custom_style",
        ""
    )

    original_caption = context.user_data.get(
        "original_caption",
        ""
    )

    if style_key == "custom":

        style_instruction = (
            custom_style
            or "Create an engaging Instagram caption."
        )

    else:

        style_instruction = (
            style["instruction"]
        )

    prompt = f"""
You are an expert social media caption writer.

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

RULES:

1. Accurately describe what is visible.
2. Use the transcript as supporting context.
3. Never invent people, locations or events.
4. Do not identify unknown people by name.
5. Make the caption natural and human.
6. Use suitable emojis.
7. Generate exactly 12 relevant hashtags.
8. Avoid spam hashtags.
9. Do not copy copyrighted song lyrics.
10. Bollywood style must be original.
11. Do not mention AI.
12. Do not mention Gemini.
13. Do not explain your reasoning.

OUTPUT EXACTLY:

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
                max_output_tokens=1200,
            ),
        )
    )

    if not response:

        raise RuntimeError(
            "Gemini returned no response."
        )

    result = response.text

    if not result:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return clean_text(
        result
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
            "Please wait for your Gemini quota "
            "to reset and try again."
        )

    if (
        "api key" in lower
        or "unauthenticated" in lower
        or "permission denied" in lower
        or "401" in lower
        or "403" in lower
    ):

        return (
            "🔐 *Gemini API key problem.*\n\n"
            "Please check the GEMINI_API_KEY "
            "variable in Railway."
        )

    if (
        "404" in lower
        or "not_found" in lower
        or "model not found" in lower
    ):

        return (
            "⚠️ *Gemini model unavailable.*\n\n"
            "The bot attempted to automatically "
            "find an available model, but the "
            "selected model could not be used."
        )

    return (
        "❌ *Gemini generation failed.*\n\n"
        "Please try again."
    )


# ============================================================
# PHOTO CAPTION
# ============================================================

async def generate_photo_caption(
    message,
    context,
):

    media_path = None
    processing_message = None

    try:

        processing_message = (
            await message.reply_text(
                "⏳ *Analyzing your photo...*\n\n"
                "✍️ Writing your caption",
                parse_mode="Markdown",
            )
        )

        media_path = await download_media(
            message,
            context
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
            photo_result_keyboard()
        )

    except Exception as error:

        logger.exception(
            "Photo caption generation failed."
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
# VIDEO CAPTION
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
                "🎬 Extracting frames...",
                parse_mode="Markdown",
            )
        )

        video_path = await download_media(
            message,
            context
        )

        duration = video_duration(
            video_path
        )

        if (
            duration
            and duration > MAX_VIDEO_SECONDS
        ):

            await safe_edit_message(
                processing_message,
                
                f"⚠️ Video is {duration:.0f} seconds.\n\n"
                f"Maximum supported duration is "
                f"{MAX_VIDEO_SECONDS} seconds."
            )

            return

        work_directory = Path(
            tempfile.mkdtemp(
                prefix="video_caption_"
            )
        )

        frames_directory = (
            work_directory
            / "frames"
        )

        frames_directory.mkdir()

        frames = extract_video_frames(
            video_path,
            frames_directory
        )

        if not frames:

            raise RuntimeError(
                "Unable to extract video frames."
            )

        await safe_edit_message(
                processing_message,
                
            "🎙 *Transcribing video...*\n\n"
            "Whisper is processing the audio.",
            parse_mode="Markdown",
        )

        audio_path = (
            work_directory
            / "audio.wav"
        )

        extract_audio(
            video_path,
            audio_path
        )

        language = context.user_data.get(
            "language",
            "english"
        )

        transcript_data = await asyncio.to_thread(
            transcribe_audio,
            audio_path,
            language
        )

        transcript = "\n".join(
            segment["text"]
            for segment
            in transcript_data["segments"]
        )

        await safe_edit_message(
                processing_message,
                
            "🤖 *Creating caption...*\n\n"
            "✍️ creating your caption.",
            parse_mode="Markdown",
        )

        prompt = build_caption_prompt(
            context,
            transcript
        )

        result = await asyncio.to_thread(
            generate_with_gemini,
            prompt,
            frames
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
            video_result_keyboard()
        )

    except Exception as error:

        logger.exception(
            "Video caption generation failed."
        )

        if processing_message:

            try:

                await processing_message.delete()

            except Exception:

                pass

        await message.reply_text(
            "❌ *Video processing failed.*\n\n"
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

            shutil.rmtree(
                work_directory,
                ignore_errors=True
            )


# ============================================================
# SUBTITLE PROCESSING
# ============================================================

async def process_video_subtitles(
    message,
    context,
    include_caption=False,
    podcast_mode=False,
):

    if (
        context.user_data.get(
            "media_type"
        )
        != "video"
    ):

        await message.reply_text(
            "❌ Please send a video first."
        )

        return

    processing_message = None
    video_path = None
    work_directory = None

    try:

        processing_message = (
            await message.reply_text(
                "🎬 *Preparing captioned video...*\n\n"
                "This may take some time.",
                parse_mode="Markdown",
            )
        )

        video_path = await download_media(
            message,
            context
        )

        duration = video_duration(
            video_path
        )

        if (
            duration
            and duration > MAX_VIDEO_SECONDS
        ):

            await safe_edit_message(
                processing_message,
                
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
            work_directory
            / "audio.wav"
        )

        extract_audio(
            video_path,
            audio_path
        )

        if podcast_mode:
            await safe_edit_message(
                processing_message,
                
                "🎙 *Creating podcast captions...*\n\n"
                "Whisper is generating word-level timings.",
                parse_mode="Markdown",
            )
        else:
            await safe_edit_message(
                processing_message,
                
                "🎙 *Generating subtitles...*\n\n"
                "Whisper is transcribing speech.",
                parse_mode="Markdown",
            )

        language = context.user_data.get(
            "language",
            "english"
        )

        transcript_data = await asyncio.to_thread(
            transcribe_audio,
            audio_path,
            language
        )

        if not transcript_data[
            "segments"
        ]:

            await safe_edit_message(
                processing_message,
                
                "❌ No speech was detected."
            )

            return

        subtitle_file = (
            work_directory
            / "captions.ass"
        )

        if podcast_mode:
            create_karaoke_subtitles(
                transcript_data,
                subtitle_file,
                language
            )
        else:
            create_ass_subtitles(
                transcript_data,
                subtitle_file,
                language
            )

        output_file = (
            work_directory
            / "captioned_video.mp4"
        )

        await safe_edit_message(
                processing_message,
                
            "🎬 *Rendering captions...*\n\n"
            "FFmpeg is creating the final MP4.",
            parse_mode="Markdown",
        )

        burn_subtitles(
            video_path,
            subtitle_file,
            output_file
        )

        if not output_file.exists():

            raise RuntimeError(
                "Captioned video was not created."
            )

        if (
            output_file.stat().st_size
            > 49 * 1024 * 1024
        ):

            await safe_edit_message(
                processing_message,
                
                "⚠️ Captioned video is too large "
                "to send through Telegram."
            )

            return

        caption = None

        if include_caption:

            caption = context.user_data.get(
                "last_result"
            )

            if not caption:

                transcript = "\n".join(
                    segment["text"]
                    for segment
                    in transcript_data[
                        "segments"
                    ]
                )

                prompt = build_caption_prompt(
                    context,
                    transcript
                )

                caption = await asyncio.to_thread(
                    generate_with_gemini,
                    prompt,
                    []
                )

        try:

            await processing_message.delete()

        except Exception:

            pass

        await send_captioned_video(
            message,
            output_file,
            caption
        )

    except Exception as error:

        logger.exception(
            "Subtitle generation failed."
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

            shutil.rmtree(
                work_directory,
                ignore_errors=True
            )


# ============================================================
# ASS TIME
# ============================================================

def ass_time(
    seconds
):

    seconds = max(
        float(seconds),
        0
    )

    hours = int(
        seconds // 3600
    )

    minutes = int(
        (seconds % 3600) // 60
    )

    remaining = (
        seconds
        % 60
    )

    return (
        f"{hours}:"
        f"{minutes:02d}:"
        f"{remaining:05.2f}"
    )


# ============================================================
# ASS ESCAPE
# ============================================================

def escape_ass(
    text
):

    return (
        text
        .replace(
            "\\",
            r"\\"
        )
        .replace(
            "{",
            r"\{"
        )
        .replace(
            "}",
            r"\}"
        )
    )


# ============================================================
# CREATE ASS SUBTITLES
# ============================================================

def create_ass_subtitles(
    transcript_data,
    output_file,
    language_key,
):

    if language_key == "hindi":

        font_name = "Noto Sans Devanagari"

    elif language_key == "punjabi":

        font_name = "Noto Sans Gurmukhi"

    else:

        font_name = "Arial"

    lines = [

        "[Script Info]",

        "ScriptType: v4.00+",

        "PlayResX: 1920",

        "PlayResY: 1080",

        "ScaledBorderAndShadow: yes",

        "",

        "[V4+ Styles]",

        (
            "Format: Name, Fontname, Fontsize, "
            "PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, "
            "Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        ),

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

        (
            "Format: Layer, Start, End, Style, "
            "Name, MarginL, MarginR, MarginV, "
            "Effect, Text"
        ),
    ]

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

        text = text.replace(
            "\n",
            r"\N"
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
        encoding="utf-8"
    )


# ============================================================
# PODCAST KARAOKE SUBTITLES
# ============================================================

def create_karaoke_subtitles(
    transcript_data,
    output_file,
    language_key,
):

    if language_key == "hindi":
        font_name = "Noto Sans Devanagari"
    elif language_key == "punjabi":
        font_name = "Noto Sans Gurmukhi"
    else:
        font_name = "Arial"

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Podcast,{font_name},68,&H00FFFFFF,&H0000FFFF,&H00101010,&H99000000,-1,0,0,0,100,100,0,0,1,4,2,2,120,120,100,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    all_words = transcript_data.get("words", [])
    chunks = []
    chunk = []

    for word in all_words:
        text = (word.get("text") or "").strip()
        start = float(word.get("start", 0))
        end = float(word.get("end", start + 0.2))
        if not text:
            continue
        chunk.append({"text": text, "start": start, "end": max(end, start + 0.05)})
        # 6 words gives a readable podcast/reel rhythm.
        if len(chunk) >= 6 or text.endswith((".", "?", "!", ",")):
            chunks.append(chunk)
            chunk = []

    if chunk:
        chunks.append(chunk)

    for words in chunks:
        if not words:
            continue

        start = ass_time(words[0]["start"])
        end = ass_time(max(words[-1]["end"], words[0]["start"] + 0.4))
        parts = []

        for index, word in enumerate(words):
            duration_cs = max(1, round((word["end"] - word["start"]) * 100))
            safe = escape_ass(word["text"])
            if index == 0:
                parts.append(f"{{\\k{duration_cs}}}{safe}")
            else:
                parts.append(f" {{\\k{duration_cs}}}{safe}")

        lines.append(
            "Dialogue: 0,"
            f"{start},{end},Podcast,,0,0,0,,"
            + "".join(parts)
        )

    output_file.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# BURN SUBTITLES
# ============================================================

def burn_subtitles(
    video_path,
    subtitle_file,
    output_file,
):

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
            "FFmpeg failed:\n%s",
            result.stderr[-5000:]
        )

        raise RuntimeError(
            "FFmpeg could not burn subtitles."
        )


# ============================================================
# SEND CAPTIONED VIDEO
# ============================================================

async def send_captioned_video(
    message,
    video_path,
    caption=None,
):

    video_caption = (
        "🎬 Captioned video generated."
    )

    if caption:

        short_caption = extract_short_caption(
            caption
        )

        if short_caption:

            video_caption = short_caption[:900]

    await message.reply_video(
        video=str(video_path),
        caption=video_caption,
        supports_streaming=True,
    )

    if caption:

        await send_long_message(
            message,
            caption,
            photo_result_keyboard()
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

            return result

    return ""


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
            reply_markup=keyboard
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

        if index == len(chunks) - 1:

            await message.reply_text(
                chunk,
                reply_markup=keyboard
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
        "Unhandled application error:",
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
        "MEDIA CAPTION GENERATOR"
    )

    logger.info(
        "Gemini requested model: %s",
        REQUESTED_GEMINI_MODEL
        or "(automatic)"
    )

    logger.info(
        "Gemini selected model: %s",
        GEMINI_MODEL
    )

    logger.info(
        "Whisper model: %s",
        WHISPER_MODEL_NAME
    )

    logger.info(
        "Maximum video duration: %s seconds",
        MAX_VIDEO_SECONDS
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
            start
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command
        )
    )

    application.add_handler(
        CommandHandler(
            "cancel",
            cancel
        )
    )

    # --------------------------------------------------------
    # CALLBACKS
    # --------------------------------------------------------

    application.add_handler(
        CallbackQueryHandler(
            callback_handler
        )
    )

    # --------------------------------------------------------
    # PHOTO
    # --------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            media_handler
        )
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.VIDEO,
            media_handler
        )
    )

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.Document.ALL,
            media_handler
        )
    )

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_handler
        )
    )

    # --------------------------------------------------------
    # ERRORS
    # --------------------------------------------------------

    application.add_error_handler(
        error_handler
    )

    logger.info(
        "Telegram bot is running."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
