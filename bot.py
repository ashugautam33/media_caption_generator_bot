import os
import re
import asyncio
import logging
import tempfile
import subprocess
import mimetypes

from pathlib import Path

from dotenv import load_dotenv

from google import genai
from google.genai import types

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
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

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
).strip()

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024

MAX_VIDEO_SECONDS = 60

VIDEO_FRAME_COUNT = 5


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)


# ============================================================
# GEMINI CLIENT
# ============================================================

if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN environment variable is missing."
    )

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is missing."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# CAPTION STYLES
# ============================================================

CAPTION_STYLES = {
    "instagram": "Instagram",
    "short": "Short",
    "funny": "Funny",
    "professional": "Professional",
    "travel": "Travel",
    "romantic": "Romantic",
    "viral": "Viral",
    "aesthetic": "Aesthetic",
    "attitude": "Attitude",
    "bollywood": "Bollywood",
    "custom": "Custom"
}


LANGUAGES = {
    "english": "English",
    "hindi": "Hindi",
    "hinglish": "Hinglish",
    "punjabi": "Punjabi"
}


# ============================================================
# KEYBOARDS
# ============================================================

def style_keyboard():

    buttons = [
        [
            InlineKeyboardButton(
                "📸 Instagram",
                callback_data="style:instagram"
            ),
            InlineKeyboardButton(
                "✨ Short",
                callback_data="style:short"
            )
        ],
        [
            InlineKeyboardButton(
                "😂 Funny",
                callback_data="style:funny"
            ),
            InlineKeyboardButton(
                "💼 Professional",
                callback_data="style:professional"
            )
        ],
        [
            InlineKeyboardButton(
                "✈️ Travel",
                callback_data="style:travel"
            ),
            InlineKeyboardButton(
                "❤️ Romantic",
                callback_data="style:romantic"
            )
        ],
        [
            InlineKeyboardButton(
                "🔥 Viral",
                callback_data="style:viral"
            ),
            InlineKeyboardButton(
                "🌸 Aesthetic",
                callback_data="style:aesthetic"
            )
        ],
        [
            InlineKeyboardButton(
                "😎 Attitude",
                callback_data="style:attitude"
            ),
            InlineKeyboardButton(
                "🎬 Bollywood",
                callback_data="style:bollywood"
            )
        ],
        [
            InlineKeyboardButton(
                "✍️ Custom Style",
                callback_data="style:custom"
            )
        ]
    ]

    return InlineKeyboardMarkup(buttons)


def language_keyboard():

    buttons = [
        [
            InlineKeyboardButton(
                "🇬🇧 English",
                callback_data="lang:english"
            ),
            InlineKeyboardButton(
                "🇮🇳 Hindi",
                callback_data="lang:hindi"
            )
        ],
        [
            InlineKeyboardButton(
                "🗣 Hinglish",
                callback_data="lang:hinglish"
            ),
            InlineKeyboardButton(
                "🪯 Punjabi",
                callback_data="lang:punjabi"
            )
        ]
    ]

    return InlineKeyboardMarkup(buttons)


def result_keyboard():

    buttons = [
        [
            InlineKeyboardButton(
                "🔄 Regenerate",
                callback_data="action:regenerate"
            )
        ],
        [
            InlineKeyboardButton(
                "🎨 New Style",
                callback_data="action:style"
            ),
            InlineKeyboardButton(
                "🌐 New Language",
                callback_data="action:language"
            )
        ],
        [
            InlineKeyboardButton(
                "📤 New Media",
                callback_data="action:new"
            )
        ]
    ]

    return InlineKeyboardMarkup(buttons)


# ============================================================
# COMMANDS
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    context.user_data.clear()

    message = (
        "🎨 MEDIA CAPTION GENERATOR\n\n"
        "Welcome!\n\n"
        "Send me a photo or video and I will "
        "generate creative captions and hashtags.\n\n"
        "✨ Features:\n"
        "📸 Photo captions\n"
        "🎥 Video captions\n"
        "🔥 Instagram captions\n"
        "😂 Funny captions\n"
        "❤️ Romantic captions\n"
        "✈️ Travel captions\n"
        "🌐 Multiple languages\n"
        "#️⃣ Relevant hashtags\n\n"
        "Powered by Google Gemini.\n\n"
        "📤 Send a photo or video to begin."
    )

    await update.message.reply_text(message)


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    message = (
        "📖 HOW TO USE\n\n"
        "1. Send a photo or video.\n"
        "2. Choose a caption style.\n"
        "3. Select your language.\n"
        "4. Receive captions and hashtags.\n\n"
        "You can regenerate captions or change "
        "the style and language.\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show instructions\n"
        "/cancel - Cancel current request"
    )

    await update.message.reply_text(message)


async def cancel(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    context.user_data.clear()

    await update.message.reply_text(
        "✅ Current request cancelled.\n\n"
        "Send another photo or video."
    )


# ============================================================
# MEDIA HANDLER
# ============================================================

async def media_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    message = update.effective_message

    media_type = None
    file_id = None

    if message.photo:

        media_type = "photo"
        file_id = message.photo[-1].file_id

    elif message.video:

        media_type = "video"
        file_id = message.video.file_id

    elif message.document:

        document = message.document

        mime = document.mime_type or ""

        if mime.startswith("image/"):

            media_type = "photo"

        elif mime.startswith("video/"):

            media_type = "video"

        else:

            await message.reply_text(
                "❌ Unsupported file type.\n\n"
                "Please send an image or video."
            )

            return

        file_id = document.file_id

    else:

        return

    context.user_data.clear()

    context.user_data["file_id"] = file_id

    context.user_data["media_type"] = media_type

    context.user_data["original_caption"] = (
        message.caption or ""
    )

    context.user_data["stage"] = "style"

    await message.reply_text(
        "✅ Media received!\n\n"
        "🎨 Choose your caption style:",
        reply_markup=style_keyboard()
    )


# ============================================================
# CALLBACK HANDLER
# ============================================================

async def callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    data = query.data or ""

    if data.startswith("style:"):

        style = data.split(":", 1)[1]

        if style not in CAPTION_STYLES:
            return

        if not context.user_data.get("file_id"):

            await query.message.reply_text(
                "Please send a photo or video first."
            )

            return

        context.user_data["style"] = style

        if style == "custom":

            context.user_data["stage"] = "custom"

            await query.message.reply_text(
                "✍️ Describe the caption style you want.\n\n"
                "Example:\n"
                "Create an emotional Instagram caption "
                "with a poetic tone."
            )

            return

        context.user_data["stage"] = "language"

        await query.message.reply_text(
            "🌐 Select caption language:",
            reply_markup=language_keyboard()
        )

        return

    if data.startswith("lang:"):

        language = data.split(":", 1)[1]

        if language not in LANGUAGES:
            return

        context.user_data["language"] = language

        context.user_data["stage"] = "generating"

        await generate_caption(
            query.message,
            context
        )

        return

    if data == "action:regenerate":

        await generate_caption(
            query.message,
            context
        )

        return

    if data == "action:style":

        context.user_data["stage"] = "style"

        await query.message.reply_text(
            "🎨 Choose another style:",
            reply_markup=style_keyboard()
        )

        return

    if data == "action:language":

        context.user_data["stage"] = "language"

        await query.message.reply_text(
            "🌐 Choose another language:",
            reply_markup=language_keyboard()
        )

        return

    if data == "action:new":

        context.user_data.clear()

        await query.message.reply_text(
            "📤 Send a new photo or video."
        )

        return


# ============================================================
# CUSTOM STYLE HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    stage = context.user_data.get("stage")

    if stage == "custom":

        custom_style = update.message.text.strip()

        if not custom_style:

            await update.message.reply_text(
                "Please enter a custom style."
            )

            return

        context.user_data["custom_style"] = custom_style

        context.user_data["stage"] = "language"

        await update.message.reply_text(
            "🌐 Select caption language:",
            reply_markup=language_keyboard()
        )

        return

    await update.message.reply_text(
        "📤 Please send a photo or video.\n\n"
        "Use /help for instructions."
    )


# ============================================================
# PROMPT GENERATOR
# ============================================================

def build_prompt(user_data):

    style_key = user_data.get(
        "style",
        "instagram"
    )

    language_key = user_data.get(
        "language",
        "english"
    )

    style = CAPTION_STYLES.get(
        style_key,
        "Instagram"
    )

    language = LANGUAGES.get(
        language_key,
        "English"
    )

    custom_style = user_data.get(
        "custom_style",
        ""
    )

    original_caption = user_data.get(
        "original_caption",
        ""
    )

    if style_key == "custom" and custom_style:

        style_instruction = custom_style

    else:

        style_instruction = (
            f"Create a {style} style caption."
        )

    prompt = f"""
You are a professional social media caption writer.

Analyze the attached visual media carefully.

Your job is to create engaging captions for social media.

CAPTION STYLE:
{style_instruction}

LANGUAGE:
{language}

USER'S ORIGINAL DESCRIPTION:
{original_caption or "No description provided."}

IMPORTANT RULES:

1. Analyze the visible content accurately.

2. Do not invent specific people, locations,
   events, or identities.

3. Create one main caption.

4. Make the caption natural and engaging.

5. Include suitable emojis.

6. Generate 12 relevant hashtags.

7. Generate one short alternative caption.

8. Do not mention that you are an AI.

9. Avoid generic unrelated hashtags.

10. Match the requested language.

11. For Hindi, use Devanagari script.

12. For Hinglish, use natural Roman-script
    Hindi mixed with English.

13. For Punjabi, use Gurmukhi script.

14. If the media is a video, consider the
    progression of events visible in its frames.

OUTPUT FORMAT:

📸 CAPTION

[Main caption]


#️⃣ HASHTAGS

[12 relevant hashtags]


✨ SHORT CAPTION

[Short alternative]

Return only the requested content.
"""

    return prompt


# ============================================================
# VIDEO PROCESSING
# ============================================================

def video_duration(video_path):

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=20
    )

    if result.returncode != 0:
        return 0

    try:
        return float(result.stdout.strip())

    except ValueError:
        return 0


def extract_video_frames(video_path, output_dir):

    duration = video_duration(video_path)

    if duration <= 0:
        duration = 10

    duration = min(
        duration,
        MAX_VIDEO_SECONDS
    )

    frame_paths = []

    for index in range(VIDEO_FRAME_COUNT):

        timestamp = (
            duration * (index + 0.5)
            / VIDEO_FRAME_COUNT
        )

        output_path = (
            Path(output_dir)
            / f"frame_{index}.jpg"
        )

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-
