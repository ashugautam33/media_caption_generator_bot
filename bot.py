import os
import base64
import logging
import tempfile
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Change this in .env if your API account uses another model.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6").strip()


if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN is missing. Add it to your .env file."
    )

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is missing. Add it to your .env file."
    )


client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("media-caption-generator")


# ============================================================
# SETTINGS
# ============================================================

MAX_VIDEO_SECONDS = 60

VIDEO_FRAME_COUNT = 5

MAX_IMAGE_BYTES = 8 * 1024 * 1024


# ============================================================
# STYLE LABELS
# ============================================================

STYLE_LABELS = {

    "instagram":
        "Instagram",

    "short":
        "Short & Catchy",

    "funny":
        "Funny",

    "professional":
        "Professional",

    "travel":
        "Travel",

    "romantic":
        "Romantic",

    "viral":
        "Trendy / Viral",

    "aesthetic":
        "Aesthetic",

    "custom":
        "Custom",
}


# ============================================================
# LANGUAGE LABELS
# ============================================================

LANGUAGE_LABELS = {

    "english":
        "English",

    "hindi":
        "Hindi",

    "hinglish":
        "Hinglish",

    "punjabi":
        "Punjabi",
}


# ============================================================
# MAIN STYLE KEYBOARD
# ============================================================

def main_menu():

    keyboard = [

        [
            InlineKeyboardButton(
                "📸 Instagram",
                callback_data="style:instagram"
            ),

            InlineKeyboardButton(
                "⚡ Short",
                callback_data="style:short"
            ),
        ],

        [
            InlineKeyboardButton(
                "😂 Funny",
                callback_data="style:funny"
            ),

            InlineKeyboardButton(
                "💼 Professional",
                callback_data="style:professional"
            ),
        ],

        [
            InlineKeyboardButton(
                "✈️ Travel",
                callback_data="style:travel"
            ),

            InlineKeyboardButton(
                "❤️ Romantic",
                callback_data="style:romantic"
            ),
        ],

        [
            InlineKeyboardButton(
                "🔥 Viral",
                callback_data="style:viral"
            ),

            InlineKeyboardButton(
                "✨ Aesthetic",
                callback_data="style:aesthetic"
            ),
        ],

        [
            InlineKeyboardButton(
                "🎨 Custom Style",
                callback_data="style:custom"
            )
        ],
    ]

    return InlineKeyboardMarkup(keyboard)


# ============================================================
# LANGUAGE KEYBOARD
# ============================================================

def language_menu():

    keyboard = [

        [
            InlineKeyboardButton(
                "🇬🇧 English",
                callback_data="lang:english"
            ),

            InlineKeyboardButton(
                "🇮🇳 Hindi",
                callback_data="lang:hindi"
            ),
        ],

        [
            InlineKeyboardButton(
                "🗣 Hinglish",
                callback_data="lang:hinglish"
            ),

            InlineKeyboardButton(
                "🅿️ Punjabi",
                callback_data="lang:punjabi"
            ),
        ],
    ]

    return InlineKeyboardMarkup(keyboard)


# ============================================================
# RESULT KEYBOARD
# ============================================================

def result_menu():

    keyboard = [

        [
            InlineKeyboardButton(
                "🔄 Regenerate",
                callback_data="action:regenerate"
            ),

            InlineKeyboardButton(
                "🎨 New Style",
                callback_data="action:new_style"
            ),
        ],

        [
            InlineKeyboardButton(
                "🌐 New Language",
                callback_data="action:new_language"
            )
        ],

        [
            InlineKeyboardButton(
                "📤 New Media",
                callback_data="action:new_media"
            )
        ],
    ]

    return InlineKeyboardMarkup(keyboard)


# ============================================================
# START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    context.user_data.clear()

    text = (
        "✨ *Media Caption Generator*\n\n"

        "Send me a *photo or video* and I'll create "
        "a caption for it.\n\n"

        "You can choose:\n"

        "• 📸 Instagram\n"
        "• ⚡ Short\n"
        "• 😂 Funny\n"
        "• 💼 Professional\n"
        "• ✈️ Travel\n"
        "• ❤️ Romantic\n"
        "• 🔥 Viral\n"
        "• ✨ Aesthetic\n"
        "• 🎨 Custom\n\n"

        "Languages:\n"

        "🇬🇧 English\n"
        "🇮🇳 Hindi\n"
        "🗣 Hinglish\n"
        "🅿️ Punjabi\n\n"

        "📸 *Send your media to begin.*"
    )

    await update.message.reply_text(
        text,
        parse_mode="Markdown"
    )


# ============================================================
# HELP
# ============================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    text = (
        "📖 *How to use the bot*\n\n"

        "1️⃣ Send a photo or video.\n"
        "2️⃣ Select a caption style.\n"
        "3️⃣ Select a language.\n"
        "4️⃣ The AI generates your caption.\n"
        "5️⃣ Use Regenerate for another version.\n\n"

        "*Commands*\n\n"

        "/start - Start the bot\n"
        "/help - Show help\n"
        "/cancel - Cancel current operation"
    )

    await update.message.reply_text(
        text,
        parse_mode="Markdown"
    )


# ============================================================
# CANCEL
# ============================================================

async def cancel(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    context.user_data.clear()

    await update.message.reply_text(
        "❌ Cancelled.\n\n"
        "Send a new photo or video whenever you're ready. 📸🎥"
    )


# ============================================================
# MEDIA DETECTION
# ============================================================

def detect_media(message):

    # Telegram photo
    if message.photo:

        return (
            "photo",
            message.photo[-1].file_id
        )

    # Telegram video
    if message.video:

        return (
            "video",
            message.video.file_id
        )

    # Document
    if message.document:

        mime = message.document.mime_type or ""

        if mime.startswith("image/"):

            return (
                "photo",
                message.document.file_id
            )

        if mime.startswith("video/"):

            return (
                "video",
                message.document.file_id
            )

    return None, None


# ============================================================
# MEDIA HANDLER
# ============================================================

async def media_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    message = update.effective_message

    kind, file_id = detect_media(message)

    if not kind:

        await message.reply_text(
            "❌ Please send a photo or video."
        )

        return

    context.user_data["media_kind"] = kind

    context.user_data["file_id"] = file_id

    context.user_data["original_caption"] = (
        message.caption or ""
    )

    context.user_data["custom_style"] = None

    context.user_data["style"] = None

    context.user_data["language"] = None

    await message.reply_text(
        "✅ Media received!\n\n"
        "Choose your caption style:",
        reply_markup=main_menu()
    )


# ============================================================
# CUSTOM STYLE
# ============================================================

async def custom_style_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    context.user_data["awaiting_custom_style"] = True

    await query.edit_message_text(
        "🎨 *Custom Caption Style*\n\n"

        "Send me the style you want.\n\n"

        "Examples:\n"

        "• Luxury and classy\n"
        "• Gen Z slang\n"
        "• Bollywood style\n"
        "• Deep and emotional\n"
        "• Minimal one-liner\n"
        "• Attitude caption\n"
        "• Punjabi swag\n"
        "• Corporate professional\n",
        parse_mode="Markdown"
    )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not context.user_data.get(
        "awaiting_custom_style"
    ):

        await update.message.reply_text(
            "📸 Please send a photo or video first."
        )

        return

    style = (
        update.message.text or ""
    ).strip()

    if not style:

        return

    context.user_data[
        "custom_style"
    ] = style

    context.user_data[
        "awaiting_custom_style"
    ] = False

    await update.message.reply_text(
        "✅ Custom style saved:\n\n"
        f"*{style}*\n\n"
        "Now choose your language:",
        parse_mode="Markdown",
        reply_markup=language_menu()
    )


# ============================================================
# STYLE CALLBACK
# ============================================================

async def style_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    if not context.user_data.get("file_id"):

        await query.edit_message_text(
            "❌ Please send a photo or video first."
        )

        return

    style = query.data.split(
        ":",
        1
    )[1]

    if style == "custom":

        await custom_style_handler(
            update,
            context
        )

        return

    context.user_data["style"] = style

    await query.edit_message_text(
        f"🎨 Style selected: "
        f"*{STYLE_LABELS[style]}*\n\n"
        "Now choose the language:",
        parse_mode="Markdown",
        reply_markup=language_menu()
    )


# ============================================================
# LANGUAGE CALLBACK
# ============================================================

async def language_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    language = query.data.split(
        ":",
        1
    )[1]

    context.user_data[
        "language"
    ] = language

    await generate_and_send(
        update,
        context
    )


# ============================================================
# ACTION CALLBACK
# ============================================================

async def action_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    action = query.data.split(
        ":",
        1
    )[1]

    # --------------------------------------------------------
    # REGENERATE
    # --------------------------------------------------------

    if action == "regenerate":

        if not context.user_data.get(
            "language"
        ):

            await query.message.reply_text(
                "Please choose a language first."
            )

            return

        await generate_and_send(
            update,
            context,
            regenerate=True
        )

        return

    # --------------------------------------------------------
    # NEW STYLE
    # --------------------------------------------------------

    if action == "new_style":

        await query.edit_message_text(
            "🎨 Choose a new caption style:",
            reply_markup=main_menu()
        )

        return

    # --------------------------------------------------------
    # NEW LANGUAGE
    # --------------------------------------------------------

    if action == "new_language":

        await query.edit_message_text(
            "🌐 Choose a new language:",
            reply_markup=language_menu()
        )

        return

    # --------------------------------------------------------
    # NEW MEDIA
    # --------------------------------------------------------

    if action == "new_media":

        context.user_data.clear()

        await query.edit_message_text(
            "📸 Send your new photo or video."
        )

        return


# ============================================================
# DOWNLOAD TELEGRAM MEDIA
# ============================================================

async def download_media(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    kind = context.user_data[
        "media_kind"
    ]

    file_id = context.user_data[
        "file_id"
    ]

    telegram_file = (
        await context.bot.get_file(
            file_id
        )
    )

    if kind == "photo":

        suffix = ".jpg"

    else:

        suffix = ".mp4"

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp.close()

    await telegram_file.download_to_drive(
        temp.name
    )

    return Path(temp.name)


# ============================================================
# IMAGE -> DATA URL
# ============================================================

def image_to_data_url(
    path: Path
):

    size = path.stat().st_size

    if size > MAX_IMAGE_BYTES:

        raise ValueError(
            "Image is too large. "
            "Please send a smaller image."
        )

    raw = path.read_bytes()

    encoded = base64.b64encode(
        raw
    ).decode("utf-8")

    extension = (
        path.suffix.lower()
    )

    if extension == ".png":

        mime = "image/png"

    elif extension == ".webp":

        mime = "image/webp"

    else:

        mime = "image/jpeg"

    return (
        f"data:{mime};base64,{encoded}"
    )


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_video_frames(
    video_path: Path
):

    output_dir = Path(
        tempfile.mkdtemp(
            prefix="caption_frames_"
        )
    )

    # Get duration
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ],
        capture_output=True,
        text=True,
        timeout=20
    )

    try:

        duration = float(
            probe.stdout.strip()
        )

    except Exception:

        duration = 0

    if duration <= 0:

        duration = MAX_VIDEO_SECONDS

    duration = min(
        duration,
        MAX_VIDEO_SECONDS
    )

    frame_paths = []

    for index in range(
        VIDEO_FRAME_COUNT
    ):

        if VIDEO_FRAME_COUNT == 1:

            timestamp = 0

        else:

            timestamp = (
                duration
                * index
                / (VIDEO_FRAME_COUNT - 1)
            )

        frame_path = (
            output_dir
            / f"frame_{index}.jpg"
        )

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",

                "-ss",
                str(timestamp),

                "-i",
                str(video_path),

                "-frames:v",
                "1",

                "-vf",
                "scale='min(1024,iw)':-2",

                str(frame_path)
            ],
            capture_output=True,
            timeout=30
        )

        if (
            result.returncode == 0
            and frame_path.exists()
        ):

            frame_paths.append(
                frame_path
            )

    if not frame_paths:

        raise RuntimeError(
            "Could not extract frames from video."
        )

    return frame_paths


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    style,
    language,
    original_caption="",
    custom_style=None,
    regenerate=False
):

    if custom_style:

        style_text = custom_style

    else:

        style_text = STYLE_LABELS.get(
            style,
            "Instagram"
        )

    language_text = LANGUAGE_LABELS.get(
        language,
        "English"
    )

    if regenerate:

        regeneration_text = (
            "This is a regeneration. "
            "Create a distinctly different "
            "caption from the previous version."
        )

    else:

        regeneration_text = (
            "Create the strongest first version."
        )

    prompt = f"""
You are an expert social-media copywriter.

Carefully analyze the supplied photo or
video frames.

Only describe things that are reasonably
visible or inferable.

Never invent:

- People's names
- Locations
- Brands
- Events
- Relationships
- Dates
- Facts

unless they are clearly supported by
the media or the user's supplied caption.

CAPTION STYLE:
{style_text}

LANGUAGE:
{language_text}

Requirements:

1. Make the caption natural and human.
2. Make it suitable for Instagram and Telegram.
3. Do not mention AI.
4. Do not mention filenames.
5. Do not include URLs.
6. Do not include technical metadata.
7. Do not write "Video by..." or similar credits
   unless explicitly supplied by the user.
8. Use tasteful emojis where appropriate.
9. Generate 8-15 relevant hashtags.
10. Avoid repetitive generic hashtags.
11. Put hashtags after the caption.
12. Keep the caption concise unless the style
    requires a longer caption.
13. If readable text appears in the media,
    use it only when clearly visible.
14. Do not make unsupported claims.

Return exactly:

CAPTION:
<caption>

HASHTAGS:
#tag1 #tag2 #tag3

OPTIONAL_SHORT:
<one short alternative>

{regeneration_text}

Original user media caption:
{original_caption or "(none)"}
"""

    return prompt.strip()


# ============================================================
# OPENAI REQUEST
# ============================================================

def call_openai_for_images(
    image_urls,
    prompt
):

    content = [
        {
            "type": "input_text",
            "text": prompt
        }
    ]

    for image_url in image_urls:

        content.append(
            {
                "type": "input_image",
                "image_url": image_url,
                "detail": "high"
            }
        )

    response = client.responses.create(

        model=OPENAI_MODEL,

        input=[
            {
                "role": "user",

                "content": content
            }
        ]
    )

    return response.output_text.strip()


# ============================================================
# CAPTION GENERATION
# ============================================================

async def generate_caption(
    context,
    media_path
):

    style = context.user_data.get(
        "style",
        "instagram"
    )

    language = context.user_data.get(
        "language",
        "english"
    )

    custom_style = context.user_data.get(
        "custom_style"
    )

    original_caption = context.user_data.get(
        "original_caption",
        ""
    )

    regenerate = context.user_data.get(
        "regenerate",
        False
    )

    prompt = build_prompt(

        style=style,

        language=language,

        original_caption=original_caption,

        custom_style=custom_style,

        regenerate=regenerate
    )

    # --------------------------------------------------------
    # PHOTO
    # --------------------------------------------------------

    if context.user_data[
        "media_kind"
    ] == "photo":

        image_urls = [
            image_to_data_url(
                media_path
            )
        ]

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    else:

        frames = extract_video_frames(
            media_path
        )

        image_urls = []

        for frame in frames:

            image_urls.append(
                image_to_data_url(
                    frame
                )
            )

    return call_openai_for_images(
        image_urls,
        prompt
    )


# ============================================================
# CLEAN RESPONSE
# ============================================================

def clean_result(
    text
):

    text = text.strip()

    if len(text) > 3800:

        text = (
            text[:3800].rstrip()
            + "…"
        )

    return text


# ============================================================
# GENERATE AND SEND
# ============================================================

async def generate_and_send(
    update,
    context,
    regenerate=False
):

    context.user_data[
        "regenerate"
    ] = regenerate

    if update.callback_query:

        message = (
            update.callback_query.message
        )

        await message.edit_text(
            "⏳ Analyzing your media...\n\n"
            "✨ Creating your caption..."
        )

    else:

        message = update.effective_message

    await context.bot.send_chat_action(
        chat_id=message.chat_id,
        action=ChatAction.TYPING
    )

    media_path = None

    try:

        media_path = await download_media(
            update,
            context
        )

        result = await generate_caption(
            context,
            media_path
        )

        result = clean_result(
            result
        )

        await message.reply_text(

            "✨ *Generated Caption*\n\n"
            + result,

            parse_mode="Markdown",

            reply_markup=result_menu()
        )

    except Exception as error:

        logger.exception(
            "Caption generation failed"
        )

        error_text = str(error)

        error_text = (
            error_text
            .replace("*", "")
            .replace("_", "")
        )

        await message.reply_text(

            "❌ *Caption generation failed.*\n\n"

            f"Reason:\n{error_text[:1000]}\n\n"

            "Please try another photo/video "
            "or use /start.",

            parse_mode="Markdown"
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
# ERROR HANDLER
# ============================================================

async def error_handler(
    update,
    context
):

    logger.exception(
        "Unhandled bot error",
        exc_info=context.error
    )


# ============================================================
# MAIN
# ============================================================

def main():

    application = (
        Application
        .builder()
        .token(BOT_TOKEN)
        .concurrent_updates(False)
        .build()
    )

    # Commands
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

    # Photos / videos
    application.add_handler(
        MessageHandler(
            (
                filters.PHOTO
                | filters.VIDEO
                | filters.Document.IMAGE
                | filters.Document.VIDEO
            ),
            media_handler
        )
    )

    # Style buttons
    application.add_handler(
        CallbackQueryHandler(
            style_callback,
            pattern=r"^style:"
        )
    )

    # Language buttons
    application.add_handler(
        CallbackQueryHandler(
            language_callback,
            pattern=r"^lang:"
        )
    )

    # Result/action buttons
    application.add_handler(
        CallbackQueryHandler(
            action_callback,
            pattern=r"^action:"
        )
    )

    # Custom style text
    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_handler
        )
    )

    application.add_error_handler(
        error_handler
    )

    logger.info(
        "Media Caption Generator started."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
