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

BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    ""
).strip()

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY",
    ""
).strip()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5.6"
).strip()


if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN environment variable is missing."
    )


if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is missing."
    )


# ============================================================
# OPENAI
# ============================================================

client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(
    "media-caption-generator"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_VIDEO_SECONDS = 60

VIDEO_FRAME_COUNT = 5

MAX_IMAGE_BYTES = 8 * 1024 * 1024


# ============================================================
# CAPTION STYLES
# ============================================================

STYLES = {

    "instagram": "Instagram",

    "short": "Short and Catchy",

    "funny": "Funny",

    "professional": "Professional",

    "travel": "Travel",

    "romantic": "Romantic",

    "viral": "Trendy and Viral",

    "aesthetic": "Aesthetic",

    "attitude": "Attitude",

    "bollywood": "Bollywood",

}


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {

    "english": "English",

    "hindi": "Hindi",

    "hinglish": "Hinglish",

    "punjabi": "Punjabi",

}


# ============================================================
# STYLE KEYBOARD
# ============================================================

def style_keyboard():

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
                "😎 Attitude",
                callback_data="style:attitude"
            ),

            InlineKeyboardButton(
                "🎬 Bollywood",
                callback_data="style:bollywood"
            ),
        ],

        [
            InlineKeyboardButton(
                "🎨 Custom",
                callback_data="style:custom"
            ),
        ],

    ]

    return InlineKeyboardMarkup(
        keyboard
    )


# ============================================================
# LANGUAGE KEYBOARD
# ============================================================

def language_keyboard():

    keyboard = [

        [
            InlineKeyboardButton(
                "🇬🇧 English",
                callback_data="language:english"
            ),

            InlineKeyboardButton(
                "🇮🇳 Hindi",
                callback_data="language:hindi"
            ),
        ],

        [
            InlineKeyboardButton(
                "🗣 Hinglish",
                callback_data="language:hinglish"
            ),

            InlineKeyboardButton(
                "🅿️ Punjabi",
                callback_data="language:punjabi"
            ),
        ],

    ]

    return InlineKeyboardMarkup(
        keyboard
    )


# ============================================================
# RESULT KEYBOARD
# ============================================================

def result_keyboard():

    keyboard = [

        [
            InlineKeyboardButton(
                "🔄 Regenerate",
                callback_data="result:regenerate"
            ),

            InlineKeyboardButton(
                "🎨 New Style",
                callback_data="result:style"
            ),
        ],

        [
            InlineKeyboardButton(
                "🌐 New Language",
                callback_data="result:language"
            ),
        ],

        [
            InlineKeyboardButton(
                "📤 New Media",
                callback_data="result:new"
            ),
        ],

    ]

    return InlineKeyboardMarkup(
        keyboard
    )


# ============================================================
# START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    context.user_data.clear()

    text = """
✨ *MEDIA CAPTION GENERATOR*

Welcome!

Send me a photo or video and I will
generate a social-media caption.

🎨 *Styles*

📸 Instagram
⚡ Short
😂 Funny
💼 Professional
✈️ Travel
❤️ Romantic
🔥 Viral
✨ Aesthetic
😎 Attitude
🎬 Bollywood
🎨 Custom

🌐 *Languages*

🇬🇧 English
🇮🇳 Hindi
🗣 Hinglish
🅿️ Punjabi

📸 Send your media to begin.
"""

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

    text = """
📖 *HOW TO USE*

1. Send a photo or video.

2. Select a caption style.

3. Select a language.

4. AI analyzes your media.

5. Caption and hashtags are generated.

6. Use Regenerate for another caption.

Commands:

/start
/help
/cancel
"""

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
        "Send a new photo or video."
    )


# ============================================================
# MEDIA DETECTION
# ============================================================

def detect_media(
    message
):

    if message.photo:

        return (
            "photo",
            message.photo[-1].file_id
        )

    if message.video:

        return (
            "video",
            message.video.file_id
        )

    if message.document:

        mime = (
            message.document.mime_type
            or ""
        )

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

    media_type, file_id = detect_media(
        message
    )

    if not media_type:

        await message.reply_text(
            "Please send a photo or video."
        )

        return

    context.user_data[
        "media_type"
    ] = media_type

    context.user_data[
        "file_id"
    ] = file_id

    context.user_data[
        "original_caption"
    ] = message.caption or ""

    context.user_data[
        "style"
    ] = None

    context.user_data[
        "language"
    ] = None

    context.user_data[
        "custom_style"
    ] = None

    await message.reply_text(
        "✅ Media received!\n\n"
        "Choose a caption style:",
        reply_markup=style_keyboard()
    )


# ============================================================
# CUSTOM STYLE REQUEST
# ============================================================

async def custom_style_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    context.user_data[
        "waiting_custom_style"
    ] = True

    await query.edit_message_text(
        """
🎨 *CUSTOM STYLE*

Type the caption style you want.

Examples:

• Luxury
• Gen Z
• Bollywood
• Emotional
• Savage
• Punjabi swag
• One liner
• Classy
• Motivational
• Romantic
""",
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
        "waiting_custom_style"
    ):

        await update.message.reply_text(
            "📸 Please send a photo or video."
        )

        return

    custom_style = (
        update.message.text or ""
    ).strip()

    if not custom_style:

        return

    context.user_data[
        "custom_style"
    ] = custom_style

    context.user_data[
        "waiting_custom_style"
    ] = False

    await update.message.reply_text(
        "✅ Custom style saved.\n\n"
        "Now choose the language:",
        reply_markup=language_keyboard()
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

    style = query.data.split(
        ":",
        1
    )[1]

    if style == "custom":

        await custom_style_request(
            update,
            context
        )

        return

    context.user_data[
        "style"
    ] = style

    await query.edit_message_text(
        "🎨 Style selected:\n\n"
        f"*{STYLES[style]}*\n\n"
        "Choose your language:",
        parse_mode="Markdown",
        reply_markup=language_keyboard()
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

    await generate_caption_response(
        update,
        context
    )


# ============================================================
# RESULT CALLBACK
# ============================================================

async def result_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    action = query.data.split(
        ":",
        1
    )[1]

    if action == "regenerate":

        await generate_caption_response(
            update,
            context,
            regenerate=True
        )

        return

    if action == "style":

        await query.edit_message_text(
            "🎨 Choose a new style:",
            reply_markup=style_keyboard()
        )

        return

    if action == "language":

        await query.edit_message_text(
            "🌐 Choose a language:",
            reply_markup=language_keyboard()
        )

        return

    if action == "new":

        context.user_data.clear()

        await query.edit_message_text(
            "📸 Send your new photo or video."
        )


# ============================================================
# DOWNLOAD MEDIA
# ============================================================

async def download_media(
    context
):

    media_type = context.user_data[
        "media_type"
    ]

    file_id = context.user_data[
        "file_id"
    ]

    telegram_file = await (
        context.bot.get_file(
            file_id
        )
    )

    if media_type == "photo":

        suffix = ".jpg"

    else:

        suffix = ".mp4"

    temporary = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temporary.close()

    await telegram_file.download_to_drive(
        temporary.name
    )

    return Path(
        temporary.name
    )


# ============================================================
# IMAGE TO DATA URL
# ============================================================

def image_to_data_url(
    path
):

    if path.stat().st_size > MAX_IMAGE_BYTES:

        raise ValueError(
            "Image is too large."
        )

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")

    extension = path.suffix.lower()

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
    video_path
):

    output_directory = Path(
        tempfile.mkdtemp(
            prefix="caption_frames_"
        )
    )

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

        duration = MAX_VIDEO_SECONDS

    duration = min(
        duration,
        MAX_VIDEO_SECONDS
    )

    frames = []

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

        frame = (
            output_directory
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
                str(frame)
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30
        )

        if (
            result.returncode == 0
            and frame.exists()
        ):

            frames.append(
                frame
            )

    if not frames:

        raise RuntimeError(
            "Could not extract video frames."
        )

    return frames


# ============================================================
# PROMPT
# ============================================================

def create_prompt(
    context,
    regenerate=False
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

    if custom_style:

        style_name = custom_style

    else:

        style_name = STYLES.get(
            style,
            "Instagram"
        )

    language_name = LANGUAGES.get(
        language,
        "English"
    )

    if regenerate:

        regeneration = (
            "Create a substantially different "
            "caption from the previous version."
        )

    else:

        regeneration = (
            "Create the best possible caption."
        )

    return f"""
You are an expert social media copywriter.

Analyze the supplied image or video frames.

Caption style:
{style_name}

Language:
{language_name}

Rules:

- Only describe what is visible or reasonably
  inferable.
- Never invent names.
- Never invent locations.
- Never invent events.
- Never invent brands.
- Never invent dates.
- Never invent relationships.
- Do not mention AI.
- Do not mention filenames.
- Do not include URLs.
- Do not create fake credits.
- Do not say "Video by..." unless supplied
  by the user.
- Use natural language.
- Use tasteful emojis.
- Generate 8-15 relevant hashtags.
- Avoid repetitive hashtags.
- Match the selected language.
- Make the caption suitable for social media.

Return:

CAPTION:
<caption>

HASHTAGS:
#hashtag1 #hashtag2 #hashtag3

OPTIONAL_SHORT:
<short alternative>

{regeneration}

Original media caption:

{original_caption or "(none)"}
""".strip()


# ============================================================
# OPENAI IMAGE REQUEST
# ============================================================

def analyze_images(
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
# GENERATE CAPTION
# ============================================================

async def generate_caption(
    context,
    media_path
):

    prompt = create_prompt(
        context,
        context.user_data.get(
            "regenerate",
            False
        )
    )

    if context.user_data[
        "media_type"
    ] == "photo":

        image_urls = [

            image_to_data_url(
                media_path
            )

        ]

    else:

        frames = extract_video_frames(
            media_path
        )

        image_urls = [

            image_to_data_url(
                frame
            )

            for frame in frames
        ]

    return analyze_images(
        image_urls,
        prompt
    )


# ============================================================
# GENERATE RESPONSE
# ============================================================

async def generate_caption_response(
    update,
    context,
    regenerate=False
):

    context.user_data[
        "regenerate"
    ] = regenerate

    query = update.callback_query

    message = query.message

    await message.edit_text(
        "⏳ *Analyzing your media...*\n\n"
        "✨ Creating your caption...",
        parse_mode="Markdown"
    )

    await context.bot.send_chat_action(
        chat_id=message.chat_id,
        action=ChatAction.TYPING
    )

    media_path = None

    try:

        media_path = await download_media(
            context
        )

        result = await generate_caption(
            context,
            media_path
        )

        if len(result) > 3800:

            result = (
                result[:3800]
                + "..."
            )

        await message.reply_text(
            "✨ *GENERATED CAPTION*\n\n"
            + result,
            parse_mode="Markdown",
            reply_markup=result_keyboard()
        )

    except Exception as error:

        logger.exception(
            "Caption generation failed"
        )

        await message.reply_text(
            "❌ Caption generation failed.\n\n"
            f"Error:\n{str(error)[:1000]}"
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
        "Unhandled error",
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

    # Photos and videos

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
            pattern=r"^language:"
        )
    )

    # Result buttons

    application.add_handler(
        CallbackQueryHandler(
            result_callback,
            pattern=r"^result:"
        )
    )

    # Text

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
        "Media Caption Generator is running."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    main()
