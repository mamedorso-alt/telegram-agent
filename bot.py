import asyncio
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import whisper
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


# Best practice: keep secrets in env vars. If BOT_TOKEN is not set,
# we fall back to the token provided by the user (not recommended).
DEFAULT_TOKEN = "8758355117:AAHGojzM_JzWs8jH7SRuDiYBC9rooneFbBo"


@dataclass(frozen=True)
class TaskType:
    code: str


TASK_CALENDAR = TaskType("КАЛЕНДАРЬ")
TASK_MESSAGE = TaskType("СООБЩЕНИЕ")
TASK_DOC = TaskType("ДОКУМЕНТ")
TASK_SEARCH = TaskType("ПОИСК")
TASK_NOTE = TaskType("ЗАМЕТКА")


RULES: list[tuple[TaskType, list[str]]] = [
    (TASK_CALENDAR, ["встреча", "запиши", "напомни", "календарь"]),
    (TASK_MESSAGE, ["сообщение для", "напиши", "передай"]),
    (TASK_DOC, ["регламент", "документ", "инструкция"]),
    (TASK_SEARCH, ["погода", "найди", "какой курс"]),
]


_WHITESPACE_RE = re.compile(r"\s+")
_MODEL: Optional[whisper.Whisper] = None


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    return _WHITESPACE_RE.sub(" ", text)


def detect_task_type(text: str) -> TaskType:
    t = normalize_text(text)
    for task_type, keywords in RULES:
        for kw in keywords:
            if kw in t:
                return task_type
    return TASK_NOTE


def ensure_model_loaded() -> whisper.Whisper:
    global _MODEL
    if _MODEL is None:
        _MODEL = whisper.load_model("base")
    return _MODEL


def convert_to_wav(input_path: Path, output_path: Path) -> None:
    # 16kHz mono PCM WAV is a safe default for STT.
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr[-1200:]}")


async def transcribe_wav(wav_path: Path) -> str:
    model = ensure_model_loaded()

    def _run() -> str:
        result = model.transcribe(str(wav_path), language="ru")
        return (result.get("text") or "").strip()

    return await asyncio.to_thread(_run)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Отправь голосовое или текст — я распознаю/прочитаю и определю тип задачи."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        return

    text = message.text.strip()
    task_type = detect_task_type(text)
    await message.reply_text(f"[{task_type.code}] {text}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.voice:
        return

    await message.chat.send_action(ChatAction.TYPING)

    tmpdir = Path(tempfile.mkdtemp(prefix="tg_voice_"))
    try:
        voice = message.voice
        tg_file = await context.bot.get_file(voice.file_id)

        src_path = tmpdir / "voice.ogg"
        wav_path = tmpdir / "voice.wav"

        await tg_file.download_to_drive(custom_path=str(src_path))

        await asyncio.to_thread(convert_to_wav, src_path, wav_path)
        await message.chat.send_action(ChatAction.TYPING)

        text = await transcribe_wav(wav_path)
        if not text:
            await message.reply_text("Не удалось распознать речь (пустой результат).")
            return

        task_type = detect_task_type(text)
        await message.reply_text(f"[{task_type.code}] {text}")
    except Exception as e:
        await message.reply_text(f"Ошибка обработки голосового: {e}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    token = os.environ.get("BOT_TOKEN") or DEFAULT_TOKEN
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

