"""
LINE Webhook Adapter for Biomedical RAG Bot

This Flask application provides a LINE Bot interface to the Biomedical RAG system.
Features:
- Text message support
- Audio message support with Speech-to-Text (Whisper API)
- Automatic message chunking for LINE's character limits
- Audio duration validation
"""

import os
import subprocess
import re
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any, Optional

load_dotenv()

from flask import Flask, request

from biomedical_rag import BiomedicalRAGBot, chunk_text

# ==================== Configuration ====================

LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Maximum allowed audio length (seconds)
def _get_audio_max_seconds():
    """Parse AUDIO_MAX_SECONDS from environment variable."""
    val = os.environ.get("AUDIO_MAX_SECONDS", "15")
    try:
        return float(val)
    except Exception:
        # Tolerate values like '15s' or malformed input
        try:
            return float(str(val).strip().lower().rstrip('s'))
        except Exception:
            print(f"[Config] Invalid AUDIO_MAX_SECONDS='{val}', using default 15s")
            return 15.0

AUDIO_MAX_SECONDS = _get_audio_max_seconds()

# ==================== Flask App Setup ====================

app = Flask(__name__)

if not LINE_CHANNEL_SECRET or not LINE_CHANNEL_ACCESS_TOKEN:
    print("[Warning] LINE credentials not set. Webhook will not function.")

# ==================== LINE SDK Import ====================

try:
    from linebot import LineBotApi, WebhookHandler
    from linebot.exceptions import InvalidSignatureError
    from linebot.models import (
        MessageEvent, TextMessage, TextSendMessage,
        AudioMessage, ImageMessage, VideoMessage,
        FileMessage, LocationMessage, StickerMessage
    )
    
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(LINE_CHANNEL_SECRET)

    # Try to import v3 blob API (recommended for large media)
    try:
        from linebot.v3.messaging import MessagingApiBlob
        messaging_blob_api = MessagingApiBlob(channel_access_token=LINE_CHANNEL_ACCESS_TOKEN)
        print("[LINE SDK] Using v3 MessagingApiBlob for media downloads")
    except Exception:
        messaging_blob_api = None
        print("[LINE SDK] Using legacy API for media downloads")

    LINE_SDK_AVAILABLE = True
    print("[LINE SDK] Initialized successfully")
    
except Exception as e:
    print(f"[LINE SDK] Not available: {e}")
    LINE_SDK_AVAILABLE = False
    line_bot_api = None
    handler = None
    messaging_blob_api = None

# ==================== RAG Bot Instance ====================

rag_bot = BiomedicalRAGBot()
print("[RAG Bot] Initialized successfully")

# ==================== Webhook Endpoint ====================

@app.route("/", methods=["POST"])
def callback():
    """Main webhook endpoint for LINE messages."""
    if not LINE_SDK_AVAILABLE:
        return "LINE SDK not available", 503
        
    body = request.get_data(as_text=True)
    
    try:
        signature = request.headers['X-Line-Signature']
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("[Webhook] Invalid signature")
        return "Invalid signature", 400
    except Exception as e:
        print(f"[Webhook] Handler error: {e}")
        return "OK", 200

    return "OK", 200

# ==================== Helper Functions ====================

def _download_message_content(message_id: str, out_path: str) -> bool:
    """
    Download binary content for a LINE message to file.
    
    Args:
        message_id: LINE message ID
        out_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if messaging_blob_api is not None:
            resp = messaging_blob_api.get_message_content(message_id)
        elif line_bot_api is not None:
            resp = line_bot_api.get_message_content(message_id)
        else:
            print("[Media Download] No LINE SDK available")
            return False

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return True
        
    except Exception as e:
        print(f"[Media Download] Failed: {e}")
        return False


def _get_audio_duration_seconds(path: str) -> Optional[float]:
    """
    Get audio file duration using ffmpeg.
    
    Args:
        path: Audio file path
        
    Returns:
        Duration in seconds, or None if cannot determine
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10
        )
        output = result.stdout
        match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", output)
        if match:
            h, m, s = match.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)
            return duration
    except Exception as e:
        print(f"[Audio Duration] Parse failed: {e}")
    
    return None


def _transcribe_audio(audio_path: str) -> Optional[str]:
    """
    Transcribe audio file using OpenAI Whisper API.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Transcribed text, or None if failed
    """
    if not OPENAI_API_KEY:
        print("[STT] OPENAI_API_KEY not set")
        return None
    
    try:
        # Try new OpenAI client (>=1.0)
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        # Handle both object and dict responses
        text = getattr(transcript, "text", None) or (
            transcript.get("text") if isinstance(transcript, dict) else None
        )
        return text.strip() if text else None
        
    except ImportError:
        # Fallback to legacy openai package (0.28)
        try:
            import openai as openai_legacy
            openai_legacy.api_key = OPENAI_API_KEY
            
            with open(audio_path, "rb") as audio_file:
                transcript = openai_legacy.Audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            text = getattr(transcript, "text", None) or (
                transcript.get("text") if isinstance(transcript, dict) else None
            )
            return text.strip() if text else None
            
        except Exception as e:
            print(f"[STT] Legacy API failed: {e}")
            return None
    
    except Exception as e:
        print(f"[STT] Transcription failed: {e}")
        return None


def _send_reply(reply_token: str, answer: str):
    """
    Send answer to LINE user with automatic chunking.
    
    Args:
        reply_token: LINE reply token
        answer: Answer text to send
    """
    try:
        chunks = chunk_text(answer, max_len=1000)
        messages = [TextSendMessage(text=c) for c in chunks]
        line_bot_api.reply_message(reply_token, messages)
        print("[Reply] Sent successfully")
    except Exception as e:
        print(f"[Reply] Failed to send: {e}")


def _send_error_reply(reply_token: str, message: str):
    """Send error message to user."""
    try:
        line_bot_api.reply_message(reply_token, TextSendMessage(text=message))
    except Exception as e:
        print(f"[Reply] Failed to send error: {e}")


# ==================== Message Handlers ====================

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """Handle incoming text messages."""
    try:
        user_text = event.message.text.strip()
        
        # Validation
        if not user_text:
            _send_reply(event.reply_token, "您好，我是生醫QA小幫手～請使用文字或語音輸入生醫相關問題或敘述。")
            return
        
        if len(user_text) > 150:
            _send_reply(event.reply_token, "訊息太長囉！請嘗試使用短一點的敘述。")
            return

        print(f"[Text] Processing: {user_text}")

        # Call RAG bot
        try:
            answer, articles, confidence, answer_relevance, response_time = rag_bot.answer_question(user_text)
            print(f"[RAG] Response received ({len(answer)} chars, {response_time:.2f}s)")
            
        except Exception as rag_error:
            print(f"[RAG] Error: {rag_error}")
            _send_error_reply(event.reply_token, "很抱歉，處理您的問題時發生錯誤，請稍後再試或重新敘述。")
            return

        # Check answer validity
        if not answer or len(answer.strip()) == 0:
            _send_reply(event.reply_token, "抱歉，小幫手無法理解，請嘗試重新敘述您的問題。")
            return

        _send_reply(event.reply_token, answer)

    except Exception as e:
        print(f"[Text Handler] Unexpected error: {e}")
        _send_error_reply(event.reply_token, "很抱歉，小幫手遇到未預期的錯誤，請稍後再試。")


@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    """Handle incoming audio messages with STT."""
    try:
        # Download audio
        audio_path = "user_audio.m4a"
        if not _download_message_content(event.message.id, audio_path):
            _send_reply(event.reply_token, "很抱歉，小幫手無法取得語音檔案，請改用文字輸入。")
            return

        # Check duration
        duration = _get_audio_duration_seconds(audio_path)
        print(f"[Audio] Duration: {duration}s" if duration else "[Audio] Duration unknown")

        if not duration or duration <= 0:
            _send_reply(event.reply_token, "很抱歉，小幫手無法判斷語音長度，請改用短一點的語音或改成文字輸入。")
            return

        if duration > AUDIO_MAX_SECONDS:
            _send_reply(event.reply_token, f"語音訊息太長囉！請改用短一點的語音（≦{AUDIO_MAX_SECONDS}秒）或改成文字輸入。")
            return

        # Transcribe audio
        user_text = _transcribe_audio(audio_path)
        
        if not user_text:
            _send_reply(event.reply_token, "小幫手無法理解您的語音內容，請再試一次，或使用文字輸入。")
            return

        print(f"[Audio] Transcribed: {user_text}")

        # Call RAG bot
        try:
            answer, articles, confidence, answer_relevance, response_time = rag_bot.answer_question(user_text)
            print(f"[RAG] Response received ({len(answer)} chars, {response_time:.2f}s)")
            
        except Exception as rag_error:
            print(f"[RAG] Error: {rag_error}")
            _send_error_reply(event.reply_token, "很抱歉，處理您的問題時發生錯誤，請稍後再試或使用文字輸入。")
            return

        if not answer or len(answer.strip()) == 0:
            _send_reply(event.reply_token, "很抱歉，小幫手無法理解，請嘗試重新表述您的問題。")
            return

        # Prepend transcribed text to answer
        full_answer = f"小幫手聽到的內容是：{user_text}\n\n{answer}"
        _send_reply(event.reply_token, full_answer)

    except Exception as e:
        print(f"[Audio Handler] Unexpected error: {e}")
        _send_error_reply(event.reply_token, "很抱歉，小幫手遇到未預期的錯誤，請稍後再試。")


@handler.add(MessageEvent, message=(ImageMessage, VideoMessage, FileMessage, LocationMessage, StickerMessage))
def handle_other_messages(event):
    """Handle non-text/audio messages."""
    try:
        _send_reply(event.reply_token, "小幫手目前只支援文字和語音訊息喔！")
    except Exception as e:
        print(f"[Other Message] Error: {e}")


# ==================== Health Check ====================

@app.route("/healthz", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring."""
    return {"status": "ok", "rag_bot": "ready" if rag_bot else "unavailable"}, 200


# ==================== Main Entry Point ====================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    print(f"[Server] Starting on port {port}")
    print(f"[Config] Audio max duration: {AUDIO_MAX_SECONDS}s")
    
    app.run(host="0.0.0.0", port=port, debug=debug)