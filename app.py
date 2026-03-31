import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from chatbot import BankChatbot

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تهيئة الشات بوت (يستخدم نفس قاعدة البيانات)
try:
    chatbot = BankChatbot()
    logger.info("✅ تم تهيئة الشات بوت")
except Exception as e:
    logger.error(f"❌ فشل تهيئة الشات بوت: {e}")
    chatbot = None

# بناء تطبيق تلغرام (سيتم تهيئته في lifespan)
telegram_app = Application.builder().token(TOKEN).build()

# ------------------- معالجات تلغرام -------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "مرحباً بك في مساعد بنك البركة الرقمي!\n\n"
        "يمكنك طرح أسئلتك حول خدمات البنك، الحسابات، التمويل، وغيرها.\n"
        "أنا هنا لمساعدتك. اسألني أي شيء 😊"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if chatbot is None:
        await update.message.reply_text("الخدمة غير متاحة حالياً، حاول لاحقاً.")
        return

    user_id = str(update.effective_user.id)
    question = update.message.text.strip()
    if not question:
        return

    # إظهار حالة "يكتب..."
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        full_response = ""
        async for chunk in chatbot.get_response_stream(question, session_id=user_id):
            full_response += chunk

        # تقسيم الرد إذا تجاوز الحد المسموح
        if len(full_response) > 4096:
            for i in range(0, len(full_response), 4096):
                await update.message.reply_text(full_response[i:i+4096])
        else:
            await update.message.reply_text(full_response)
    except Exception as e:
        logger.error(f"خطأ في معالجة السؤال: {e}", exc_info=True)
        await update.message.reply_text("عذراً، حدث خطأ أثناء معالجة سؤالك. حاول مجدداً لاحقاً.")

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# ------------------- Lifespan لإدارة التهيئة -------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # بدء التشغيل: تهيئة تطبيق تلغرام
    await telegram_app.initialize()
    logger.info("✅ تم تهيئة تطبيق تلغرام")
    yield
    # إيقاف التشغيل: إغلاق التطبيق
    await telegram_app.shutdown()
    logger.info("🛑 تم إغلاق تطبيق تلغرام")

app = FastAPI(lifespan=lifespan)

# ------------------- نقاط النهاية -------------------
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.process_update(update)
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Bot is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
