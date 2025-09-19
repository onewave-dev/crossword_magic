import os
from fastapi import FastAPI, Request

# Создаём каталоги на диске, если их нет
os.makedirs("/var/data/puzzles", exist_ok=True)
os.makedirs("/var/data/states", exist_ok=True)

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# Пример вебхука
from telegram import Update
from telegram.ext import Application

# Инициализация PTB (упрощённо; подставьте свой токен)
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
telegram_app = Application.builder().token(BOT_TOKEN).build()

@app.post(os.getenv("WEBHOOK_PATH", "/webhook"))
async def webhook(req: Request):
    data = await req.json()
    update = Update.de_json(data, telegram_app.bot)
    await telegram_app.initialize()
    await telegram_app.process_update(update)
    return {"ok": True}

@app.get("/set_webhook")
async def set_webhook():
    from telegram.request import HTTPXRequest
    public_url = os.getenv("PUBLIC_URL")
    secret = os.getenv("WEBHOOK_SECRET", "")
    path = os.getenv("WEBHOOK_PATH", "/webhook")
    await telegram_app.initialize()
    await telegram_app.bot.set_webhook(
        url=f"{public_url}{path}",
        secret_token=secret,
        allowed_updates=[]
    )
    return {"status": "webhook set"}
