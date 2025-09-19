"""FastAPI application entrypoint for Telegram webhook processing."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from telegram import Update
from telegram.ext import Application
from telegram.request import HTTPXRequest

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Settings:
    """Container for application environment variables."""

    telegram_bot_token: str
    public_url: str
    webhook_secret: str
    webhook_path: str = "/webhook"
    webhook_check_interval: int = 300


def load_settings() -> Settings:
    """Load and validate required settings from environment variables."""

    required_vars = {
        "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN"),
        "PUBLIC_URL": os.getenv("PUBLIC_URL"),
        "WEBHOOK_SECRET": os.getenv("WEBHOOK_SECRET"),
        "WEBHOOK_PATH": os.getenv("WEBHOOK_PATH", "/webhook"),
    }

    missing = [name for name, value in required_vars.items() if not value]
    if missing:
        logger.critical("Missing required environment variables: %s", ", ".join(missing))
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    logger.debug("Loaded environment variables: %s", {k: v for k, v in required_vars.items() if k != "TELEGRAM_BOT_TOKEN"})

    interval_raw = os.getenv("WEBHOOK_CHECK_INTERVAL", "300")
    check_interval = 300
    with suppress(ValueError):
        check_interval = max(int(interval_raw), 60)
    if not interval_raw.isdigit():
        logger.debug("WEBHOOK_CHECK_INTERVAL is not a digit, defaulting to 300 seconds")
        check_interval = 300

    return Settings(
        telegram_bot_token=required_vars["TELEGRAM_BOT_TOKEN"],
        public_url=required_vars["PUBLIC_URL"].rstrip("/"),
        webhook_secret=required_vars["WEBHOOK_SECRET"],
        webhook_path=required_vars["WEBHOOK_PATH"] if required_vars["WEBHOOK_PATH"].startswith("/") else f"/{required_vars['WEBHOOK_PATH']}",
        webhook_check_interval=check_interval,
    )


# ---------------------------------------------------------------------------
# Directories for storage
# ---------------------------------------------------------------------------


def ensure_storage_directories() -> None:
    """Ensure that persistent storage directories exist."""

    for path in ("/var/data/puzzles", "/var/data/states"):
        os.makedirs(path, exist_ok=True)
        logger.debug("Ensured storage directory exists: %s", path)


# ---------------------------------------------------------------------------
# FastAPI application and telegram application state
# ---------------------------------------------------------------------------


app = FastAPI()


class AppState:
    """Shared state container for the FastAPI application."""

    def __init__(self) -> None:
        self.settings: Optional[Settings] = None
        self.telegram_app: Optional[Application] = None
        self.webhook_task: Optional[asyncio.Task[None]] = None


state = AppState()


def get_telegram_application() -> Application:
    if state.telegram_app is None:
        logger.error("Telegram application is not initialized")
        raise HTTPException(status_code=503, detail="Telegram application is not initialized")
    return state.telegram_app


def register_webhook_route(path: str) -> None:
    """Register the webhook endpoint for the configured path."""

    router = app.router
    for route in list(router.routes):
        if getattr(route, "endpoint", None) is telegram_webhook:
            logger.debug("Removing existing webhook route bound to %s", getattr(route, "path", "<unknown>"))
            router.routes.remove(route)

    logger.debug("Registering webhook route at path %s", path)
    router.add_api_route(path, telegram_webhook, methods=["POST"], name="telegram_webhook")


# ---------------------------------------------------------------------------
# Webhook monitoring
# ---------------------------------------------------------------------------


async def monitor_webhook(application: Application, settings: Settings) -> None:
    """Background task to periodically ensure webhook registration is valid."""

    logger.debug("Starting webhook monitor task with interval %s seconds", settings.webhook_check_interval)
    expected_url = f"{settings.public_url}{settings.webhook_path}"
    while True:
        try:
            info = await application.bot.get_webhook_info()
            logger.debug("Current webhook info: url=%s, pending=%s", info.url, info.pending_update_count)
            if info.url != expected_url or info.secret_token != settings.webhook_secret:
                logger.warning("Webhook mismatch detected. Expected url=%s secret token=%s", expected_url, settings.webhook_secret)
                await application.bot.set_webhook(
                    url=expected_url,
                    secret_token=settings.webhook_secret,
                    allowed_updates=[],
                )
                logger.info("Webhook re-registered due to mismatch")
        except Exception:  # noqa: BLE001 - We want to log all failures
            logger.exception("Failed to validate or reset webhook")

        await asyncio.sleep(settings.webhook_check_interval)


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    logger.debug("FastAPI startup initiated")
    ensure_storage_directories()

    settings = load_settings()
    state.settings = settings

    logger.debug("Building Telegram application")
    httpx_request = HTTPXRequest(
        additional_headers={"Authorization": f"Bearer {settings.telegram_bot_token}"}
    )
    telegram_application = (
        Application.builder()
        .token(settings.telegram_bot_token)
        .request(httpx_request)
        .updater(None)
        .build()
    )

    await telegram_application.initialize()
    logger.info("Telegram application initialized")

    state.telegram_app = telegram_application

    register_webhook_route(settings.webhook_path)

    expected_url = f"{settings.public_url}{settings.webhook_path}"
    logger.debug("Ensuring webhook is configured for %s", expected_url)
    await telegram_application.bot.set_webhook(
        url=expected_url,
        secret_token=settings.webhook_secret,
        allowed_updates=[],
    )
    logger.info("Webhook configured at %s", expected_url)

    state.webhook_task = asyncio.create_task(monitor_webhook(telegram_application, settings))


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.debug("FastAPI shutdown initiated")

    if state.webhook_task:
        logger.debug("Cancelling webhook monitor task")
        state.webhook_task.cancel()
        with suppress(asyncio.CancelledError):
            await state.webhook_task
        state.webhook_task = None

    if state.telegram_app:
        logger.debug("Shutting down Telegram application")
        await state.telegram_app.shutdown()
        await state.telegram_app.stop()
        state.telegram_app = None
        logger.info("Telegram application shut down")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> JSONResponse:
    logger.debug("Health check requested")
    return JSONResponse({"status": "ok"})


async def telegram_webhook(
    request: Request,
    telegram_application: Application = Depends(get_telegram_application),
) -> JSONResponse:
    settings = state.settings
    if settings is None:
        logger.error("Application settings are not available during webhook call")
        raise HTTPException(status_code=503, detail="Application settings unavailable")

    secret_header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
    secret_query = request.query_params.get("secret_token")
    if settings.webhook_secret not in {secret_header, secret_query}:
        logger.warning(
            "Webhook secret mismatch: header=%s query=%s",
            secret_header,
            secret_query,
        )
        raise HTTPException(status_code=403, detail="Invalid secret token")

    try:
        payload = await request.json()
        logger.debug("Received webhook payload: %s", payload)
        update = Update.de_json(payload, telegram_application.bot)
    except Exception as exc:  # noqa: BLE001 - we need to report deserialization errors
        logger.exception("Failed to deserialize Telegram update")
        raise HTTPException(status_code=400, detail="Invalid update payload") from exc

    try:
        await telegram_application.process_update(update)
    except Exception as exc:  # noqa: BLE001 - log any processing errors
        logger.exception("Failed to process Telegram update")
        raise HTTPException(status_code=500, detail="Failed to process update") from exc
    logger.debug("Update processed successfully")
    return JSONResponse({"ok": True})


async def _set_webhook(telegram_application: Application, settings: Settings) -> None:
    expected_url = f"{settings.public_url}{settings.webhook_path}"
    logger.debug("Setting webhook to %s", expected_url)
    await telegram_application.bot.set_webhook(
        url=expected_url,
        secret_token=settings.webhook_secret,
        allowed_updates=[],
    )


@app.get("/set_webhook")
async def set_webhook(
    telegram_application: Application = Depends(get_telegram_application),
) -> JSONResponse:
    settings = state.settings
    if settings is None:
        logger.error("Attempted to set webhook without settings available")
        raise HTTPException(status_code=503, detail="Application settings unavailable")

    await _set_webhook(telegram_application, settings)
    return JSONResponse({"status": "webhook set"})


@app.get("/reset_webhook")
async def reset_webhook(
    telegram_application: Application = Depends(get_telegram_application),
) -> JSONResponse:
    settings = state.settings
    if settings is None:
        logger.error("Attempted to reset webhook without settings available")
        raise HTTPException(status_code=503, detail="Application settings unavailable")

    logger.debug("Deleting current webhook before reconfiguration")
    await telegram_application.bot.delete_webhook(drop_pending_updates=False)
    await _set_webhook(telegram_application, settings)
    return JSONResponse({"status": "webhook reset"})


# ---------------------------------------------------------------------------
# Dynamic webhook path mounting
# ---------------------------------------------------------------------------


__all__ = ["app"]

