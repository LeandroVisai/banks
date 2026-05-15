"""Routers FastAPI: chat, search, health, images, catalog."""

from . import catalog, chat, health, images, search

__all__ = ["catalog", "chat", "health", "images", "search"]
