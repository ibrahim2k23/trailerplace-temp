from __future__ import annotations

import uvicorn

from src.api.app import create_app
from src.config import settings

app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=settings.chatbot_api_port)
