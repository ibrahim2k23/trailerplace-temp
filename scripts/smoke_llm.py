from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm.analyze import analyze_turn
from src.llm.client import OpenAILLMClient


def main() -> None:
    state = {
        "category": None,
        "messages": [
            {
                "role": "user",
                "content": "I want a 7x14 dump trailer for hauling dirt, I'm John, 555-1234",
            }
        ],
    }
    result = analyze_turn(OpenAILLMClient(), state)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
