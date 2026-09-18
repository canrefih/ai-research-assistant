import os

import requests

class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "")

    def answer(self, question: str, evidence: str) -> str:
        if not all((self.base_url, self.api_key, self.model)):
            return (
                "LLM configuration is incomplete.\n\n"
                "Retrieved evidence:\n" + evidence
            )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Answer only from the "
                        "supplied evidence. Cite sources using [1], [2], etc. "
                        "If evidence is insufficient, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nEvidence:\n{evidence}",
                },
            ],
            "temperature": 0.1,
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]
