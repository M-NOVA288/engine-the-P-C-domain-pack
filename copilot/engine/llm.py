"""LLM provider switch.

COPILOT_LLM=local (default)  -> deterministic template writer; no keys, no network,
                                fully reproducible evals.
COPILOT_LLM=azure            -> Azure OpenAI chat completions for the memo narrative.

The LLM is only ever asked to *phrase* findings that were computed and cited by
deterministic steps. It never decides accept/refer/decline; that keeps the
decision auditable and stops hallucinated facts from reaching an underwriter.
"""
from __future__ import annotations

import os


class LLM:
    def write(self, system: str, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class LocalTemplateLLM(LLM):
    """Returns the structured draft unchanged. Good enough for demos and CI."""

    def write(self, system: str, prompt: str) -> str:
        return prompt


class AzureOpenAILLM(LLM):
    def __init__(self) -> None:
        from openai import AzureOpenAI  # imported lazily so local mode needs no SDK

        self.client = AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
        self.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    def write(self, system: str, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or prompt


def get_llm() -> LLM:
    mode = os.environ.get("COPILOT_LLM", "local").lower()
    if mode == "azure":
        return AzureOpenAILLM()
    return LocalTemplateLLM()


MEMO_SYSTEM_PROMPT = (
    "You are an assistant to a commercial underwriter. Rewrite the draft memo for "
    "clarity. Do not add facts, numbers, or recommendations that are not in the draft. "
    "Keep every [citation] tag exactly where it is."
)
