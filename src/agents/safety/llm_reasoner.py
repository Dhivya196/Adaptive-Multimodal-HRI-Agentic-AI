"""LLM-based contextual risk reasoning and fallback abstractions."""

from abc import ABC, abstractmethod
import json
import os
from typing import Any, Dict, Optional

from src.agents.safety.schemas import LLMContextualRiskAssessment, RiskLevel


class BaseLLMReasoner(ABC):
    """
    Abstract interface for LLM contextual risk reasoners.
    The LLM evaluates contextual ambiguity, multi-modal inconsistencies,
    and subtle risk factors beyond deterministic geometric boundaries.
    """

    @abstractmethod
    def assess_risk(self, safety_context: Dict[str, Any]) -> LLMContextualRiskAssessment:
        """Evaluate context and produce structured risk assessment."""
        pass


class MockLLMReasoner(BaseLLMReasoner):
    """
    Deterministic Mock LLM Reasoner for reliable, offline unit testing.
    Allows pre-configuring specific risk levels, ambiguity flags, or raising simulated errors.
    """

    def __init__(
        self,
        risk_level: RiskLevel = RiskLevel.LOW,
        contextually_safe: bool = True,
        uncertain: bool = False,
        reason: str = "Mock assessment: Context is clear and operation is safe.",
        confidence: float = 0.95,
        requires_human: bool = False,
        should_raise_error: bool = False,
        error_to_raise: Optional[Exception] = None,
    ):
        self.risk_level = risk_level
        self.contextually_safe = contextually_safe
        self.uncertain = uncertain
        self.reason = reason
        self.confidence = confidence
        self.requires_human = requires_human
        self.should_raise_error = should_raise_error
        self.error_to_raise = error_to_raise or RuntimeError("Simulated LLM API timeout/error")

    def assess_risk(self, safety_context: Dict[str, Any]) -> LLMContextualRiskAssessment:
        """Return configured mock risk assessment or raise simulated error."""
        if self.should_raise_error:
            raise self.error_to_raise

        return LLMContextualRiskAssessment(
            risk_level=self.risk_level,
            contextually_safe=self.contextually_safe,
            uncertain=self.uncertain,
            reason=self.reason,
            confidence=self.confidence,
            requires_human=self.requires_human,
            raw_response="[MockLLMReasoner response]",
        )


class OllamaLLMReasoner(BaseLLMReasoner):
    """
    Local Ollama LLM Contextual Risk Reasoner.
    Communicates directly with the local Ollama REST API (http://localhost:11434/api/chat)
    with format='json' and temperature=0.0.
    Falls back gracefully if the local Ollama daemon is offline or model is unavailable.
    """

    SYSTEM_PROMPT = """You are a Contextual Risk Reasoning module in a Human-Robot Interaction safety pipeline.
Analyze the provided structured safety context:
1. Is there contextual ambiguity or unspecified intent?
2. Are there conflicting multimodal referents?
3. Are there hazards not captured by basic geometric rules?
4. Should autonomous execution be paused for human confirmation?

Respond ONLY with a JSON object adhering to this schema:
{
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "contextually_safe": true | false,
  "uncertain": true | false,
  "reason": "<short 1-2 sentence justification>",
  "confidence": <float 0.0 to 1.0>,
  "requires_human": true | false
}"""

    def __init__(
        self,
        model_name: str = "llama3",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout_sec: float = 5.0,
    ):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout_sec = timeout_sec

    def assess_risk(self, safety_context: Dict[str, Any]) -> LLMContextualRiskAssessment:
        """
        Execute contextual safety prompt against local Ollama API.
        """
        import urllib.request
        import urllib.error

        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Safety Context:\n{json.dumps(safety_context, default=str)}"},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                message = data.get("message", {})
                content = message.get("content", "{}")
                parsed = json.loads(content)

                risk_lvl_str = parsed.get("risk_level", "MEDIUM").upper()
                try:
                    risk_lvl = RiskLevel(risk_lvl_str)
                except ValueError:
                    risk_lvl = RiskLevel.MEDIUM

                return LLMContextualRiskAssessment(
                    risk_level=risk_lvl,
                    contextually_safe=bool(parsed.get("contextually_safe", True)),
                    uncertain=bool(parsed.get("uncertain", False)),
                    reason=parsed.get("reason", f"Ollama ({self.model_name}) risk evaluation completed."),
                    confidence=float(parsed.get("confidence", 0.85)),
                    requires_human=bool(parsed.get("requires_human", False)),
                    raw_response=content,
                )

        except urllib.error.URLError as e:
            # Ollama service not running or connection refused
            return LLMContextualRiskAssessment(
                risk_level=RiskLevel.LOW,
                contextually_safe=True,
                uncertain=False,
                reason=f"Local Ollama server unreachable at {self.host} ({e}); falling back to deterministic safety.",
                confidence=1.0,
                requires_human=False,
            )
        except Exception as e:
            # JSON parsing or timeout failure
            return LLMContextualRiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                contextually_safe=False,
                uncertain=True,
                reason=f"Ollama inference error ({type(e).__name__}: {e}); falling back to safe intervention.",
                confidence=0.5,
                requires_human=True,
            )


class ProviderLLMReasoner(BaseLLMReasoner):
    """
    Live LLM Contextual Risk Reasoner supporting OpenAI-compatible APIs or remote models.
    Operates at temperature 0.0 for deterministic reasoning with JSON formatting.
    Falls back gracefully if the API is unavailable or credentials are not set.
    """

    SYSTEM_PROMPT = """You are a Contextual Risk Reasoning module in a Human-Robot Interaction safety pipeline.
Analyze the provided structured safety context:
1. Is there contextual ambiguity or unspecified intent?
2. Are there conflicting multimodal referents?
3. Are there hazards not captured by basic geometric rules?
4. Should autonomous execution be paused for human confirmation?

Respond ONLY with a JSON object adhering to this schema:
{
  "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "contextually_safe": true | false,
  "uncertain": true | false,
  "reason": "<short 1-2 sentence justification>",
  "confidence": <float 0.0 to 1.0>,
  "requires_human": true | false
}"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout_sec: float = 3.0,
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.api_key_env = api_key_env

    def assess_risk(self, safety_context: Dict[str, Any]) -> LLMContextualRiskAssessment:
        """
        Execute prompt against live LLM API if key is present, or return safe fallback assessment.
        """
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return LLMContextualRiskAssessment(
                risk_level=RiskLevel.LOW,
                contextually_safe=True,
                uncertain=False,
                reason=f"LLM API key ({self.api_key_env}) not set; falling back to deterministic evaluation.",
                confidence=1.0,
                requires_human=False,
            )

        try:
            # Optional runtime import for live provider calls
            import urllib.request
            import urllib.error

            payload = {
                "model": self.model_name,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Safety Context:\n{json.dumps(safety_context, default=str)}"},
                ],
                "response_format": {"type": "json_object"},
            }

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)

                risk_lvl_str = parsed.get("risk_level", "MEDIUM").upper()
                try:
                    risk_lvl = RiskLevel(risk_lvl_str)
                except ValueError:
                    risk_lvl = RiskLevel.MEDIUM

                return LLMContextualRiskAssessment(
                    risk_level=risk_lvl,
                    contextually_safe=bool(parsed.get("contextually_safe", True)),
                    uncertain=bool(parsed.get("uncertain", False)),
                    reason=parsed.get("reason", "Live LLM evaluation completed."),
                    confidence=float(parsed.get("confidence", 0.8)),
                    requires_human=bool(parsed.get("requires_human", False)),
                    raw_response=content,
                )

        except Exception as e:
            # Graceful fallback on any network, timeout, or parsing failure
            return LLMContextualRiskAssessment(
                risk_level=RiskLevel.MEDIUM,
                contextually_safe=False,
                uncertain=True,
                reason=f"Live LLM reasoning failed/timed out ({type(e).__name__}: {e}); falling back to safe intervention.",
                confidence=0.5,
                requires_human=True,
            )
