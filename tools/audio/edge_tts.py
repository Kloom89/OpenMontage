"""Microsoft Edge neural text-to-speech (free, no API key) via the edge-tts package."""

from __future__ import annotations

import asyncio
import importlib.util
import time
from pathlib import Path
from typing import Any

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)

DEFAULT_VOICE = "es-UY-ValentinaNeural"
DEFAULT_RATE = "+6%"


class EdgeTTS(BaseTool):
    name = "edge_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "edge"
    stability = ToolStability.PRODUCTION
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = ["pip:edge_tts"]
    install_instructions = "pip install edge-tts"
    agent_skills = ["text-to-speech"]

    capabilities = ["text_to_speech", "voice_selection", "multilingual"]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
    }
    best_for = [
        "free natural Spanish narration (es-UY-ValentinaNeural)",
        "localization — 300+ neural voices across 70+ languages at zero cost",
        "ads, promos and explainers without an API key",
    ]
    not_good_for = [
        "voice cloning",
        "emotion tags / voice direction",
    ]

    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string"},
            "voice": {"type": "string", "default": DEFAULT_VOICE},
            "voice_id": {"type": "string", "description": "Alias for 'voice'."},
            "rate": {"type": "string", "default": DEFAULT_RATE, "description": "e.g. +6%, -10%"},
            "speed": {"type": "number", "description": "Alias: 1.0 = +0%, 1.06 = +6%."},
            "pitch": {"type": "string", "default": "+0Hz"},
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=128, vram_mb=0, disk_mb=10, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=2, retryable_errors=["timeout", "connection"])
    idempotency_key_fields = ["text", "voice", "rate", "pitch"]
    side_effects = ["writes audio file to output_path"]
    user_visible_verification = ["Listen to generated audio for intelligibility"]

    def get_status(self) -> ToolStatus:
        if importlib.util.find_spec("edge_tts") is not None:
            return ToolStatus.AVAILABLE
        return ToolStatus.UNAVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        return 0.0

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        if self.get_status() != ToolStatus.AVAILABLE:
            return ToolResult(success=False, error="edge-tts not installed. " + self.install_instructions)

        start = time.time()
        try:
            result = self._generate(inputs)
        except Exception as exc:
            return ToolResult(success=False, error=f"Edge TTS generation failed: {exc}")
        result.duration_seconds = round(time.time() - start, 2)
        return result

    def _generate(self, inputs: dict[str, Any]) -> ToolResult:
        import edge_tts

        voice = inputs.get("voice") or inputs.get("voice_id") or DEFAULT_VOICE
        rate = inputs.get("rate")
        if not rate and inputs.get("speed"):
            rate = f"{round((float(inputs['speed']) - 1) * 100):+d}%"
        rate = rate or DEFAULT_RATE
        pitch = inputs.get("pitch", "+0Hz")

        output_path = Path(inputs.get("output_path", "tts_output.mp3"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        async def _run() -> None:
            await edge_tts.Communicate(inputs["text"], voice, rate=rate, pitch=pitch).save(str(output_path))

        asyncio.run(_run())

        if not output_path.exists() or output_path.stat().st_size == 0:
            return ToolResult(success=False, error=f"Edge TTS output missing/empty: {output_path}")

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "voice": voice,
                "rate": rate,
                "text_length": len(inputs["text"]),
                "output": str(output_path),
                "format": output_path.suffix.lstrip(".") or "mp3",
            },
            artifacts=[str(output_path)],
            model=voice,
        )
