#!/usr/bin/env python3
"""
CLI entrypoint to run Voice Agent speech recognition and command extraction.
Supports WAV audio files, dataset benchmarks, synthetic audio, and simulated text commands.
"""

import argparse
import glob
import os
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.voice.agent import VoiceAgent
from src.agents.voice.audio_utils import generate_synthetic_tone, load_audio_file
from src.agents.voice.recognizer import MockSpeechRecognizer, SpeechCommandsRecognizer, WhisperSpeechRecognizer
from src.agents.voice.schemas import VoiceAgentInput, VoiceAgentOutput
from src.common.logger import setup_logger
from src.pipeline.voice_pipeline import VoicePipeline
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Voice Perception Agent for Adaptive Multimodal HRI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to custom voice_config.yaml")
    parser.add_argument("--audio-path", type=str, default=None, help="Path to WAV audio file input")
    parser.add_argument("--simulate", type=str, default=None, help="Simulate speech via text command")
    parser.add_argument("--speech-commands", action="store_true", help="Run batch test on datasets/Speech_commands/")
    parser.add_argument("--backend", type=str, default=None, choices=["whisper", "mock", "speech_commands"], help="Override recognizer backend")
    parser.add_argument("--mock-transcript", type=str, default="pick up the bottle on the left", help="Transcript when using mock backend")
    parser.add_argument("--synthetic", action="store_true", help="Test with a synthetic tone audio buffer")
    parser.add_argument("--output-json", action="store_true", help="Print full JSON formatted output")
    return parser.parse_args()


def print_voice_output(output: VoiceAgentOutput, print_json: bool = False):
    if print_json:
        print(output.to_json(indent=2))
        return

    print("\n" + "=" * 60)
    print("           VOICE AGENT PERCEPTION RESULT")
    print("=" * 60)
    print(f"Status              : {'SUCCESS' if output.success else 'FAILED'}")
    print(f"Speech Detected     : {output.is_speech_detected}")
    print(f"Transcript          : \"{output.transcript}\"")
    print(f"Normalized Text     : \"{output.normalized_text}\"")
    print(f"Language            : {output.language}")
    print(f"Confidence Score    : {output.confidence:.3f}")
    print(f"Audio Duration      : {output.audio_duration_sec:.2f} s")
    print(f"RMS Acoustic Energy : {output.rms_energy:.5f}")
    print(f"Keywords Detected   : {output.detected_keywords}")
    
    if output.speech_intent:
        intent = output.speech_intent
        print("\n--- Extracted Speech Intent ---")
        print(f"  Action            : {intent.action}")
        print(f"  Target Object     : {intent.target_object or 'None'}")
        print(f"  Spatial Sector    : {intent.spatial_sector or 'None'}")
        print(f"  Proximity Cue     : {intent.proximity_cue or 'None'}")
        print(f"  Urgency Level     : {intent.urgency}")
        print(f"  Intent Confidence : {intent.confidence:.3f}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    logger = setup_logger("RunVoice")

    # Load configuration
    config_path = args.config
    if config_path is None:
        default_voice_cfg = os.path.join(PROJECT_ROOT, "configs", "voice_config.yaml")
        if os.path.exists(default_voice_cfg):
            config_path = default_voice_cfg

    config = load_config(config_path) if config_path and os.path.exists(config_path) else {}
    if "voice" not in config:
        config["voice"] = {}

    if args.backend:
        config["voice"]["backend"] = args.backend

    backend = config["voice"].get("backend", "whisper")
    logger.info(f"Configured Voice Agent backend: '{backend}'")

    # Configure recognizer
    if backend == "mock":
        recognizer = MockSpeechRecognizer(default_transcript=args.mock_transcript)
    elif backend == "speech_commands":
        recognizer = SpeechCommandsRecognizer()
    else:
        recognizer = WhisperSpeechRecognizer(
            model_name=config["voice"].get("model_name", "base"),
            device=config["voice"].get("device", "auto"),
        )

    agent = VoiceAgent(config=config, recognizer=recognizer)
    agent.initialize()

    # Mode 1: Simulated text input
    if args.simulate:
        logger.info(f"Simulating speech command: \"{args.simulate}\"")
        output = agent.process(VoiceAgentInput(text_override=args.simulate))
        print_voice_output(output, print_json=args.output_json)
        agent.shutdown()
        return

    # Mode 2: Single WAV audio file
    if args.audio_path:
        if not os.path.exists(args.audio_path):
            logger.error(f"Audio file does not exist: {args.audio_path}")
            sys.exit(1)

        logger.info(f"Transcribing audio file: {args.audio_path}")
        output = agent.process(VoiceAgentInput(audio_data=args.audio_path))
        print_voice_output(output, print_json=args.output_json)
        agent.shutdown()
        return

    # Mode 3: Synthetic tone buffer
    if args.synthetic:
        logger.info("Generating synthetic 440Hz audio tone...")
        synth_audio = generate_synthetic_tone(freq_hz=440.0, duration_sec=1.5, sample_rate=16000)
        output = agent.process(VoiceAgentInput(audio_data=synth_audio, sample_rate=16000))
        print_voice_output(output, print_json=args.output_json)
        agent.shutdown()
        return

    # Mode 4: Batch test on datasets/Speech_commands/
    if args.speech_commands:
        dataset_dir = os.path.join(PROJECT_ROOT, "datasets", "Speech_commands")
        wav_files = sorted(glob.glob(os.path.join(dataset_dir, "*.wav")))
        if not wav_files:
            logger.error(f"No WAV files found in: {dataset_dir}")
            sys.exit(1)

        logger.info(f"Found {len(wav_files)} WAV files in speech commands dataset. Processing first 5 samples...")
        for i, wav_path in enumerate(wav_files[:5], start=1):
            logger.info(f"[{i}/5] Processing: {os.path.basename(wav_path)}")
            out = agent.process(VoiceAgentInput(audio_data=wav_path))
            print(f"  File: {os.path.basename(wav_path)} | RMS: {out.rms_energy:.5f} | Active: {out.is_speech_detected} | Action: {out.speech_intent.action if out.speech_intent else 'None'}")
        
        agent.shutdown()
        return

    # Default: Run interactive command demonstration
    logger.info("No input argument provided. Running interactive simulation demonstration...")
    demo_commands = [
        "Pick up the bottle on the left",
        "Stop moving immediately",
        "Please navigate to the dining table in the center",
        "Find the nearest cup",
        "What objects are on the table?",
    ]

    for cmd in demo_commands:
        logger.info(f"\nEvaluating command: \"{cmd}\"")
        out = agent.process(VoiceAgentInput(text_override=cmd))
        print_voice_output(out, print_json=args.output_json)

    agent.shutdown()


if __name__ == "__main__":
    main()
