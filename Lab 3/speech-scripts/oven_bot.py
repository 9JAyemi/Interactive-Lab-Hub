#!/usr/bin/env python3
"""A speech-driven oven helper based on the Lab 3 echo_bot loop.

Run from the Lab 3 virtual environment:
    python speech-scripts/oven_bot.py

It can list saved recipes, repeat the available cornbread notes, start and
check a cooking timer, and answer basic cooking-time questions. It does not
control or sense a real oven. Use the recipe/package directions and check food
doneness independently; a timer is only a reminder.
"""

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice
from piper.config import SynthesisConfig

SAMPLE_RATE = 16000
LAB_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VAD = LAB_DIR / "models" / "silero_vad.onnx"
DEFAULT_VOICE = LAB_DIR / "voices" / "en_US-lessac-medium.onnx"

# These are the saved recipe titles from the Lab 3 oven scenario. Only the
# cornbread ingredients and 20-minute bake-time note were provided there.
RECIPES = {
    "chocolate chip cookies": {
        "aliases": ("chocolate chip cookies", "cookies"),
        "ingredients": None,
        "minutes": None,
    },
    "brownies": {"aliases": ("brownies", "brownie"), "ingredients": None, "minutes": None},
    "pizza": {"aliases": ("pizza",), "ingredients": None, "minutes": None},
    "cornbread": {
        "aliases": ("cornbread", "corn bread"),
        "ingredients": (
            "1 package Jiffy Corn Muffin Mix, half a cup of salted butter, melted, "
            "half a cup of sour cream, 2 to 3 tablespoons of sugar, and 2 large eggs, beaten."
        ),
        "minutes": 20,
    },
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
UNIT_SECONDS = {
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "hour": 3600,
    "hours": 3600,
    "hr": 3600,
    "hrs": 3600,
}


def words_to_number(words: str) -> int | None:
    """Convert simple spoken numbers such as 'twenty five' to an integer."""
    total = 0
    for word in words.split():
        value = NUMBER_WORDS.get(word)
        if value is None:
            return None
        if value >= 20 and total % 100 == 0:
            total += value
        else:
            total += value
    return total


def parse_duration(text: str) -> int | None:
    """Return a requested timer duration in seconds, if one was spoken."""
    numeric = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b",
        text,
    )
    if numeric:
        amount = float(numeric.group(1))
        unit = numeric.group(2)
        multiplier = 3600 if unit.startswith(("h",)) else 60 if unit.startswith("m") else 1
        return max(1, round(amount * multiplier))

    unit_pattern = r"seconds?|secs?|minutes?|mins?|hours?|hrs?"
    spoken = re.search(
        rf"\b((?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        rf"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
        rf"forty|fifty|sixty|seventy|eighty|ninety)(?:[ -](?:one|two|three|four|five|six|"
        rf"seven|eight|nine))?)\s+({unit_pattern})\b",
        text,
    )
    if not spoken:
        return None

    amount = words_to_number(spoken.group(1).replace("-", " "))
    if amount is None:
        return None
    unit = spoken.group(2)
    multiplier = 3600 if unit.startswith(("h",)) else 60 if unit.startswith("m") else 1
    return max(1, amount * multiplier)


def format_duration(seconds: int) -> str:
    """Format seconds into a short spoken duration."""
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} {'hour' if hours == 1 else 'hours'}")
    if minutes:
        parts.append(f"{minutes} {'minute' if minutes == 1 else 'minutes'}")
    if remaining_seconds or not parts:
        parts.append(
            f"{remaining_seconds} {'second' if remaining_seconds == 1 else 'seconds'}"
        )
    return " and ".join(parts)


class Speaker:
    """Synthesizes text with Piper and plays it through the default output."""

    def __init__(self, voice_path: Path) -> None:
        self.voice = PiperVoice.load(str(voice_path))
        # A larger length scale makes Piper speak more slowly.
        self.synthesis_config = SynthesisConfig(length_scale=1.25)

    def say(self, text: str) -> None:
        for chunk in self.voice.synthesize(text, syn_config=self.synthesis_config):
            audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            sd.play(audio, samplerate=chunk.sample_rate)
            sd.wait()


class OvenAssistant:
    """Small offline dialogue policy with one active countdown timer."""

    def __init__(self) -> None:
        self.selected_recipe: str | None = None
        self.timer_deadline: float | None = None
        self.timer_label = "cooking"

    @staticmethod
    def find_recipe(text: str) -> str | None:
        for name, recipe in RECIPES.items():
            if any(alias in text for alias in recipe["aliases"]):
                return name
        return None

    def finish_timer_if_due(self) -> str | None:
        if self.timer_deadline is not None and time.monotonic() >= self.timer_deadline:
            label = self.timer_label
            self.timer_deadline = None
            return label
        return None

    def reply(self, heard: str) -> str:
        text = re.sub(r"[^a-z0-9 ]+", " ", heard.lower())
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "Sorry, I didn't catch that. Please try again."

        if any(word in text.split() for word in ("goodbye", "quit", "exit")):
            return "Goodbye. Please check the oven and your food directly."

        if "cancel timer" in text or "stop timer" in text:
            if self.timer_deadline is None:
                return "There isn't an active timer to cancel."
            self.timer_deadline = None
            return "Timer cancelled. Please keep an eye on the food yourself."

        duration = parse_duration(text)
        if "timer" in text and duration is not None:
            if duration > 24 * 60 * 60:
                return "That timer is longer than one day. Please choose a shorter duration."
            self.timer_deadline = time.monotonic() + duration
            self.timer_label = self.selected_recipe or "cooking"
            return f"Timer started for {format_duration(duration)}."

        if "timer" in text and any(word in text.split() for word in ("start", "set", "begin")):
            recipe = RECIPES.get(self.selected_recipe) if self.selected_recipe else None
            if recipe and recipe["minutes"]:
                duration = recipe["minutes"] * 60
                self.timer_deadline = time.monotonic() + duration
                self.timer_label = self.selected_recipe
                return f"Starting a 20-minute cornbread timer now. Please check the food directly for doneness."
            return "How many minutes should I set the timer for?"

        asks_remaining = (
            any(phrase in text for phrase in ("time left", "time remaining", "how long left"))
            or ("time" in text and any(word in text.split() for word in ("left", "remaining")))
        )
        if asks_remaining:
            finished = self.finish_timer_if_due()
            if finished:
                return f"Your {finished} timer has finished. Please check the food directly."
            if self.timer_deadline is None:
                return "There is no active timer. Tell me to start a timer after the food is in the oven."
            remaining = max(0, math.ceil(self.timer_deadline - time.monotonic()))
            return f"There are {format_duration(remaining)} left on your {self.timer_label} timer."

        if any(phrase in text for phrase in ("saved recipes", "recipe list", "which recipes", "what recipes")):
            names = list(RECIPES)
            joined = ", ".join(names[:-1]) + f", and {names[-1]}"
            return (
                f"Your saved recipes are {joined}. I have ingredient and timing notes "
                "for cornbread; the other saved entries are titles only."
            )

        recipe_name = self.find_recipe(text)
        if recipe_name:
            self.selected_recipe = recipe_name
            recipe = RECIPES[recipe_name]
            if any(word in text for word in ("recipe", "ingredient", "directions", "how to make")):
                if recipe["ingredients"] is None:
                    return (
                        f"I have {recipe_name} saved by name, but its ingredients and directions "
                        "aren't in my recipe notes yet."
                    )
                return (
                    f"For the cornbread, you need {recipe['ingredients']} The saved bake-time "
                    "note is 20 minutes, but it doesn't include an oven temperature. Follow the "
                    "package directions for temperature. Once the food is in the oven, tell me "
                    "to start the timer."
                )

            if any(word in text for word in ("bake", "cook", "how long", "time")):
                if recipe["minutes"] is None:
                    return f"I don't have a saved cooking time for {recipe_name}. Check its recipe or package directions."
                return (
                    "The saved cornbread note says 20 minutes, but doesn't specify temperature. "
                    "Use the package directions, and check for doneness rather than relying on a timer alone."
                )

        placed = (
            any(
                phrase in text
                for phrase in ("put it in the oven", "put it in", "placed it", "in the oven now", "just put")
            )
            or bool(re.search(r"\bput\b.+\b(?:in|into)\b.+\boven\b", text))
        )
        if placed:
            recipe_name = recipe_name or self.selected_recipe
            recipe = RECIPES.get(recipe_name) if recipe_name else None
            if recipe and recipe["minutes"]:
                duration = recipe["minutes"] * 60
                self.timer_deadline = time.monotonic() + duration
                self.timer_label = recipe_name
                return f"Starting a 20-minute cornbread timer now. Please check the food directly for doneness."
            return "I don't have a saved cooking time for that. How many minutes should I set the timer for?"

        if "timer" in text or "time left" in text or "time remaining" in text:
            return "Say, for example, start a timer for 20 minutes, or ask how much time is left."
        if any(word in text for word in ("hello", "hi", "hey")):
            return "Hello. Ask me to list saved recipes, repeat the cornbread recipe, or set a cooking timer."
        return (
            "I can list saved recipes, read the cornbread notes, or set and check a timer. "
            "I don't control or sense the oven, so follow recipe directions and check the food yourself."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="tiny.en", help="Whisper model size (default: tiny.en)")
    parser.add_argument("--vad-model", type=Path, default=DEFAULT_VAD)
    parser.add_argument("--voice", type=Path, default=DEFAULT_VOICE)
    parser.add_argument("--min-silence", type=float, default=0.6,
                        help="silence duration that ends a turn (default: 0.6 seconds)")
    args = parser.parse_args()

    for path, label in ((args.vad_model, "VAD model"), (args.voice, "Piper voice")):
        if not path.is_file():
            sys.exit(f"{label} not found at {path}. Run ./setup.sh from Lab 3 first.")

    print("Loading speech models...", flush=True)
    recognizer = WhisperModel(args.model, device="cpu", compute_type="int8")
    speaker = Speaker(args.voice)
    assistant = OvenAssistant()

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = str(args.vad_model)
    config.silero_vad.min_silence_duration = args.min_silence
    config.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
    window = config.silero_vad.window_size
    samples_per_read = int(0.1 * SAMPLE_RATE)
    buffer = np.empty(0, dtype=np.float32)

    speaker.say(
        "Hello, I'm your oven helper. Ask for saved recipes, cooking times, or a timer. "
        "I cannot control or sense the oven. Say goodbye to stop."
    )
    print(f"Listening; turns end after {args.min_silence} seconds of silence. Press Ctrl-C to stop.\n")

    try:
        with sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE) as stream:
            def speak_without_listening(text: str) -> None:
                """Pause capture during TTS so the assistant cannot hear itself."""
                stream.stop()
                sd.stop()
                try:
                    speaker.say(text)
                finally:
                    stream.start()

                # Drain the device's short playback/driver tail without feeding
                # it to VAD, then discard any partial speech from the previous turn.
                for _ in range(4):
                    stream.read(samples_per_read)
                vad.reset()

            while True:
                finished = assistant.finish_timer_if_due()
                if finished:
                    message = f"Your {finished} timer is finished. Please check the food directly."
                    print(f"  oven: {message}")
                    speak_without_listening(message)
                    buffer = np.empty(0, dtype=np.float32)

                chunk, _ = stream.read(samples_per_read)
                buffer = np.concatenate([buffer, chunk.reshape(-1)])

                while len(buffer) > window:
                    vad.accept_waveform(buffer[:window])
                    buffer = buffer[window:]

                while not vad.empty():
                    utterance = np.asarray(vad.front.samples, dtype=np.float32)
                    vad.pop()
                    segments, _ = recognizer.transcribe(utterance, beam_size=1)
                    heard = " ".join(segment.text.strip() for segment in segments)
                    if not heard:
                        continue

                    reply = assistant.reply(heard)
                    print(f"  heard: {heard}\n  oven:  {reply}\n", flush=True)
                    speak_without_listening(reply)
                    buffer = np.empty(0, dtype=np.float32)
                    if any(word in re.sub(r"[^a-z ]", " ", heard.lower()).split()
                           for word in ("goodbye", "quit", "exit")):
                        return
    except KeyboardInterrupt:
        print("\nOven helper stopped.")


if __name__ == "__main__":
    main()