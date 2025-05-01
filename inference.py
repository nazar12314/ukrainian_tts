import argparse
import torch
import soundfile as sf
import re

from ipa.ipa_uk import ipa
from ukrainian_word_stress import Stressifier, StressSymbol
from wrappers import StyleTTS2InferenceWrapper


stressify = Stressifier()


def split_text(text, min_len: int = 100):
    sentences = re.split(r'(?<=[.?!:])\s+', text)
    buffer = ""

    for sentence in sentences:
        if len(buffer) + len(sentence) > min_len:
            yield buffer.strip()
            buffer = ""

        buffer += sentence + " "

    if buffer:
        yield buffer.strip()


def process_text(text, min_len: int = 100):
    text = text.strip().replace('"', '')

    text = text.replace("+", StressSymbol.CombiningAcuteAccent)

    stressed = stressify(text)
    ipa_text = ipa(stressed)

    ipa_text = re.sub(r'[᠆‐‑‒–—―⁻₋−⸺⸻]', '-', ipa_text)
    ipa_text = re.sub(r'\s*-\s*', ': ', ipa_text)

    yield from split_text(ipa_text, min_len=min_len)


def main(args):
    text = ("Джури+нський водоспад – одне з неповторно красивих місць України. Це найбільший рівнинний водоспад "
            "Європи, який має шістнадцять метрів заввишки. Тече він біля села Ни+рків на Терно+пільщині. До речі, "
            "ця пам’ятка виникла завдяки втручанню людини.")

    tts = StyleTTS2InferenceWrapper(args.config, args.device)
    style = torch.load(args.style)

    audio_chunks = []

    for chunk in process_text(text):
        chunk_audio = tts(chunk, s_pred=style)
        audio_chunks.append(torch.tensor(chunk_audio))

    full_audio = torch.cat(audio_chunks, dim=-1)

    sf.write(args.out, full_audio, 24000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for StyleTTS2")
    parser.add_argument(
        "--config", type=str,
        default="Configs/inference_config.json",
        help="Path to the configuration file"
    )
    parser.add_argument(
        "--device", type=str,
        default="cpu",
        help="Device to run the inference on (e.g., 'cpu', 'cuda:0')"
    )
    parser.add_argument(
        "--style", type=str,
        default="styles/neutral.pt",
        help="Path to the file with styles of the speaker"
    )
    parser.add_argument(
        "--out", type=str,
        default="output.wav",
        help="Path to the output audio file"
    )

    arguments = parser.parse_args()

    main(arguments)
