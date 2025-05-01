import os
import torch
import torchaudio
import math
import random

from tqdm import tqdm

from speechbrain.inference import SpeakerRecognition
from pydub import AudioSegment
from pyannote.audio import Pipeline
from transformers import AutoModelForCTC, Wav2Vec2BertProcessor
from ipa.ipa_uk import ipa
from ukrainian_word_stress import Stressifier

from config import TrainConfig
from logger import setup_logging

logger = setup_logging(__name__)


def split_segment(start, end, max_len):
    duration = end - start

    if duration <= max_len:
        return [(start, end)]

    num_splits = math.ceil(duration / max_len)
    split_length = duration / num_splits

    return [(start + j * split_length, min(start + (j + 1) * split_length, end)) for j in range(num_splits)]


def save_split(data_split, split_path):
    with open(split_path, "w") as f:
        for wav, phon in data_split:
            f.write(f"{wav}|{phon}|0\n")


class DataPreprocessor:
    def __init__(self, config_path, hf_token=None, device="cuda"):
        self.cfg = TrainConfig.from_json(config_path)

        self.data_path = self.cfg.data_path
        self.ref_path = self.cfg.reference_path
        self.device = torch.device(device)
        self.num_speakers = self.cfg.num_speakers
        self.audio_dir = self.cfg.root_path

        os.makedirs(self.audio_dir, exist_ok=True)

        if self.num_speakers > 1:
            self.diarization_pipeline = Pipeline.from_pretrained(
                self.cfg.diarization.model,
                use_auth_token=hf_token,
            ).to(self.device)

            self.speaker_recognition = SpeakerRecognition.from_hparams(
                source=self.cfg.similarity.model,
                savedir="models/speaker_embedding"
            ).to(self.device)

        self.asr_model = AutoModelForCTC.from_pretrained(self.cfg.asr.model).to(self.device)
        self.asr_processor = Wav2Vec2BertProcessor.from_pretrained(self.cfg.asr.model)

        self.stressify = Stressifier()
        self.temp_files = []

    def preprocess_audio(self, path):
        """
        Preprocesses the audio file by loading it, converting to mono, and resampling.

        Args:
            path (str): Path to the audio file.
        Returns:
            tuple: Preprocessed waveform and sample rate.
        """
        waveform, sr = torchaudio.load(path)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.cfg.sr:
            waveform = torchaudio.transforms.Resample(sr, self.cfg.sr)(waveform)

        return waveform.to(self.device), self.cfg.sr

    def split_audio(self, chunk_seconds=60):
        """
        Splits the audio into chunks of specified duration.

        Args:
            chunk_seconds (int): Duration of each chunk in seconds.
        Returns:
            list: List of tuples containing the audio chunk and its index.
        """
        waveform, sr = self.preprocess_audio(self.data_path)
        int_waveform = (waveform.cpu().squeeze().numpy() * 32767).astype("int16")

        audio = AudioSegment(
            int_waveform.tobytes(),
            frame_rate=sr,
            sample_width=2,
            channels=1
        )

        chunk_ms = chunk_seconds * 1000
        duration_ms = len(audio)
        num_chunks = math.ceil(duration_ms / chunk_ms)

        return [(audio[i * chunk_ms:min((i + 1) * chunk_ms, duration_ms)], i) for i in range(num_chunks)]

    def get_embedding(self, wav_path):
        """
        Computes the speaker embedding for a given audio file.

        Args:
            wav_path (str): Path to the audio file.
        Returns:
            torch.Tensor: Speaker embedding.
        """
        waveform, _ = self.preprocess_audio(wav_path)

        feats = self.speaker_recognition.mods.compute_features(waveform)
        feats = self.speaker_recognition.mods.mean_var_norm(
            feats,
            torch.tensor([1.0], device=self.device)
        )
        embedding = self.speaker_recognition.mods.embedding_model(feats)

        return embedding.squeeze()

    def get_speaker_segments(self, chunk_audio, ref_embedding, idx):
        """
        Segments the audio chunk into speaker segments and computes the similarity scores.

        Args:
            chunk_audio (AudioSegment): The audio chunk.
            ref_embedding (torch.Tensor): Reference speaker embedding.
            idx (int): Index of the audio chunk.

        Returns:
            tuple: List of segments with their start and end times, and the path to the chunk audio.
        """
        chunk_path = os.path.join(self.audio_dir, f"temp_chunk_{idx}.wav")
        chunk_audio.export(chunk_path, format="wav")
        self.temp_files.append(chunk_path)

        diarization = self.diarization_pipeline(
            chunk_path,
            num_speakers=self.num_speakers
        )

        audio = AudioSegment.from_wav(chunk_path)

        speaker_segments = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment_path = os.path.join(self.audio_dir, f"tmp_{speaker}_{idx}.wav")
            seg = audio[turn.start * 1000:turn.end * 1000]
            seg.export(segment_path, format="wav")

            self.temp_files.append(segment_path)
            speaker_segments.setdefault(speaker, []).append((turn.start, turn.end, segment_path))

        speaker_scores = {}
        for speaker, segments in speaker_segments.items():
            embeddings = [self.get_embedding(p) for _, _, p in segments]
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            similarity = torch.nn.functional.cosine_similarity(ref_embedding, avg_embedding, dim=0)
            speaker_scores[speaker] = similarity.item()

        target_speaker = max(speaker_scores, key=speaker_scores.get)

        return [s[:2] for s in speaker_segments[target_speaker] if (s[1] - s[0]) >= self.cfg.min_length], chunk_path

    def prepare_samples(self, segments, chunk_path):
        """
        Prepares samples from the audio segments by splitting them into smaller segments,
        and performing ASR and phonemization.

        Args:
            segments (list): List of tuples containing start and end times of segments.
            chunk_path (str): Path to the audio chunk.
        Returns:
            list: List of tuples containing the file name and phonemized text.
        """
        full_audio = AudioSegment.from_wav(chunk_path)
        samples = []

        for i, (start, end) in enumerate(segments):
            split_points = split_segment(start, end, self.cfg.max_length)

            for sp_start, sp_end in split_points:
                segment = full_audio[sp_start * 1000:sp_end * 1000]

                f_name = f"chunk_{hash((chunk_path, sp_start, sp_end)) % 10 ** 6}.wav"
                path = os.path.join(self.audio_dir, f_name)
                segment.export(path, format="wav")

                waveform, sr = self.preprocess_audio(path)
                inputs = self.asr_processor(
                    waveform.cpu().squeeze().numpy(),
                    sampling_rate=sr,
                    return_tensors="pt"
                ).input_features

                with torch.inference_mode():
                    logits = self.asr_model(inputs.to(self.device)).logits

                ids = torch.argmax(logits, dim=-1)
                text = self.asr_processor.decode(ids[0])

                if not text.strip():
                    os.remove(path)
                    continue

                stressed = self.stressify(text)
                phonemized = ipa(stressed).strip()

                samples.append((f_name, phonemized))

        return samples

    def run(self):
        """
        Main function to run the data preprocessing pipeline.
        It splits the audio into chunks, processes each chunk to get speaker segments,
        and prepares samples for training and validation.
        Finally, it saves the training and validation data to specified files.
        """
        chunks = self.split_audio()
        all_samples = []

        if self.num_speakers > 1:
            ref_embedding = self.get_embedding(self.ref_path)

            for chunk_audio, idx in tqdm(chunks):
                segments, chunk_path = self.get_speaker_segments(chunk_audio, ref_embedding, idx)
                all_samples.extend(self.prepare_samples(segments, chunk_path))
        else:
            for chunk_audio, idx in chunks:
                chunk_path = os.path.join(self.audio_dir, f"temp_chunk_{idx}.wav")
                chunk_audio.export(chunk_path, format="wav")
                self.temp_files.append(chunk_path)
                segments = [(0, len(chunk_audio) / 1000)]
                all_samples.extend(self.prepare_samples(segments, chunk_path))

        for path in self.temp_files:
            if os.path.exists(path):
                os.remove(path)

        random.shuffle(all_samples)
        split = int(self.cfg.train_split * len(all_samples))
        train, val = all_samples[:split], all_samples[split:]

        save_split(train, self.cfg.train_data)
        save_split(val, self.cfg.val_data)


if __name__ == "__main__":
    processor = DataPreprocessor("Configs/data_preprocessing.json")
    processor.run()
