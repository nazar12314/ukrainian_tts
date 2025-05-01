import torch

from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from config import TrainConfig


class EmotionRecognitionHubert:
    def __init__(self, config_path, device, **kwargs):
        if not config_path:
            raise ValueError("Config path must be provided")

        self.cfg = TrainConfig.from_json(config_path)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.cfg.extractor_path, **kwargs)
        self.hubert_model = HubertForSequenceClassification.from_pretrained(self.cfg.model_path, **kwargs)

        self.device = device
        self.hubert_model.to(device).eval()

        for param in self.hubert_model.parameters():
            param.requires_grad = False

        self.max_length = self.cfg.max_length
        self.sampling_rate = self.cfg.sr

    def extract_features_from_list(self, waveforms, return_logits=False):
        if isinstance(waveforms[0], torch.Tensor):
            waveforms = [w.numpy() for w in waveforms]

        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.cfg.sr,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.cfg.max_length
        )

        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            out = self.hubert_model(input_values, output_hidden_states=True)
            emb = self.hubert_model.projector(out.hidden_states[-1].mean(dim=1))

        return (emb, out.logits) if return_logits else emb

    def extract_features_from_tensor(self, waveforms, return_logits=False):
        if self.feature_extractor.do_normalize:
            mean = waveforms.mean(dim=1, keepdim=True)
            std = waveforms.std(dim=1, keepdim=True) + 1e-5
            waveforms = (waveforms - mean) / std

        if self.max_length:
            B, T = waveforms.shape

            if T > self.max_length:
                waveforms = waveforms[:, :self.max_length]
            elif T < self.max_length:
                pad_len = self.max_length - T
                waveforms = torch.nn.functional.pad(waveforms, (0, pad_len))

        out = self.hubert_model(waveforms.to(self.device), output_hidden_states=True)
        emb = self.hubert_model.projector(out.hidden_states[-1].mean(dim=1))

        return (emb, out.logits) if return_logits else emb

    def map_emotions(self, logits):
        predicted_index = torch.argmax(logits, dim=1).item()

        predicted_emotion = self.cfg.emotion_map[str(predicted_index)]

        return predicted_emotion
