import pytorch_lightning as pl
import random

from models import *
from utils import *

from config import TrainConfig
from monotonic_align import mask_from_lens
from optimizers import build_optimizer
from losses import GeneratorLoss, DiscriminatorLoss, WavLMLoss, MultiResolutionSTFTLoss
from Utils.loss_utils import compute_ce_and_dur_losses, compute_s2s_loss
from Modules.hubert import EmotionRecognitionHubert


class StyleTTS2TrainingWrapper(pl.LightningModule):
    def __init__(self, config_path=""):
        super().__init__()
        self._steps_per_epoch = None
        self.multi_optimizer = None
        self.running_std = []
        self.automatic_optimization = False

        self.cfg = TrainConfig.from_json(config_path)

        self.hubert = EmotionRecognitionHubert(self.cfg.data_params.hubert_config)
        self.model = StyleTTS2(self.cfg).to(self.device)

        self.model.text_aligner.train()
        self.model.text_encoder.train()
        self.model.pitch_extractor.train()
        self.model.predictor.train()
        self.model.bert_encoder.train()
        self.model.bert.train()
        self.model.msd.train()
        self.model.mpd.train()

        self.generator_loss = GeneratorLoss(self.mpd, self.msd)
        self.discriminator_loss = DiscriminatorLoss(self.mpd, self.msd)
        self.wavlm_loss = WavLMLoss(self.cfg.model_params.slm.model,
                                    self.wd,
                                    self.cfg.preprocess_params.sr,
                                    self.cfg.model_params.slm.sr)
        self.stft_loss = MultiResolutionSTFTLoss()

    def configure_optimizers(self):
        optimizer_params = self.cfg.optimizer_params

        scheduler_params = {
            "max_lr": optimizer_params.lr,
            "pct_start": float(0),
            "epochs": self.cfg.epochs,
            "steps_per_epoch": self._steps_per_epoch
        }

        scheduler_params_dict = {
            key: scheduler_params.copy()
            for key, value in self.model.named_children()
            if key != "sampler"
        }

        scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
        scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
        scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2

        optimizer = build_optimizer(
            {
                name: module.parameters()
                for name, module in self.model.named_children()
                if name != "sampler"
            },
            scheduler_params_dict=scheduler_params_dict,
            lr=optimizer_params.lr
        )

        for g in optimizer.optimizers['bert'].param_groups:
            g['betas'] = (0.9, 0.99)
            g['lr'] = optimizer_params.bert_lr
            g['initial_lr'] = optimizer_params.bert_lr
            g['min_lr'] = 0
            g['weight_decay'] = 0.01

        for module in ["decoder", "style_encoder"]:
            for g in optimizer.optimizers[module].param_groups:
                g['betas'] = (0.0, 0.99)
                g['lr'] = optimizer_params.ft_lr
                g['initial_lr'] = optimizer_params.ft_lr
                g['min_lr'] = 0
                g['weight_decay'] = 1e-4

        self.multi_optimizer = optimizer

        return list(optimizer.optimizers.values())

    def set_steps_per_epoch(self, steps):
        self._steps_per_epoch = steps

    def training_step(self, batch, batch_idx):
        n_down = self.model.text_aligner.n_down

        waves, texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

        mask = length_to_mask(mel_input_length // (2 ** n_down)).to(self.device)
        text_mask = length_to_mask(input_lengths).to(self.device)

        if self.current_epoch >= self.cfg.loss_params.diff_epoch:
            ref_ss = self.model.style_encoder(ref_mels.unsqueeze(1))
            ref_sp = self.model.predictor_encoder(ref_mels.unsqueeze(1))
            ref = torch.cat([ref_ss, ref_sp], dim=1)
        else:
            ref = None

        try:
            ppgs, s2s_pred, s2s_attn = self.model.text_aligner(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
        except Exception:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        t_en = self.model.text_encoder(texts, input_lengths, text_mask)

        if bool(random.getrandbits(1)):
            asr = (t_en @ s2s_attn)
        else:
            asr = (t_en @ s2s_attn_mono)

        d_gt = s2s_attn_mono.sum(axis=-1).detach()

        ss = []
        gs = []

        for bib in range(len(mel_input_length)):
            mel = mels[bib, :, :mel_input_length[bib]]
            s = self.model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
            g = self.model.style_encoder(mel.unsqueeze(0).unsqueeze(1))
            ss.append(s)
            gs.append(g)

        s_dur = torch.stack(ss).squeeze()
        gs = torch.stack(gs).squeeze()
        s_trg = torch.cat([gs, s_dur], dim=-1).detach()

        bert_out = self.model.bert(texts, attention_mask=(~text_mask).int())
        d_en = self.model.bert_encoder(bert_out).transpose(-1, -2)

        loss_diff, loss_sty = 0, 0

        if self.current_epoch >= self.cfg.loss_params.diff_epoch:
            num_steps = random.randint(3, 5)
            noise = torch.randn(s_trg.size(0), 1, s_trg.size(1), dtype=torch.float32, device=self.device)

            if getattr(self.cfg.model_params.diffusion.dist, "estimate_sigma_data", False):
                sigma_estimate = s_trg.std(dim=-1).mean().item()
                self.running_std.append(sigma_estimate)
                self.model.diffusion.diffusion.sigma_data = float(np.mean(self.running_std))

            trg_emo_emb, emo_trg_logits = self.hubert.extract_features_from_list(waves, return_logits=True)

            s_preds = self.model.sampler(noise=noise,
                                         embedding=bert_out,
                                         features=ref+trg_emo_emb,
                                         embedding_scale=1,
                                         embedding_mask_proba=0.1,
                                         num_steps=num_steps).squeeze(1)

            loss_diff = self.model.diffusion(s_trg.unsqueeze(1), embedding=bert_out, features=ref).mean()
            loss_sty = F.l1_loss(s_preds, s_trg.detach())

        d, p = self.model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)
        mel_len = min(int(mel_input_length.min().item() / 2 - 1), self.cfg.max_len // 2)

        en, p_en, gt, wav = [], [], [], []

        for bib in range(len(mel_input_length)):
            mel_length = int(mel_input_length[bib].item() / 2)
            random_start = np.random.randint(0, mel_length - mel_len)

            en.append(asr[bib, :, random_start:random_start + mel_len])
            p_en.append(p[bib, :, random_start:random_start + mel_len])
            gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])

            y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
            wav.append(torch.from_numpy(y.astype(np.float32)).to(self.device))

        en = torch.stack(en)
        p_en = torch.stack(p_en)
        gt = torch.stack(gt).detach()
        wav = torch.stack(wav).float().detach()

        if gt.size(-1) < 80:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        s = self.model.style_encoder(gt.unsqueeze(1))
        s_dur = self.model.predictor_encoder(gt.unsqueeze(1))
        F0_fake, N_fake = self.model.predictor.F0Ntrain(p_en, s_dur)

        y_rec = self.model.decoder(en, F0_fake, N_fake, s)

        if self.current_epoch >= self.cfg.loss_params.diff_epoch:
            emo_rec_emb, emo_rec_logits = self.hubert.extract_features_from_tensor(y_rec.squeeze(1), return_logits=True)

            emo_emb_loss = F.mse_loss(emo_rec_emb, trg_emo_emb)

            with torch.no_grad():
                target_labels = torch.argmax(emo_trg_logits, dim=1)

            emo_logits_loss = F.cross_entropy(emo_rec_logits, target_labels)
        else:
            emo_emb_loss = torch.tensor(0.0, device=self.device)
            emo_logits_loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            F0_real, _, _ = self.model.pitch_extractor(gt.unsqueeze(1))
            N_real = log_norm(gt.unsqueeze(1)).squeeze(1)
            wav = wav.unsqueeze(1)

        d_loss = self.discriminator_loss(wav.detach(), y_rec.detach()).mean()
        d_loss.backward()
        self.multi_optimizer.step('mpd')
        self.multi_optimizer.step('msd')
        self.multi_optimizer.zero_grad()

        loss_F0_rec = F.smooth_l1_loss(F0_real, F0_fake) / 10
        loss_norm_rec = F.smooth_l1_loss(N_real, N_fake)
        loss_mel = self.stft_loss(y_rec, wav)
        loss_gen_all = self.generator_loss(wav, y_rec).mean()
        loss_lm = self.wavlm_loss(wav.squeeze(1), y_rec.squeeze(1)).mean()
        loss_ce, loss_dur = compute_ce_and_dur_losses(d, d_gt, input_lengths)
        loss_s2s = compute_s2s_loss(s2s_pred, texts, input_lengths)
        loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

        g_loss = (
                self.cfg.loss_params.lambda_mel * loss_mel +
                self.cfg.loss_params.lambda_F0 * loss_F0_rec +
                self.cfg.loss_params.lambda_ce * loss_ce +
                self.cfg.loss_params.lambda_norm * loss_norm_rec +
                self.cfg.loss_params.lambda_dur * loss_dur +
                self.cfg.loss_params.lambda_gen * loss_gen_all +
                self.cfg.loss_params.lambda_slm * loss_lm +
                self.cfg.loss_params.lambda_sty * loss_sty +
                self.cfg.loss_params.lambda_diff * loss_diff +
                self.cfg.loss_params.lambda_mono * loss_mono +
                self.cfg.loss_params.lambda_s2s * loss_s2s +
                emo_emb_loss +
                emo_logits_loss
        )

        g_loss.backward()

        self.multi_optimizer.step('bert')
        self.multi_optimizer.step('bert_encoder')
        self.multi_optimizer.step('predictor')
        self.multi_optimizer.step('predictor_encoder')
        self.multi_optimizer.step('style_encoder')
        self.multi_optimizer.step('decoder')
        self.multi_optimizer.step('text_encoder')
        self.multi_optimizer.step('text_aligner')

        if self.current_epoch >= self.cfg.loss_params.diff_epoch:
            self.multi_optimizer.step('diffusion')

        self.multi_optimizer.zero_grad()

        self.log('train/total_loss', g_loss, prog_bar=True)
        self.log('train/loss_mel', loss_mel, prog_bar=True)
        self.log('train/loss_gen', loss_gen_all, prog_bar=True)
        self.log('train/d_loss', d_loss, prog_bar=True)
        self.log('train/loss_diff', loss_diff, prog_bar=True)
        self.log('train/loss_sty', loss_sty, prog_bar=True)
        self.log('train/loss_ce', loss_ce, prog_bar=True)
        self.log('train/loss_dur', loss_dur, prog_bar=True)
        self.log('train/loss_F0', loss_F0_rec, prog_bar=True)
        self.log('train/loss_norm', loss_norm_rec, prog_bar=True)
        self.log('train/loss_lm', loss_lm, prog_bar=True)
        self.log('train/loss_mono', loss_mono, prog_bar=True)
        self.log('train/loss_s2s', loss_s2s, prog_bar=True)
        self.log('train/loss_emo_emb', emo_emb_loss, prog_bar=True)
        self.log('train/loss_emo_logits', emo_logits_loss, prog_bar=True)

        return g_loss

    def validation_step(self, batch, batch_idx):
        n_down = self.model.text_aligner.n_down

        waves, texts, input_lengths, ref_texts, ref_lengths, mels, mel_input_length, ref_mels = batch

        with torch.no_grad():
            mask = length_to_mask(mel_input_length // (2 ** n_down)).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)

            _, _, s2s_attn = self.model.text_aligner(mels, mask, texts)
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)

            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** n_down))
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            t_en = self.model.text_encoder(texts, input_lengths, text_mask)
            asr = (t_en @ s2s_attn_mono)
            d_gt = s2s_attn_mono.sum(axis=-1).detach()

            ss = []

            for bib in range(len(mel_input_length)):
                mel = mels[bib, :, :mel_input_length[bib]]
                s = self.model.predictor_encoder(mel.unsqueeze(0).unsqueeze(1))
                ss.append(s)

            s_dur = torch.stack(ss).squeeze()

            bert_out = self.model.bert(texts, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_out).transpose(-1, -2)
            d, p = self.model.predictor(d_en, s_dur, input_lengths, s2s_attn_mono, text_mask)

            mel_len = int(mel_input_length.min().item() / 2 - 1)
            en, p_en, gt, wav = [], [], [], []

            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)

                en.append(asr[bib, :, random_start:random_start + mel_len])
                p_en.append(p[bib, :, random_start:random_start + mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start + mel_len) * 2)])
                y = waves[bib][(random_start * 2) * 300:((random_start + mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y.astype(np.float32)).to(self.device))

            en = torch.stack(en)
            p_en = torch.stack(p_en)
            gt = torch.stack(gt).detach()
            wav = torch.stack(wav).float().detach()

            s = self.model.predictor_encoder(gt.unsqueeze(1))
            F0_fake, N_fake = self.model.predictor.F0Ntrain(p_en, s)

            y_rec = self.model.decoder(en, F0_fake, N_fake, self.model.style_encoder(gt.unsqueeze(1)))
            F0_real, _, _ = self.model.pitch_extractor(gt.unsqueeze(1))

            loss_F0 = F.l1_loss(F0_real, F0_fake) / 10
            loss_mel = self.stft_loss(y_rec.squeeze(), wav).mean()
            loss_ce, loss_dur = compute_ce_and_dur_losses(d, d_gt, input_lengths)

            self.log('val/loss_mel', loss_mel, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log('val/loss_dur', loss_dur, prog_bar=True, on_epoch=True, sync_dist=True)
            self.log('val/loss_F0', loss_F0, prog_bar=True, on_epoch=True, sync_dist=True)

            return {'mel_loss': loss_mel, 'dur_loss': loss_dur, 'F0_loss': loss_F0}


class StyleTTS2InferenceWrapper(nn.Module):
    def __init__(self, config_path="", device="cpu"):
        super().__init__()

        if config_path == "":
            raise ValueError("Please provide a config path")

        self.device = device
        self.cfg = TrainConfig.from_json(config_path)
        self.tokenizer = Tokenizer(self.cfg.model_params.vocab)
        self.model = StyleTTS2(self.cfg).to(device)

    def forward(self, text, s_pred):
        tokens = self.tokenizer(text)

        tokens.insert(0, 0)

        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)

        if s_pred.dim() == 1:
            s_pred = s_pred.unsqueeze(0)

        s_pred = s_pred.to(self.device)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))

            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(self.device)),
                                     F0_pred, N_pred, ref.squeeze().unsqueeze(0))

            return out.squeeze().cpu().numpy()
