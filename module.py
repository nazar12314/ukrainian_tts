import pytorch_lightning as pl

from config import TrainConfig
from models import *
from Utils.ASR.models import ASRCNN
from transformers import AlbertConfig
from collections import OrderedDict
from Utils.JDC.model import JDCNet

from Modules.discriminators import MultiPeriodDiscriminator, MultiResSpecDiscriminator, WavLMDiscriminator


def build_text_aligner(args, model_weights):
    model = ASRCNN(**args)

    if "text_aligner" not in model_weights:
        raise ValueError("Text aligner not found in checkpoint")

    new_state_dict = {}

    for k, v in model_weights["text_aligner"].items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

    return model


def build_bert(args, model_weights):
    config = AlbertConfig(**args)
    bert = CustomAlbert(config)

    if "bert" not in model_weights:
        raise ValueError("BERT not found in checkpoint")

    new_state_dict = OrderedDict()

    for k, v in model_weights["bert"].items():
        name = k[7:]  # remove `module.`

        if name.startswith('encoder.'):
            name = name[8:]  # remove `encoder.`
            new_state_dict[name] = v

    bert.load_state_dict(new_state_dict, strict=False)

    return bert


def build_pitch_extractor(model_weights):
    F0_model = JDCNet(num_class=1, seq_len=192)

    if "pitch_extractor" not in model_weights:
        raise ValueError("Pitch extractor not found in checkpoint")

    new_state_dict = {}

    for k, v in model_weights["pitch_extractor"].items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v

    F0_model.load_state_dict(new_state_dict)

    _ = F0_model.train()

    return F0_model


class StyleTTS2(pl.LightningModule):
    def __init__(self, config_path=""):
        super().__init__()

        if not config_path:
            raise ValueError("Config path must be provided")

        # Load the configuration
        cfg = TrainConfig.from_json(config_path)

        self.cfg = cfg
        self.model_weights = torch.load(cfg.pretrained_model, map_location='cpu')

        assert cfg.model_params.decoder.type in ['istftnet', 'hifigan'], 'Decoder type unknown'

        self.initialize_first_stage(cfg.model_params)
        self.initialize_second_stage(cfg.model_params)

    def initialize_first_stage(self, args):
        text_aligner = build_text_aligner(args.text_aligner, self.model_weights)
        bert = build_bert(args.bert, self.model_weights)
        pitch_extractor = build_pitch_extractor(self.model_weights)

        self.add_module("text_aligner", text_aligner)
        self.add_module("bert", bert)
        self.add_module("pitch_extractor", pitch_extractor)

    def initialize_second_stage(self, args):
        if args.decoder.type == "istftnet":
            from Modules.istftnet import Decoder

            decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                              resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
                              upsample_rates=args.decoder.upsample_rates,
                              upsample_initial_channel=args.decoder.upsample_initial_channel,
                              resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                              upsample_kernel_sizes=args.decoder.upsample_kernel_sizes,
                              gen_istft_n_fft=args.decoder.gen_istft_n_fft,
                              gen_istft_hop_size=args.decoder.gen_istft_hop_size)
        else:
            from Modules.hifigan import Decoder

            decoder = Decoder(dim_in=args.hidden_dim, style_dim=args.style_dim, dim_out=args.n_mels,
                              resblock_kernel_sizes=args.decoder.resblock_kernel_sizes,
                              upsample_rates=args.decoder.upsample_rates,
                              upsample_initial_channel=args.decoder.upsample_initial_channel,
                              resblock_dilation_sizes=args.decoder.resblock_dilation_sizes,
                              upsample_kernel_sizes=args.decoder.upsample_kernel_sizes)

        self.add_module("decoder", decoder)

        text_encoder = TextEncoder(
            channels=args.hidden_dim, kernel_size=5,
            depth=args.n_layer, n_symbols=args.n_token
        )

        predictor = ProsodyPredictor(
            style_dim=args.style_dim, d_hid=args.hidden_dim,
            nlayers=args.n_layer, max_dur=args.max_dur, dropout=args.dropout
        )

        style_encoder = StyleEncoder(
            dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim
        )  # acoustic style encoder

        predictor_encoder = StyleEncoder(
            dim_in=args.dim_in, style_dim=args.style_dim, max_conv_dim=args.hidden_dim
        )  # prosodic style encoder

        self.add_module("text_encoder", text_encoder)
        self.add_module("predictor", predictor)
        self.add_module("style_encoder", style_encoder)
        self.add_module("predictor_encoder", predictor_encoder)
        self.add_module("bert_encoder", nn.Linear(self.bert.config.hidden_size, args.hidden_dim))

        transformer = StyleTransformer1d(
            channels=args.style_dim * 2, context_embedding_features=self.bert.config.hidden_size,
            context_features=args.style_dim * 2, **args.diffusion.transformer
        )

        diffusion = AudioDiffusionConditional(
            in_channels=1, embedding_max_length=self.bert.config.max_position_embeddings,
            embedding_features=self.bert.config.hidden_size,
            embedding_mask_proba=args.diffusion.embedding_mask_proba,  # Conditional dropout of batch elements,
            channels=args.style_dim * 2, context_features=args.style_dim * 2,
        )

        diffusion.diffusion = KDiffusion(
            net=diffusion.unet,
            sigma_distribution=LogNormalDistribution(mean=args.diffusion.dist.mean, std=args.diffusion.dist.std),
            sigma_data=args.diffusion.dist.sigma_data, dynamic_threshold=0.0
        )

        diffusion.diffusion.net = transformer
        diffusion.unet = transformer

        self.add_module("diffusion", diffusion)
        self.add_module("mpd", MultiPeriodDiscriminator())
        self.add_module("msd", MultiResSpecDiscriminator())
        self.add_module("wd", WavLMDiscriminator(args.slm.hidden, args.slm.nlayers, args.slm.initial_channel))
