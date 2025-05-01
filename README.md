# Development of an Enhanced Ukrainian Text-to-Speech System

This repository is adaptation of [StyleTTS2](https://github.com/yl4579/StyleTTS2), designed for fine-tuning and inference on Ukrainian data.
We refactored the original implementation of training process of StyleTTS2 model, by covering the functionality in Pytorch lightning module.
Also, we provide scripts for training and inference, as well as a data preprocessing pipeline, which you can use to prepare a custom dataset for training.

Our training module supports only fine-tuning of StylTTS2 model, in multi-speaker manner.

 We provide our trained checkpoint, together with collected data and examples on [Google Drive](https://drive.google.com/drive/folders/1_Sy6rNe3nV-1TMnvuwHNZaBBmRvqnN2y?usp=share_link). 

### Installation

```bash
pip install -r requirements.txt
```

Other requirements:
- ffmpeg
- NVIDIA GPU with CUDA support

### Fine-tuning

To fine-tune the **StyleTTS2** model on your own dataset, use the `finetune.py` script. Make sure your config file is properly set up (see `config/train_config.json` for reference).

Fine-tuning will be automatically performed with Emotion control functionality.

#### Usage:

```bash
python finetune.py \
  --config path/to/your_config.json \
  --gpus 1 \
  --num_workers 4
```

#### Arguments
- `--config`: (Required) Path to the training config JSON file.
- `--gpus`: Number of GPUs to use (set to 0 for CPU training).
- `--num_workers`: Number of data loader worker processes (default: 2).

### Inference

To synthesize speech from text using a pre-trained or fine-tuned StyleTTS2 model, use the `inference.py` script.

#### Usage:

```bash
python inference.py \
  --config path/to/inference_config.json \
  --device cuda:0 \
  --style path/to/style.pt \
  --out output.wav
```

#### Arguments
- `--config`: Path to the inference configuration file (default: `Configs/inference_config.json`)
- `--device`: Device for inference (cpu, cuda, or cuda:0 for specific GPU)
- `--style`: Path to a .pt file with the speaker's style embedding (you can find some preprocessed styles in Google Drive)
- `--out`: Path to save the generated audio (.wav)

In order to set custom text for generation, you have to adjust text field in main function of the `inference.py` script.

### Data preprocessing

To prepare your dataset for fine-tuning StyleTTS2, use the `data_preprocessing.py` script. This script:

- Performs speaker diarization and speaker matching (if multi-speaker).
- Applies ASR transcription.
- Stress-marks and phonemizes Ukrainian text with [ipa-uk](https://github.com/lang-uk/ipa-uk) module.
- Outputs processed audio chunks, together with training and validation file lists in the expected format.

#### Usage:
Before launching script, make sure that you set hf_token while instantiating DataPreprocessor object.

```bash
python3 data_preprocessing.py
```

Make sure to update the config path inside the script if needed:

```processor = DataPreprocessor("Configs/data_preprocessing.json", hf_token=<your token>)```

or adjust already created config file. 

#### Output
- `train_list.txt`, `val_list.txt` in the format: ```chunk_123456.wav|phonemized_text|0```
- Corresponding `.wav` chunks saved under `Data/wavs/`

### Additional Notions

The most important thing that you should remember are config files. If you set up them appropriately, you will be able to run all scripts without any problems.

