# Handles dataset loading, splitting, and preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import torchvision.transforms as TV
from torch_audiomentations import Compose, Gain, AddColoredNoise, PitchShift, Shift
from datasets import load_dataset

N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SAMPLE_RATE = 16000
VIT_INPUT_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FREQ_MASK_PARAM = 25
TIME_MASK_PARAM = 40


class HandleChannels(nn.Module):
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.ndim == 2:
            spec = spec.unsqueeze(0)
        if spec.shape[0] == 1:
            spec = spec.repeat(3, 1, 1)
        return spec


mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    power=2.0,
)
amplitude_to_db_transform = T.AmplitudeToDB(stype="power", top_db=80.0)
freq_mask_transform = T.FrequencyMasking(freq_mask_param=FREQ_MASK_PARAM)
time_mask_transform = T.TimeMasking(time_mask_param=TIME_MASK_PARAM)
handle_channels_transform = HandleChannels()
resize_transform = TV.Resize(VIT_INPUT_SIZE, antialias=True)
normalize_transform = TV.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

time_domain_transforms_train = Compose(
    transforms=[
        # Apply gain variation
        Gain(min_gain_in_db=-5.0, max_gain_in_db=5.0, p=0.5),
        # Add Gaussian noise
        AddColoredNoise(
            min_snr_in_db=10.0,
            max_snr_in_db=25.0,
            min_f_decay=-2.0,
            max_f_decay=2.0,
            p=0.3,
        ),
        # Apply pitch shift
        PitchShift(
            min_transpose_semitones=-1,
            max_transpose_semitones=1,
            sample_rate=TARGET_SAMPLE_RATE,
            p=0.3,
        ),
        # Circularly shift the audio in time
        Shift(min_shift=-0.05, max_shift=0.05, p=0.4, sample_rate=TARGET_SAMPLE_RATE),
    ],
    output_type="dict",
    p=1,
)

eval_transforms = TV.Compose(
    [
        mel_spectrogram_transform,
        amplitude_to_db_transform,
        handle_channels_transform,
        resize_transform,
        normalize_transform,
    ]
)

training_transforms = TV.Compose(
    [
        mel_spectrogram_transform,
        amplitude_to_db_transform,
        # These transforms expect (..., freq, time)
        freq_mask_transform,
        time_mask_transform,
        # Image transforms
        handle_channels_transform,
        resize_transform,
        normalize_transform,
    ]
)


def preprocess_data(is_training: bool):
    """Returns the function that processes a BATCH dictionary"""

    processor = training_transforms if is_training else eval_transforms
    FIXED_LENGTH = 4 * TARGET_SAMPLE_RATE

    def preprocess_batch(batch: dict) -> dict:
        processed_spectrograms = []
        labels = []
        audio_list = batch["audio"]
        classID_list = batch["classID"]
        num_items = len(classID_list)

        for i in range(num_items):
            audio_data = audio_list[i]
            waveform = torch.from_numpy(audio_data["array"]).float()
            sample_rate = audio_data["sampling_rate"]
            label = torch.tensor(classID_list[i], dtype=torch.long)

            # Resample if necessary
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = T.Resample(
                    orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform)

            # Convert to mono if necessary
            if waveform.ndim > 1 and waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            if waveform.ndim == 0:
                waveform = waveform.unsqueeze(0)

            # --- Pad or crop to fixed length ---
            if waveform.shape[-1] < FIXED_LENGTH:
                padding_needed = FIXED_LENGTH - waveform.shape[-1]
                waveform = F.pad(waveform, (0, padding_needed))
            elif waveform.shape[-1] > FIXED_LENGTH:
                waveform = waveform[..., :FIXED_LENGTH]
            # --- End of Padding ---

            # Time domain augmentations
            if is_training:
                try:
                    # torch-audiomentations expects (batch_size, num_samples) or (batch_size, num_channels, num_samples)
                    # Adding a batch dimension
                    augmented_waveform = time_domain_transforms_train(
                        samples=waveform.unsqueeze(0).unsqueeze(1),
                        sample_rate=TARGET_SAMPLE_RATE,
                    )
                    waveform = augmented_waveform["samples"].squeeze()
                except Exception as e:
                    print(
                        f"Time domain augmentation failed: {e}. Using original waveform."
                    )
            # Apply the Compose pipeline
            processed_spectrogram = processor(waveform)

            processed_spectrograms.append(processed_spectrogram)
            labels.append(label)

        return {
            "pixel_values": torch.stack(processed_spectrograms),
            "labels": labels,
        }

    return preprocess_batch


def get_datasets():
    ds = load_dataset("danavery/urbansound8K")
    full_dataset = ds["train"]
    if "fold" in full_dataset.features:
        train_dataset = full_dataset.filter(lambda x: x["fold"] <= 8)
        val_dataset = full_dataset.filter(lambda x: x["fold"] == 9)
        test_dataset = full_dataset.filter(lambda x: x["fold"] == 10)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
    else:
        train_test_split = full_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = train_test_split["train"]
        test_temp_dataset = train_test_split["test"]
        val_test_split = test_temp_dataset.train_test_split(test_size=0.5, seed=42)
        val_dataset = val_test_split["train"]
        test_dataset = val_test_split["test"]

    train_dataset.set_transform(preprocess_data(is_training=True))
    val_dataset.set_transform(preprocess_data(is_training=False))
    test_dataset.set_transform(preprocess_data(is_training=False))
    return train_dataset, val_dataset, test_dataset
