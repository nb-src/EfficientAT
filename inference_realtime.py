#!/usr/bin/env python3
"""inspired by EfficientAT inference.py and soundevice plot mic example"""

import argparse
import queue
import sys
import time
from collections import deque
from contextlib import nullcontext

import numpy as np
import sounddevice as sd
import torch
import torchaudio.transforms as T
from torch import autocast

from helpers.utils import NAME_TO_WIDTH, labels
from models.dymn.model import get_model as get_dymn
from models.ensemble import get_ensemble_model
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT

SAMPLE_RATE = 32000
SEGMENT_SIZE = 0.5
WINDOW_SIZE = 5  # 2.5 seconds

torch.set_num_threads(3)

model_name = "dymn10_as"
cuda = False
device = (
    torch.device("cuda") if cuda and torch.cuda.is_available() else torch.device("cpu")
)

window_size = 800
hop_size = 320
n_mels = 128

strides = (2, 2, 2, 2)
head_type = "mlp"


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l",
    "--list-devices",
    action="store_true",
    help="show list of audio devices and exit",
)
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser],
)
parser.add_argument(
    "-d", "--device", type=int_or_str, help="input device (numeric ID or substring)"
)
args = parser.parse_args(remaining)
q = queue.Queue()


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata[:, 0].copy())


# model to preprocess waveform into mel spectrograms
mel = AugmentMelSTFT(
    n_mels=n_mels, sr=SAMPLE_RATE, win_length=window_size, hopsize=hop_size
)
mel.to(device)
mel.eval()


def audio_tagging(waveform):
    """
    Running Inference on an audio clip.
    """
    # our models are trained in half precision mode (torch.float16)
    # run on cuda with torch.float16 to get the best performance
    # running on cpu with torch.float32 gives similar performance, using torch.bfloat16 is worse

    with (
        torch.no_grad(),
        autocast(device_type=device.type) if cuda else nullcontext(),
    ):
        spec = mel(waveform)
        preds, _features = model(spec.unsqueeze(0))
    preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()

    # sorted_indexes = np.argsort(preds)[::-1]

    # Print audio tagging top probabilities
    # print("************* Acoustic Event Detected: *****************")
    # for k in range(10):
    #     print(f"{labels[sorted_indexes[k]]}: {preds[sorted_indexes[k]]:.3f}")
    # print("********************************************************")
    return preds


# load pre-trained model
if len([]) > 0:
    model = get_ensemble_model(args.ensemble)
else:
    if model_name.startswith("dymn"):
        model = get_dymn(
            width_mult=NAME_TO_WIDTH(model_name),
            pretrained_name=model_name,
            strides=strides,
        )
    else:
        model = get_mobilenet(
            width_mult=NAME_TO_WIDTH(model_name),
            pretrained_name=model_name,
            strides=strides,
            head_type=head_type,
        )
model.to(device)
model.eval()


def main(device_id):
    args.device = device_id
    try:
        dev_sr = int(
            sd.query_devices(device=args.device, kind="input")["default_samplerate"]
        )
        if dev_sr == SAMPLE_RATE:
            stream = sd.InputStream(
                device=args.device,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=int(SAMPLE_RATE * SEGMENT_SIZE),
                dtype="float32",
                callback=audio_callback,
            )
        else:
            resampler = T.Resample(
                orig_freq=48000,
                new_freq=32000,
                lowpass_filter_width=6,
            ).to(device)
            stream = sd.InputStream(
                device=args.device,
                channels=1,
                blocksize=int(dev_sr * SEGMENT_SIZE),
                dtype="float32",
                callback=audio_callback,
            )
        d = deque(maxlen=WINDOW_SIZE)
        with stream:
            while True:
                new_block = q.get()
                waveform = torch.from_numpy(new_block[None, :]).to(device)
                if SAMPLE_RATE != dev_sr:
                    with torch.no_grad():
                        waveform = resampler(waveform)
                d.append(waveform)

                while True:  # always empty queue
                    try:
                        new_block = q.get_nowait()
                    except queue.Empty:
                        break
                    waveform = torch.from_numpy(new_block[None, :]).to(device)

                    if SAMPLE_RATE != dev_sr:
                        with torch.no_grad():
                            waveform = resampler(waveform)
                    d.append(waveform)
                audio_window = torch.cat(list(d), dim=1)
                t0 = time.perf_counter()
                preds = audio_tagging(audio_window)
                # print(f"Inference time: {(time.perf_counter() - t0) * 1000:.1f} ms")
                yield preds

    except Exception as e:
        parser.exit(1, type(e).__name__ + ": " + str(e))
