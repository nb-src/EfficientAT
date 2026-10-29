#!/usr/bin/env python3
"""inspired by EfficientAT inference.py and soundevice plot mic example"""

import argparse
import copy
import queue
import sys
from collections import deque
from contextlib import nullcontext

import numpy as np
import sounddevice as sd
import torch
from torch import autocast

from helpers.utils import NAME_TO_WIDTH, labels
from models.dymn.model import get_model as get_dymn
from models.ensemble import get_ensemble_model
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT

SAMPLE_RATE = 32000
SEGMENT_SIZE = 0.5
BLOCKSIZE = int(SAMPLE_RATE * SEGMENT_SIZE)
WINDOW_SIZE = 5  # 2.5 seconds


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
    q.put(copy.copy(indata[:, 0]))


def audio_tagging(waveform):
    """
    Running Inference on an audio clip.
    """
    # model to preprocess waveform into mel spectrograms
    mel = AugmentMelSTFT(
        n_mels=n_mels, sr=SAMPLE_RATE, win_length=window_size, hopsize=hop_size
    )
    mel.to(device)
    mel.eval()

    waveform = torch.from_numpy(waveform[None, :]).to(device)

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

    sorted_indexes = np.argsort(preds)[::-1]

    # Print audio tagging top probabilities
    print("************* Acoustic Event Detected: *****************")
    for k in range(10):
        print(f"{labels[sorted_indexes[k]]}: {preds[sorted_indexes[k]]:.3f}")
    print("********************************************************")


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


try:
    stream = sd.InputStream(
        device=args.device,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        callback=audio_callback,
    )
    d = deque(maxlen=WINDOW_SIZE)
    with stream:
        while True:
            new_block = q.get()
            d.append(new_block)
            while True:  # always empty queue
                try:
                    new_block = q.get_nowait()
                except queue.Empty:
                    break
                d.append(new_block)
            audio_window = np.concatenate(d)
            audio_tagging(audio_window)

except Exception as e:
    parser.exit(1, type(e).__name__ + ": " + str(e))
