import torch
import numpy as np
import wandb

def to_tensor(*elements):
    return [torch.as_tensor(e, dtype=torch.float) for e in elements]

def denormalize(images, height, width):
    denormalized = (images / 2 + 0.5) * 255 # After normalization, the images are in [-1,+1]
    reshaped = denormalized.view(-1, 3, height, width)
    return reshaped.cpu().numpy().round().astype("uint8")

def concat_and_video(previous, video):
    # TODO: Separate into two parts: 1. Concatenate video frames 2. Publish video on Wandb
    video = np.concatenate([previous, video], axis=0)
    return wandb.Video(video, fps=1, format="gif")