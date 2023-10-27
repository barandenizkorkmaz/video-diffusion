import pytorch_lightning as pl
import numpy as np
import wandb

from .loss import *
from src.utils.data import denormalize, concat_and_video
from .sampler import Sampler


class DiffusionModel(pl.LightningModule):

    def __init__(
            self,
            model,
            noise_scheduler,
            num_inference_steps,
            optimizer=None,
            lr_scheduler=None,
            ema_model=None,
            num_samples=3,
            mask=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.num_inference_steps = num_inference_steps
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.ema_model = ema_model

        self.seeds = torch.randint(
            0, 2 ** 32, (num_samples,),
            generator=torch.manual_seed(0))

        self.mask = mask

        self.metric_functions = [masked_mae_loss, masked_rmse_loss, masked_mse_loss]
        self.metric_names = ["mae", "rmse", "mse"]
        self.val_metrics = {
            metric_name: []
            for metric_name in self.metric_names
        }

    def forward(self, previous_images, num_samples=8) -> list[np.ndarray]:
        B = previous_images.shape[0]
        height, width = previous_images.shape[-2:]
        previous_images = previous_images.view(B, -1, height, width) # TODO: Concatenate all frames of a video in color channel?

        pipeline = Sampler( # Actually sampler
            unet=self.ema_model.averaged_model if self.ema_model is not None else self.model,
            scheduler=self.noise_scheduler,
        )

        videos = [[] for _ in range(previous_images.shape[0])] # TODO: B is the number of videos probably?

        seeds = torch.randint(
            0, 2 ** 32, (num_samples,),
            generator=torch.manual_seed(0))

        for seed in seeds:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed.item())
            # run pipeline in inference (sample random noise and denoise)
            images = pipeline(
                previous_images=previous_images,
                generator=generator,
                num_inference_steps=self.num_inference_steps
            ).images

            # denormalize the images
            for video, image in zip(videos, images):
                video.append(image.clamp(-1, 1))

        # Note: Denormalization takes concatenated frames as input!
        videos = np.stack([
            np.stack([
                denormalize(image, height, width)
                for image in video])
            for video in videos])
        return videos

    def training_step(self, batch, batch_idx):
        previous_images, clean_images = batch
        # Sample noise that we'll add to the images
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bsz = clean_images.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        # TODO: I understand here that clean_images denote the immediate current frames (to be predicted).
        # TODO: Why noise_scheduler take a noise input instead of automatically generating noise differently at each timestep? (i.e. sample noise timestep times)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Note: Dim is 1
        all_images = torch.cat([previous_images, noisy_images], dim=1)

        # Predict the noise residual
        # Note: The 2D-UNet model denoises the samples for t timesteps!
        # Note: .sample for obtaining the sample
        # TODO: Here we denoise all_images which contains past frames + current frames, where do we apply masking?
        # TODO: self.forward is not used?
        # TODO: Please note that, from my understanding, the model outputs the predicted noise!
        # I think that this one line below should be in `forward` pass?
        # TODO: How do we use DPM+ Solver in the backward process? -> We don't because we are only aiming at predicting the noise added in the current timestep.
        model_output = self.model(all_images, timesteps).sample

        # Note: I guess that the model outputs predicted noise and we calculate loss based on the predicted vs actual noise
        loss = F.mse_loss(model_output, noise)  # this could have different weights!

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        if self.ema_model:
            self.ema_model.averaged_model.to(self.device)
            self.ema_model.step(self.model)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "monitor": "train_loss",
                "strict": False,
                "name": None
            }
        }

    def validation_step(self, batch, batch_idx):
        previous_images, clean_images = batch

        height, width = previous_images.shape[-2:]

        pipeline = Sampler(
            unet=self.ema_model.averaged_model if self.ema_model is not None else self.model,
            scheduler=self.noise_scheduler,
        )

        videos = [[] for _ in range(previous_images.shape[0])]

        for seed in self.seeds:
            generator = torch.Generator(device=pipeline.device).manual_seed(seed.item())
            # run pipeline in inference (sample random noise and denoise)
            images = pipeline(
                previous_images=previous_images,
                generator=generator,
                num_inference_steps=self.num_inference_steps
            ).images

            # denormalize the images
            for video, image in zip(videos, images):
                video.append(image.clamp(-1, 1))

        clean_images = [denormalize(image, height, width) for image in clean_images]
        previous_images = [denormalize(image, height, width) for image in previous_images]
        videos = [np.stack([denormalize(image, height, width) for image in video]) for video in videos]

        for clean_image, video in zip(clean_images, videos):
            for metric_function, metric_name in zip(self.metric_functions, self.metric_names):
                metrics = [metric_function(v, clean_image, self.mask).item() for v in video]
                self.val_metrics[metric_name].append(metrics)

        num_samples = 20 # TODO: Why?
        previous_samples = batch_idx * len(previous_images)
        if previous_samples <= num_samples:
            table = wandb.Table(
                columns=["previous images", "ground truth"] + [f"seed {seed.item()}" for seed in self.seeds])
            remaining_samples = num_samples - previous_samples
            for _, video, gt, previous in zip(range(remaining_samples), videos, clean_images, previous_images):
                gt = concat_and_video(previous, gt)
                video = [concat_and_video(previous, v) for v in video]
                table.add_data(wandb.Video(previous, fps=1, format="gif"), gt, *video)

            wandb.log({"validation": table, "global_step": self.global_step})

    def on_validation_epoch_end(self, outputs):
        metrics = np.array([self.val_metrics[metric_name] for metric_name in self.metric_names])

        # scatter plot
        ids = np.arange(metrics.shape[1])[:, None].repeat(metrics.shape[2], axis=1)
        scatter_metrics = np.concatenate([ids[None], metrics], axis=0)
        scatter_metrics = scatter_metrics.reshape(scatter_metrics.shape[0], -1)
        scatter_table = wandb.Table(data=scatter_metrics.T, columns=["image id"] + self.metric_names)
        for name in self.metric_names:
            wandb.log({f"val_{name}_plot": wandb.plot.scatter(scatter_table, x="image id", y=name, title=name)})

        # line plots and aggregations
        aggregations = [np.mean, np.min, np.max]
        for aggregation in aggregations:
            aggregated_metrics = aggregation(metrics, axis=2)
            for name, metric in zip(self.metric_names, aggregated_metrics):
                self.log(f"val_{name}_{aggregation.__name__}", metric.mean(), on_step=False, on_epoch=True)

        self.val_metrics = {metric_name: [] for metric_name in self.metric_names}