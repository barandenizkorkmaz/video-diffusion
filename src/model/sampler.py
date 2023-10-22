from typing import List, Optional, Tuple, Union
from inspect import signature

import torch

from diffusers.configuration_utils import FrozenDict
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
# from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import deprecate


class Sampler(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        previous_images: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]: # TODO: This function is actually sampler!
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """
        message = (
            "Please make sure to instantiate your scheduler with `prediction_type` instead. E.g. `scheduler ="
            " DDPMScheduler.from_pretrained(<model_id>, prediction_type='epsilon')`."
        )
        predict_epsilon = deprecate("predict_epsilon", "0.12.0", message, take_from=kwargs)

        if predict_epsilon is not None:
            new_config = dict(self.scheduler.config)
            new_config["prediction_type"] = "epsilon" if predict_epsilon else "sample"
            self.scheduler._internal_dict = FrozenDict(new_config)

        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{self.device}`, so the `generator` will be ignored. "
                f'Please use `torch.Generator(device="{self.device}")` instead.'
            )
            deprecate(
                "generator.device == 'cpu'",
                "0.12.0",
                message,
            )
            generator = None

        batch_size, num_channels, height, width = previous_images.shape
        new_num_channels = self.unet.in_channels - num_channels

        # Sample gaussian noise to begin loop
        sample_shape = (batch_size, new_num_channels, height, width)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            sample = torch.randn(sample_shape, generator=generator)
            sample = sample.to(self.device)
        else:
            sample = torch.randn(sample_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.scheduler.timesteps: # TODO: Does scheduler automatically determines the required timesteps? Otherwise, it would redundantly go through all the steps?
            # 1. predict noise model_output
            model_input = torch.cat([previous_images, sample], dim=1)
            # Note: UNet predicts the epsilon in the current timestep t
            model_output = self.unet(model_input, t).sample

            scheduler_step_kwargs = dict(
                model_output=model_output,
                timestep=t,
                sample=sample,
            )
            if 'generator' in signature(self.scheduler.step).parameters.keys():
                scheduler_step_kwargs['generator'] = generator

            # 2. compute previous image: x_t -> x_t-1
            sample = self.scheduler.step(**scheduler_step_kwargs).prev_sample

        if not return_dict:
            return (sample,)

        return ImagePipelineOutput(images=sample)
