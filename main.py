import argparse
import torch
import pytorch_lightning as pl
from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
    RandomRotation,
)
from diffusers import DPMSolverMultistepScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
import wandb

from src.callbacks.log_model import LogModelWightsCallback
from src.data.reader import ASIReader
from src.data.dataset import ASIDatamodule
from src.utils.model import circle_mask
from src.model.model import DiffusionModel

def parse_args():
    parser = argparse.ArgumentParser(description="Training script of video diffusion.")
    parser.add_argument("--num_prev_asi", type=int, default=5)
    parser.add_argument("--num_post_asi", type=int, default=5)
    parser.add_argument("--time_delta", type=int, default=1)
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        #default="/home/dominik/Documents/AuSeSol/Data/HDF5/solar_nowcasting_data_asi.h5",
        help="Path to the HDF5 file containing the dataset.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--beta_schedule", type=str, default="linear") # TODO: Different modes such as cosine and sigmoid should be added later.
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of inference steps.")

    parser.add_argument(
        "--train_batch_size", type=int, default=128, help="Batch size (per device) for the train dataloaders."
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=256, help="Batch size (per device) for the evaluation dataloaders."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process"
        )
    )
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
    )
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=False,
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
    parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
    parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
    parser.add_argument("--overwrite_output_dir", action="store_true")

    # More Arguments Added by Baran Deniz Korkmaz
    parser.add_argument("--resume-from", nargs="?", type=str, default=None, help="The path to checkpoint to resume training")
    parser.add_argument("--logger_type", type=str, default="wandb", help="The type of logger")

    args = parser.parse_args()
    return args

def train(args):
    asi_reader = ASIReader(
        hdf5_file=args.data_dir,
        key_group='/PSA/station_HP/cloud_cam_hp', # TODO: ?
        num_prev_asi=args.num_prev_asi,
        num_post_asi=args.num_post_asi,
        time_delta=args.time_delta,
        exclude_current_asi=False,
        unit='min',
        squeeze_single_asi=True
    )

    train_transform = Compose(
        [
            ToTensor(),
            Resize((args.resolution, args.resolution), interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(),
            RandomRotation(degrees=(0, 360), interpolation=InterpolationMode.BILINEAR), #TODO: Why Random Rotation ?
            Normalize([0.5], [0.5]),
        ]
    )

    eval_transform = Compose(
        [
            ToTensor(),
            Resize((args.resolution, args.resolution), interpolation=InterpolationMode.BILINEAR),
            Normalize([0.5], [0.5]),
        ]
    )

    datamodule = ASIDatamodule(
        asi_reader=asi_reader,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        train_transform=train_transform,
        eval_transform=eval_transform
    )

    datamodule.prepare_data()
    datamodule.setup('fit')

    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3 * (args.num_prev_asi + 1 + args.num_post_asi), #TODO: Why + 1 ? 3 probably stands for the number of channels (RGB)
        out_channels=3 * args.num_post_asi, # Note: The size of output is equal to the number of current frames (to be predicted)
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    noise_scheduler = DPMSolverMultistepScheduler( # TODO: Dig deeper here!
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        solver_order=3,
        prediction_type='epsilon',
        algorithm_type='dpmsolver++'
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(datamodule.train_dataloader()) * args.num_epochs),
    )

    ema_model = EMAModel(
        model,
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        max_value=args.ema_max_decay
    ) if args.use_ema else None

    mask = circle_mask(args.resolution, args.resolution, args.resolution // 2 + 3) # TODO: Understand here

    diffusion_model = DiffusionModel(
        model, noise_scheduler, args.num_inference_steps, optimizer, lr_scheduler,
        ema_model=ema_model, num_samples=10, mask=mask
    ) # TODO: num_samples is 3 by default

    # Create logger
    if args.logger_type == "wandb":
        wandb.login()
        wandb.init(
            project="base-diffusion",
            #id=run_id,
            entity="thesis",
            config=args
        )
        logger = pl_loggers.WandbLogger(project="base-diffusion", log_model="all")
        logger.watch(diffusion_model)
    elif args.logger_type == "tensorboard":
        try:
            logger = pl_loggers.TensorBoardLogger(save_dir=args.logging_path)
        except:
            raise ValueError("Missing arguments for logging_path")
    else:
        logger = None

    trainer = pl.Trainer(
        check_val_every_n_epoch=1,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=args.num_epochs,
        accelerator='auto',
        # devices=-1,
        # strategy='ddp'
        callbacks=[
            LogModelWightsCallback(log_every=10)
        ]
    )
    trainer.fit(
        model=diffusion_model,
        datamodule=datamodule,
        ckpt_path=args.resume_from
    )
    #trainer.validate(
    #    model=diffusion_model, datamodule=datamodule,
    #    ckpt_path='/home/wiss/schnaus/PycharmProjects/video_diffusion/asi diffusion/y33egh2p/checkpoints/epoch=510-step=1179178.ckpt')



if __name__ == "__main__":
    args = parse_args()

    # Make everything deterministic
    # seed_everything(args.seed)

    # Run training process
    print(f"Running training process..")
    try:
        # TODO: There are hard-coded arguments. Fine-tune arguments!
        train(args)
    except KeyboardInterrupt:
        print("Training successfully interrupted.")
