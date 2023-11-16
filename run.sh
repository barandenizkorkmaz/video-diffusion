python3 -W ignore main.py \
  --data_dir /usr/stud/korkmaz/storage/user/solar_nowcasting_data_asi/solar_nowcasting_data_asi.h5 \
  --split_dir /usr/stud/korkmaz/storage/user/solar_nowcasting_data_asi/splits/overfit/ \
  --num_train_timesteps 1000 \
  --num_inference_timesteps 100 \
  --resolution 32 \
  --num_epochs 1000 \
  --val_every_n_epoch 50 \
  --save_checkpoint_every_n_steps 500 \
  --save_checkpoint_path /usr/stud/korkmaz/storage/user/runs/