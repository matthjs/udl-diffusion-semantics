python -m src.scripts.train_RGBclassifier \
  --data_dir=./data \
  --save_dir=./ckpts_rgb \
  --n_epochs=100 \
  --batch_size=128 \
  --n_cpus=4 \
  --lr=1e-3 \
  --weight_decay=5e-4 \
  --freeze_until=layer4 \
  --unfreeze_epoch=5