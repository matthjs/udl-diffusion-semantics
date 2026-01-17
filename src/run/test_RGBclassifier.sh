python -m src.scripts.test_RGBclassifier \
  --data_dir=./data \
  --ckpt=./resnet_checkpoints/best_resnet18_rgb-epoch=10-val_acc=0.9100.pth \
  --batch_size=256 \
  --n_cpus=4