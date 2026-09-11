# SHENet: Spatial relation-aware Hierarchical semantic Enhancement Network for Image Captioning
<img width="1255" height="557" alt="论文_框架图" src="https://github.com/user-attachments/assets/7f871a07-ffba-4ad2-b1ca-6f55e37bbd1a" />
<img width="1515" height="615" alt="论文_GSA" src="https://github.com/user-attachments/assets/654a1290-b356-4d99-b2d5-abff59768761" />


## Table of Contents
- [Environment setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Training](#training)

## Environment setup
Clone the repository and create a conda environment:
```bash
conda env create -f environment.yaml
conda activate shenet
```

## Data Preparation
- 📄 **Annotation:** Download the annotation file [annotations](https://drive.google.com/file/d/12EdMHuwLjHZPAMRJNrt3xSE2AMf7Tz8y/view?usp=sharing) Extract and put it in the project root directory.
- 🖼️ **Feature:** Grid visual features are extracted via [openai-clip-feature](https://github.com/jianjieluo/OpenAI-CLIP-Feature). Object visual feature are obtained using [VinVL](https://github.com/michelecafagna26/vinvl-visualbackbone).

## Training

## Training

Run `python train.py` with the following arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--exp_name` | Experiment name | `shenet` |
| `--batch_size` | Batch size for training | `50` |
| `--bs_reduct` | Batch size reduction factor | `5` |
| `--workers` | Number of dataloader workers | `6` |
| `--topk` | Top‑k selection | `8` |
| `--warmup` | Warmup training steps | `10000` |
| `--lr_xe` | Learning rate for XE loss | `1e‑4` |
| `--lr_rl` | Learning rate for RL loss | `5e‑6` |
| `--wd_rl` | Weight decay for RL optimizer | `0.05` |
| `--drop_rate` | Dropout rate | `0.1` |
| `--devices` | GPU device ids, support multiple gpus | `[0]` |

To train the model, you can run the following command:
```bash
python train.py \
  --devices 0 \
  --dataset_root ./data/features/ctx_features_dataset \
  --obj_file ./data/features/vinvl.hdf5 \
  --grid_file ./data/features/CLIP_features.hdf5 \
  --batch_size 50 \
  --lr_xe 1e-4
```
