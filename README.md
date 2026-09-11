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

Run `python train.py` using the following arguments:

| Argument | Possible values |
|----------|-----------------|
| `--exp_name` | Experiment name |
| `--batch_size` | Batch size (default: 50) |
| `--workers` | Number of workers, accelerate model training in the xe stage. |
| `--head` | Number of heads (default: 8) |
| `--resume_last` | If used, the training will be resumed from the last checkpoint. |
| `--resume_best` | If used, the training will be resumed from the best checkpoint. |
| `--features_path` | Path to visual features file (h5py) |
| `--annotation_folder` | Path to annotations |
| `--num_clusters` | Number of pseudo regions |

For example, to train the model, run the following command:
```bash
python train_transformer.py --exp_name S2 --batch_size 50 --m 40 --head 8 --features_path /path/to/features
```

or just run:
