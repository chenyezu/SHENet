import torch
import pickle
import json
import h5py
from models.transformer import (
    Transformer, TransformerEncoder, TransformerDecoder, ScaledDotProductAttention, Projector
)
from data import TextField

import numpy as np

IMG_IDS = [1554, 999, 2473, 72, 36]
device = torch.device('cuda:0')

# 加载 GT captions
gt_captions = {}
for split in ['captions_train2014.json', 'captions_val2014.json']:
    data = json.load(open(f'./annotations/{split}', 'r'))
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_captions:
            gt_captions[img_id] = []
        gt_captions[img_id].append(ann['caption'])

text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True,
                       tokenize='spacy', remove_punctuation=True, nopoints=False)
text_field.vocab = pickle.load(open('vocab/vocab_coco.pkl', 'rb'))

obj_file = h5py.File('./data/features/vinvl.hdf5', 'r')
grid_file = h5py.File('./data/features/CLIP_features.hdf5', 'r')

encoder = TransformerEncoder(3, 0, attention_module=ScaledDotProductAttention,
                             use_osce=True)
decoder = TransformerDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
projector = Projector(f_obj=2048, f_grid=2048, f_out=512, drop_rate=0.1)
model = Transformer(bos_idx=text_field.vocab.stoi['<bos>'],
                    encoder=encoder, decoder=decoder, projector=projector).to(device)

ckpt = torch.load('outputs/[m2][xmodal-ctx]/ckpt_best.pth', map_location=device)
model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model'].items()},
                       strict=False)
model.eval()

for img_id in IMG_IDS:
    obj_raw = obj_file[str(img_id)][:]
    n = obj_raw.shape[0]
    if n < 50:
        obj_raw = np.concatenate([obj_raw, np.zeros((50 - n, obj_raw.shape[1]), dtype=obj_raw.dtype)], axis=0)
    elif n > 50:
        obj_raw = obj_raw[:50]
    obj_feat = torch.from_numpy(obj_raw).unsqueeze(0).to(device)

    grid_raw = grid_file['%d_features' % img_id][:]
    if grid_raw.shape[0] != 49:
        if grid_raw.shape[0] < 49:
            grid_raw = np.concatenate([grid_raw, np.zeros((49 - grid_raw.shape[0], 2048), dtype=np.float32)], axis=0)
        else:
            grid_raw = grid_raw[:49]
    grid_feat = torch.from_numpy(grid_raw).unsqueeze(0).to(device)

    with torch.no_grad():
        out, _ = model(obj=obj_feat, grid=grid_feat, max_len=20, mode="rl",
                       eos_idx=text_field.vocab.stoi['<eos>'], beam_size=5, out_size=1)

    caps = text_field.decode(out)
    gts = gt_captions.get(img_id, ['(no GT found)'])
    print(f"img_id {img_id}:")
    for gt in gts:
        print(f"  GT: {gt}")
    print(f"  GEN: {caps[0]}")
    print()

obj_file.close()
grid_file.close()
