import h5py 
import numpy as np 
from pathlib import Path 
from tqdm import tqdm 

# 设置输入和输出路径
input_dir = "/home/sda1/cyz/SHENet-main/OpenAI-CLIP-Feature-main/clip_grid_448/"  # CLIP特征提取的输出目录
output_file = "./CLIP_features.hdf5"  # 输出的HDF5文件路径

def convert_clip_to_hdf5(input_dir, output_file): 
    """将CLIP网格特征转换为KMCN需要的HDF5格式"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    feature_files = sorted(Path(input_dir).glob('*.npz'))
    total_files = len(feature_files)
    
    print(f"Found {total_files} .npz files in {input_dir}")
    print(f"Converting to HDF5 format...")
    
    with h5py.File(output_file, 'w') as hf: 
        processed_files = 0
        for feature_file in tqdm(feature_files, desc="转换特征"): 
            npz_data = np.load(feature_file) 
            
            # 提取网格特征
            if 'features' in npz_data:
                features = npz_data['features'] 
            elif 'grid_feats' in npz_data:
                features = npz_data['grid_feats']
            else:
                print(f"Warning: No features found in {feature_file.name}")
                continue
            
            if features.shape[0] != 49:
                print(f"Warning: Unexpected feature shape {features.shape} in {feature_file.name}")
                continue
            
            # 从文件名提取图像ID 
            filename = feature_file.stem 
            if 'COCO_' in filename: 
                img_id = int(filename.split('_')[-1]) 
            else: 
                img_id = int(filename) 
            
            # 保存网格特征
            hf.create_dataset(f'{img_id}_features', data=features) 
            
            # 提取全局特征
            if 'g_feature' in npz_data:
                global_features = npz_data['g_feature']
            elif 'global_feats' in npz_data:
                global_features = npz_data['global_feats']
            elif 'global_features' in npz_data:
                global_features = npz_data['global_features']
            elif 'feat' in npz_data:
                global_features = npz_data['feat']
            else:
                global_features = None
            
            if global_features is not None:
                if global_features.shape[-1] == 512:
                    hf.create_dataset(f'{img_id}_global', data=global_features)
                else:
                    print(f"Warning: Unexpected global feature shape {global_features.shape} in {feature_file.name}")
            
            processed_files += 1
            if processed_files % 10000 == 0:
                print(f"Processed {processed_files}/{total_files} files ({processed_files/total_files*100:.1f}%)")
    
    print(f"转换完成！总共处理了 {processed_files} 个文件")
    print(f"HDF5 file saved to {output_file}")

if __name__ == '__main__': 
    convert_clip_to_hdf5(input_dir, output_file)
