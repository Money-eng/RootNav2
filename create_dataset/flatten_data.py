import os
import shutil
from pathlib import Path

def flatten_dataset(source_dir, target_dir):
    source_path = Path(source_dir).resolve()
    target_path = Path(target_dir).resolve()
    
    splits = ['train', 'val', 'test']

    for split in splits:
        split_src = source_path / split
        if not split_src.exists():
            print(f"Info: Split '{split}' between {source_path} does not exist. Skipping.")
            continue
            
        split_dst = target_path / split
        split_dst.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTraitement du split : {split}")
        
        processed_count = 0
        
        for root, dirs, files in os.walk(split_src):
            root_path = Path(root)
            
            img_file = None
            for f in files:
                if f.lower() == "image.tif":
                    img_file = f
                    break
            
            rsml_file = "graph_projected.rsml"
            mask_file = "mask_projected.tif" 
            
            if img_file and (rsml_file in files) and (mask_file in files):
                time_name = root_path.name 
                series_name = root_path.parent.name
                
                unique_id = f"{series_name}_{time_name}"
                
                src_img = root_path / img_file
                src_mask = root_path / mask_file
                src_rsml = root_path / rsml_file
                
                dst_img = split_dst / f"{unique_id}.tif"
                dst_rsml = split_dst / f"{unique_id}.rsml"
                dst_mask = split_dst / f"{unique_id}_mask.tif"
                
                try:
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_rsml, dst_rsml)
                    shutil.copy2(src_mask, dst_mask)
                        
                    processed_count += 1
                except Exception as e:
                    print(f"Erreur sur {unique_id}: {e}")

        print(f" -> {processed_count} time serie images processed in split '{split}'.")

if __name__ == "__main__":
    input =  "" # path
    output = "" # path
    flatten_dataset(input, output)