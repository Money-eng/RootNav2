import xml.etree.ElementTree as ET
from pathlib import Path
import tqdm

def remove_rootnavspline_from_rsml(directory_path):
    root_dir = Path(directory_path)
    
    rsml_files = list(root_dir.rglob("*.rsml"))
    
    print(f"🔍 {len(rsml_files)} found rsml files")
    cleaned_count = 0
    
    for file_path in tqdm.tqdm(rsml_files, desc="Cleaning RSML files"):
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            modified = False
            
            for parent in root.iter():
                to_remove = []
                for child in parent:
                    tag_name = child.tag.split('}')[-1].lower()
                    
                    if tag_name == 'rootnavspline' or tag_name == 'rootnav-spline':
                        to_remove.append(child)
                
                for child in to_remove:
                    parent.remove(child)
                    modified = True
            
            if modified:
                tree.write(file_path, encoding="utf-8", xml_declaration=True)
                cleaned_count += 1
                
        except ET.ParseError:
            print(f"❌ Error parsing {file_path}. Skipping.")
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")

dossier_a_nettoyer = "/home/loai/Documents/code/RSMLExtraction/temp/rn2_rec"

remove_rootnavspline_from_rsml(dossier_a_nettoyer)