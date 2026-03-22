import argparse
import io
import json
import logging
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from dataset import load_image_samples

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_hdf5_dataset(
    csv_path: str,
    images_root: str,
    output_hdf5: str,
    resize: int = 0,
):
    output_path = Path(output_hdf5)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_image_samples(csv_path, images_root)
    logging.info(f"Adding {len(samples)} images to HDF5 database: {output_hdf5}")

    with h5py.File(output_path, "w") as h5f:
        images_grp = h5f.create_group("images")
        
        # Keep track of paths and labels mapping
        metadata = []
        for img_path, rcs_val in tqdm(samples, desc="Writing to HDF5"):
            rel_path = img_path.relative_to(Path(images_root)).as_posix()
            
            try:
                if resize > 0:
                    # Parse image, resize, save as raw JPEG bytes
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        img = img.resize((resize, resize), Image.Resampling.BILINEAR)
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=95)
                        binary_data = buffer.getvalue()
                else:
                    # Just read raw binary
                    with open(img_path, "rb") as f:
                        binary_data = f.read()

                # Store as variable-length string / void type
                images_grp.create_dataset(rel_path, data=np.void(binary_data))
                metadata.append({"path": rel_path, "rcs": rcs_val})
            except Exception as e:
                logging.error(f"Failed to process {img_path}: {e}")

        # Save metadata directly into HDF5
        meta_json = json.dumps(metadata)
        h5f.attrs["metadata"] = meta_json

    logging.info(f"Successfully created HDF5 database at {output_hdf5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to HDF5 single-file format")
    parser.add_argument("--csv_path", type=str, default="data/aircraft_rcs.csv")
    parser.add_argument("--images_root", type=str, default="data/images")
    parser.add_argument("--output_hdf5", type=str, default="data/aircraft_dataset.h5")
    parser.add_argument("--resize", type=int, default=0, help="Resize images before saving (0 to keep original)")
    
    args = parser.parse_args()
    create_hdf5_dataset(args.csv_path, args.images_root, args.output_hdf5, args.resize)
