import re
from pathlib import Path

with open("src/dataset.py", "r") as f:
    text = f.read()

target = """def load_image_samples(csv_path: str, images_root: str = "data/images") -> List[ImageSample]:
    csv_file = Path(csv_path)
    root = Path(images_root)
    df = pd.read_csv(csv_file)

    required_columns = {"image_folder", "rcs"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_file}: {missing_text}")

    samples: List[ImageSample] = []
    for row in df.itertuples(index=False):
        folder = resolve_image_folder(str(row.image_folder), root)
        if not folder.exists() or not folder.is_dir():
            continue
        rcs_value = float(row.rcs)
        image_files = [p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS]
        for image_file in sorted(image_files):
            samples.append((image_file, rcs_value))

    if not samples:
        raise ValueError("No image samples found. Check CSV paths and image directories.")

    return samples"""

replacement = """def load_image_samples(csv_path: str, images_root: str = "data/images", hdf5_path: str = None) -> List[ImageSample]:
    csv_file = Path(csv_path)
    root = Path(images_root)
    df = pd.read_csv(csv_file)

    required_columns = {"image_folder", "rcs"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_file}: {missing_text}")

    samples: List[ImageSample] = []
    
    # Check if we should read from HDF5 structure instead of disk
    if hdf5_path and Path(hdf5_path).exists():
        import h5py
        with h5py.File(hdf5_path, 'r') as f:
            for row in df.itertuples(index=False):
                folder_name = Path(str(row.image_folder)).name
                rcs_value = float(row.rcs)
                if 'images' in f and folder_name in f['images']:
                    for image_name in f['images'][folder_name].keys():
                        # We use a dummy path, the dataset __getitem__ will extract relative name
                        dummy_path = root / folder_name / image_name
                        samples.append((dummy_path, rcs_value))
    else:
        for row in df.itertuples(index=False):
            folder = resolve_image_folder(str(row.image_folder), root)
            if not folder.exists() or not folder.is_dir():
                continue
            rcs_value = float(row.rcs)
            image_files = [p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS]
            for image_file in sorted(image_files):
                samples.append((image_file, rcs_value))

    if not samples:
        raise ValueError(f"No image samples found. Checked HDF5 ({hdf5_path}) and disk paths.")

    return samples"""

with open("src/dataset.py", "w") as f:
    f.write(text.replace(target, replacement))

print("Done")
