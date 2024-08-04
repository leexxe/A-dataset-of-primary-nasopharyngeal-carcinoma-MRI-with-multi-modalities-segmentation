import pydicom
import numpy as np
import cv2
from pathlib import Path
import nibabel as nib
import tifffile as tiff
import argparse


def load_dicom_image(path):
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array
    image = (image / np.max(image) * 255).astype(np.uint8)
    return image


def load_nifti_image(path):
    nifti = nib.load(path)
    image = nifti.get_fdata()
    return image


def preprocess_image(image, target_size):
    image_resized = cv2.resize(image, target_size)
    image_resized = np.expand_dims(image_resized, axis=-1)
    return image_resized


def save_tiff(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(path, image, photometric="minisblack")


def save_nifti_slices(nifti_path, save_dir, nii_idx, axis=0):
    volume = (load_nifti_image(nifti_path) * 255).astype(np.uint8)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    num_slices = volume.shape[axis]
    for i in range(num_slices):
        if axis == 0:
            slice_data = volume[i, :, :]
        elif axis == 1:
            slice_data = volume[:, i, :]
        else:
            slice_data = volume[:, :, i]

        slice_filename = f"{nii_idx}_{i:03d}_mask.tif"
        slice_path = Path(save_dir) / slice_filename

        save_tiff(slice_data, slice_path)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process and save medical images at specified resolution."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Resolution of the image after resizing",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    resolution = args.resolution

    base_dir = Path("data/MRI-Segments/")
    save_dir = Path("imgs")

    patient_ids = np.arange(1, 278)
    mri_idxs = ["T1WI", "T2WI", "CE-T1WI"]
    nii_idxs = ["T1", "T2", "CE-T1"]

    for patient_id in patient_ids:
        for t in range(len(mri_idxs)):
            mri_idx = mri_idxs[t]
            nii_idx = nii_idxs[t]
            file_folder = base_dir / f"{patient_id:03d}"
            mri_folder = file_folder / mri_idx
            file_names = sorted([file.name for file in mri_folder.iterdir()])

            for idx, file_name in enumerate(file_names):
                if file_name.endswith(".dcm"):
                    file_path = mri_folder / file_name
                    dicom_image = load_dicom_image(file_path)
                    preprocessed_dicom_image = preprocess_image(
                        dicom_image, (resolution, resolution)
                    )
                    dicom_save_path = (
                        save_dir / f"{patient_id:03d}" / f"{mri_idx}_{idx:03d}.tif"
                    )
                    save_tiff(preprocessed_dicom_image, dicom_save_path)

            nifti_path = file_folder / f"ROI-{nii_idx}.nii"
            save_nii_dir = save_dir / f"{patient_id:03d}"
            save_nifti_slices(nifti_path, save_nii_dir, mri_idx, axis=2)


if __name__ == "__main__":
    main()
