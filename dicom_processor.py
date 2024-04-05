import pydicom
import argparse
from pathlib import Path
from typing import List


def load_dicom_files(folder_path: Path) -> List[Path]:
    """
    Load DICOM files from a given folder.

    Parameters
    ----------
    folder_path : Path
        The path to the folder containing DICOM files.

    Returns
    -------
    List[Path]
        A list of paths to the DICOM files.
    """
    return list(folder_path.rglob("*.dcm"))


def anonymize_dicom(dicom: pydicom.dataset.FileDataset, patient_id: int) -> None:
    """
    Anonymize sensitive patient information in a DICOM file.

    Parameters
    ----------
    dicom : pydicom.dataset.FileDataset
        The DICOM file dataset to be anonymized.
    patient_id : int
        The patient ID to use for anonymization.

    Returns
    -------
    None
    """
    attributes_to_anonymize = [
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "PatientAge",
        "InstitutionName",
        "InstitutionAddress",
        "ReferringPhysicianName",
        "PerformingPhysicianName",
        "OperatorsName",
        "PhysiciansOfRecord",
        "PatientWeight",
        "UsedPatientWeight",
        "ImageText",
        "StudyDate",
        "SeriesDate",
        "AcquisitionDate",
        "ContentDate",
    ]
    text_tags = [
        (0x0051, 0x1010),  # ImageText
        (0x0011, 0x1110),  # Registration Date
        (0x0011, 0x1123),  # Used Patient Weight
        # Add other tags as needed
    ]
    for attr in attributes_to_anonymize:
        if attr in dicom:
            dicom.data_element(attr).value = (
                f"{patient_id:03d}" if attr == "PatientName" else ""
            )

    for tag in text_tags:
        if tag in dicom:
            dicom[tag].value = ""


def extract_dicom_metadata(dicom: pydicom.dataset.FileDataset) -> dict:
    """
    Extract and return metadata from a DICOM file.

    Parameters
    ----------
    dicom : pydicom.dataset.FileDataset
        The DICOM file from which to extract metadata.

    Returns
    -------
    dict
        A dictionary containing extracted metadata.
    """
    metadata = {
        "echo_time": float(dicom.EchoTime) if "EchoTime" in dicom else None,
        "repetition_time": float(dicom.RepetitionTime)
        if "RepetitionTime" in dicom
        else None,
        "pixel_spacing": float(dicom.PixelSpacing[0])
        if "PixelSpacing" in dicom
        else None,
        "spacing_between_slices": float(dicom.SpacingBetweenSlices)
        if "SpacingBetweenSlices" in dicom
        else None,
        "slice_thickness": float(dicom.SliceThickness)
        if "SliceThickness" in dicom
        else None,
        "scanner_device_name": dicom.ManufacturerModelName
        if "ManufacturerModelName" in dicom
        else "Unknown",
    }
    return metadata


def process_mri(
    patient_id: int, mri_sequence: str, base_folder: str = "data/MRI-Segments"
) -> None:
    """
    Process MRI DICOM files for a specific patient and MRI index.

    Parameters
    ----------
    patient_id : int
        The ID of the patient.
    mri_sequence : str
        The index of the MRI sequence.
    base_folder : str, optional
        The base folder where MRI DICOM files are
        located. Default is 'data/MRI-Segments'.

    Returns
    -------
    None
    """

    file_folder = Path(base_folder) / f"{patient_id:03d}" / mri_sequence
    dicom_files = load_dicom_files(file_folder)

    for t, dicom_file in enumerate(dicom_files):
        dicom = pydicom.dcmread(dicom_file)
        anonymize_dicom(dicom, patient_id)
        metadata = extract_dicom_metadata(dicom)
        dicom.save_as(dicom_file)
        if t == 0:
            print(metadata)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process MRI DICOM files and anonymize patient information."
    )
    parser.add_argument("patient_id", type=int, help="The ID of the patient.")
    parser.add_argument(
        "mri_sequence",
        type=str,
        choices=["T1WI", "T2WI", "CE-T1WI"],
        help="The name of the MRI sequence to process.",
    )

    args = parser.parse_args()

    process_mri(args.patient_id, args.mri_sequence)


if __name__ == "__main__":
    main()
