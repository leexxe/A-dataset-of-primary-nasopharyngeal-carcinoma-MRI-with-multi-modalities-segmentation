# NPC-MRISegmentationToolkit

This toolkit accompanies the data analysis presented in the paper titled "A dataset of primary nasopharyngeal carcinoma MRI with multi-modalities segmentation."

## Getting Started

### Prerequisites

Before you begin, ensure you have Python installed on your system. You will also need `pip` for installing the required packages.

### Installation

To install the necessary packages, run the following command:

```bash
pip install -r requirements.txt
```

### Dataset

The dataset used in this study is available at [Zenodo](https://zenodo.org/records/10900202). To download the dataset, execute the script provided:

```bash
python download_dataset.py
```
## Usage

### Computing Morphological Parameters and Post-processing

To compute the morphological parameters and de-identification for a specific patient and ROI sequence, use the following command:

```bash
python morphological_parameters.py <patient_id> <roi_sequence>
```

For example:

```bash
python morphological_parameters.py 1 "ROI-T1"
```
## Citing This Work

If you find our scripts or dataset useful in your research, please consider citing our work:

```bibtex
@misc{li2024dataset,
      title={A dataset of primary nasopharyngeal carcinoma MRI with multi-modalities segmentation},
      author={Yin Li and Qi Chen and Kai Wang and Meige Li and Liping Si and Yingwei Guo and Yu Xiong and Qixing Wang and Yang Qin and Ling Xu and Patrick van der Smagt and Jun Tang and Nutan Chen},
      year={2024},
      eprint={2404.03253},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
