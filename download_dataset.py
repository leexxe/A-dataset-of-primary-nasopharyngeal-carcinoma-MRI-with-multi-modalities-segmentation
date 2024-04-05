import requests


def download_file_from_zenodo(file_url, save_path):
    """
    Download a file from Zenodo.

    Parameters:
    - file_url: The URL of the file to download.
    - save_path: The local path where the file should be saved.
    """
    response = requests.get(file_url, stream=True)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")


file_url = "https://zenodo.org/records/10900202/files/data.zip?download=1"  # Replace this with the actual file URL
save_path = "data.zip"
download_file_from_zenodo(file_url, save_path)
