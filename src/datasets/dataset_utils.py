import os.path
import shutil
import wget

from ood_detection.config import Config


def extract_all(archive, extract_path):
    shutil.unpack_archive(archive, extract_path)


def download_file(url, dirname):
    filename = wget.download(url, out=dirname)

    return filename


def download_and_extract(url, dirname):
    filename = download_file(url, dirname)
    file_path = os.path.join(dirname, filename)
    extract_all(file_path, dirname)
