import os
import tarfile

directory = "data/raw/images"

for filename in os.listdir(directory):
    if filename.endswith(".tar.gz"):
        file_path = os.path.join(directory, filename)
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=directory)
