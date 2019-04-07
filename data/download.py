import requests
import tqdm
import os
import tarfile
import zipfile


def download(url, path=None):
    print("[+] Downloading", url)
    if not path:
        # if path is not specified
        # just use the filename from url
        path = os.path.basename(url)
    elif os.path.isdir(path):
        # if path is a directory
        # then just join that directory with
        # the filename from url
        path = os.path.join(path, os.path.basename(url))
    print("[*] Writing to", path)
    with open(path, "wb") as f:
        res = requests.get(url, stream=True)
        total_length = res.headers.get("content-length")

        if not total_length: # no content length header in the response
            # write immediately
            print("[!] Unknown file size")
            print("[*] Downloading ...")
            f.write(res.content)
        else:
            chunk_size = 1024 * 4
            total_length = int(total_length)
            bar = tqdm.tqdm(res.iter_content(chunk_size), "[*] Downloading",
                                                            total=total_length,
                                                            unit="B",
                                                            unit_scale=True,
                                                            unit_divisor=1024)
            for data in bar:
                length = f.write(data)
                bar.update(length)
            print(f"[+] Successfully downloaded {path}")
        

        if path.endswith(".tar.gz"):
            print("Extracting files...")
            tar = tarfile.open(path, mode="r:gz")
            extracted_path = os.path.split(path)[0]
            tar.extractall(extracted_path)
            tar.close()
        elif path.endswith(".tar"):
            print("Extracting files...")
            tar = tarfile.open(path, mode="r:")
            extracted_path = os.path.split(path)[0]
            tar.extractall(extracted_path)
            tar.close()
        elif path.endswith(".zip"):
            print("Extracting files...")
            zip_file = zipfile.ZipFile(path, "r")
            extracted_path = os.path.split(path)[0]
            zip_file.extractall(extracted_path)
            zip_file.close()



if __name__ == "__main__":
    URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    download(URL, "data")