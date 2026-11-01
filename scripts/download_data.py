from calidad_aire.config import DATA_URL
from calidad_aire.data import download, sha256

if __name__ == "__main__":
    path = download()
    print("Fuente:", DATA_URL)
    print("ZIP SHA-256:", sha256(path.with_suffix(".zip")))
    print("CSV guardado en:", path)
    print("CSV SHA-256:", sha256(path))
