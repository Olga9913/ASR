import gzip
import os
import pathlib
import shutil

import gdown
import wget

from src.utils.io_utils import ROOT_PATH

URL_LINKS = {
    "simple_bpe": "11cBBNNy1k10TEYGHf3AXoCZBcToS3sgY",
    "bpe": "1ZoTP6UIJIi7nnWiI-AKu0et7SfPS7-Na",
    "libri": "https://openslr.elda.org/resources/11/librispeech-vocab.txt",
    "lm": "https://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz",
    "best_model": "1YBZZfyJf9S8dwhT2XBiIZR6LM6CFbWnU", 
}


def download_vocab(vocab_type):
    vocab_dir = ROOT_PATH / "data" / "libri_lm"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = vocab_dir / f"{vocab_type}_vocab.txt"

    print("Downloading vocabulary...")
    if vocab_type == "libri":
        wget.download(URL_LINKS[vocab_type], str(vocab_path))
    else:
        gdown.download(id=URL_LINKS[vocab_type], output=str(vocab_path))
    print("\nVocabulary successfully downloaded!")
    return str(vocab_path)


def download_lm():
    lm_dir = ROOT_PATH / "data/libri_lm"
    gz_file = lm_dir / "uppercase_3e-7.arpa.gz"
    uppercase_file = lm_dir / "uppercase_3e-7.arpa"
    lowercase_file = lm_dir / "lowercase_3e-7.arpa"

    if lowercase_file.exists():
        return str(lowercase_file)
    
    lm_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading language model...")
    wget.download(URL_LINKS["lm"], str(gz_file))
    with gzip.open(str(gz_file), "rb") as compressed_file:
        with open(str(uppercase_file), "wb") as uncompressed_file:
            shutil.copyfileobj(compressed_file, uncompressed_file)

    with open(str(uppercase_file), "r") as upper_file:
        with open(str(lowercase_file), "w") as lower_file:
            for line in upper_file:
                lower_file.write(line.lower())

    print("\nLanguage model successfully downloaded!")
    return str(lowercase_file)


def download_best_model(path=None):
    if path is None:
        models_dir = ROOT_PATH / "data" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "best_model.pth"
    else:
        target_dir = os.path.dirname(path)
        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)
        model_path = path

    print("Downloading the best model...")
    gdown.download(id=URL_LINKS["best_model"], output=str(model_path))
    print("\nBest model successfully downloaded!")
    return str(model_path)

