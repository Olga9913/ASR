import json
from argparse import ArgumentParser
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import CTCTextEncoder


def main(data_dir):
    """
    Вычисляет средние WER (Word Error Rate) и CER (Character Error Rate) по всем JSON-файлам в папке.

    Файлы должны содержать предсказанный текст и целевой текст в формате:
    {
        "prediction": "распознанный текст",
        "target": "эталонный текст"
    }

    Args:
        data_dir (str): Путь к директории с JSON-файлами предсказаний.

    Prints:
        Средние значения WER и CER по всем файлам.
    """
    dir = Path(data_dir)
    wer, cer = 0, 0
    n = 0
    for instance_path in list(dir.glob("**/*.json")):
        with open(instance_path, "r") as f:
            data = json.load(f)
            pred_text, target_text = data["prediction"], data["target"]
            target_text = CTCTextEncoder.normalize_text(target_text)
            wer += calc_wer(target_text, pred_text)
            cer += calc_cer(target_text, pred_text)
            n += 1
    wer = round(wer / n, 3)
    cer = round(cer / n, 3)
    print(f"WER = {wer}, CER = {cer}")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--data_dir", default="data/saved/inference", type=str)
    args = args.parse_args()
    main(args.data_dir) 
