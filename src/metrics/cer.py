from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer


class ArgmaxCERMetric(BaseMetric):
    """
    Метрика CER (Character Error Rate) с жадным (argmax) декодированием.

    Использует жадное (argmax) декодирование для восстановления текста из лог-вероятностей 
    модели CTC и вычисляет среднюю долю неверных символов по батчу.

    Args:
        text_encoder: Объект, реализующий методы `normalize_text` и `ctc_decode` 
                      для нормализации и декодирования текста.

    """
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        """
        Вычисляет CER (Character Error Rate) с жадным (argmax) декодированием.

        Args:
            log_probs (Tensor): Лог-вероятности размерностью (batch, time, vocab_size).
            log_probs_length (Tensor): Длины реальных последовательностей в батче.
            text (List[str]): Истинные текстовые расшифровки.

        Returns:
            float: Среднее значение CER по батчу.
        """
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)


class BeamSearchCERMetric(BaseMetric):
    """
    Метрика CER (Character Error Rate) с beam search-декодированием.

    Использует beam search для поиска наиболее вероятного текста из лог-вероятностей CTC. 
    Поддерживает использование языковой модели для более точного декодирования.

    Args:
        text_encoder: Объект, реализующий методы `normalize_text` и `ctc_beamsearch`.
        type (str): Тип beam search (например, "lm" для использования языковой модели).
        beam_size (int): Число гипотез в beam search (чем больше, тем точнее, но медленнее).

    """
    def __init__(self, text_encoder, type: str = "lm", beam_size=10, *args, **kwargs): # использует языковую модель в beam search.
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.type = type
        self.beam_size = beam_size # число лучших вариантов наиболее вероятных последовательностей

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        """
        Вычисляет CER (Character Error Rate) с beam search-декодированием.

        Args:
            log_probs (Tensor): Лог-вероятности размерностью (batch, time, vocab_size).
            log_probs_length (Tensor): Длины реальных последовательностей в батче.
            text (List[str]): Истинные текстовые расшифровки.

        Returns:
            float: Среднее значение CER по батчу.
        """
        cers = []
        probs = log_probs.cpu().detach().numpy()
        lengths = log_probs_length.detach().numpy()
        for prob, length, target_text in zip(probs, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_beamsearch(
                prob[:length], type=self.type, beam_size=self.beam_size
            )[0]["hypothesis"]
            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
