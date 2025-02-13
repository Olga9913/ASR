from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Класс Trainer. Определяет логику обработки батчей, вычисления метрик и обучения модели
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Обрабатывает батч: выполняет прямой проход через модель, вычисляет метрики и потери,
        а также выполняет шаг обучения (если это этап обучения).

        Функция ожидает, что критерий (criterion) объединяет все потери (если их несколько)
        в одну, которая хранится в ключе 'loss'.

        Аргументы:
            batch (dict): Батч данных, полученный из DataLoader.
            metrics (MetricTracker): Объект для вычисления и агрегации метрик.
                Метрики зависят от типа раздела данных (train или inference).

        Возвращает:
            batch (dict): Батч данных, обновлённый с выходами модели и значениями потерь.
        """
        torch.cuda.empty_cache()
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Логирование данных из батча. Использует self.writer.add_* для записи данных
        в систему отслеживания экспериментов.

        Аргументы:
            batch_idx (int): Индекс текущего батча.
            batch (dict): Батч данных после обработки в функции process_batch.
            mode (str): Режим (train или inference). Определяет, какие данные логировать.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        """
        Логирование спектрограммы.

        Аргументы:
            spectrogram (Tensor): Спектрограмма.
            **batch: Дополнительные данные из батча.
        """
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_audio(self, audio, **batch):
        """
        Логирование аудио.

        Аргументы:
            audio (Tensor): Аудиоданные.
            **batch: Дополнительные данные из батча.
        """
        self.writer.add_audio("audio", audio[0], 16000)
    
    def log_predictions(
    self, text, log_probs, log_probs_length, audio_path, examples_to_log=10, **batch
):
        """
        Логирование предсказаний модели.

        Аргументы:
            text (list): Целевые тексты.
            log_probs (Tensor): Логарифмические вероятности, предсказанные моделью.
            log_probs_length (Tensor): Длины последовательностей вероятностей.
            audio_path (list): Пути к аудиофайлам.
            examples_to_log (int, опционально): Количество примеров для логирования. По умолчанию 10.
            **batch: Дополнительные данные из батча.
        """
        metrics_to_log = self.evaluation_metrics.keys()
        
        results = {}
        results["Target"] = [
            self.text_encoder.normalize_text(target) for target in text[:examples_to_log]
        ]

        truncated_probs = [
            prob[:length] for prob, length in zip(log_probs, log_probs_length.numpy())
        ]

        if "CER_(LM-BeamSearch)" in metrics_to_log:
            lm_predictions = [
                self.text_encoder.ctc_beamsearch(prob, type="lm")[0]["hypothesis"]
                for prob in truncated_probs[:examples_to_log]
            ]
            results["BS_LM_predictions"] = lm_predictions

            lm_metrics = np.array([
                [calc_cer(target, pred) * 100, calc_wer(target, pred) * 100]
                for target, pred in zip(results["Target"], lm_predictions)
            ])
            results["BS_LM_CER"] = lm_metrics[:, 0]
            results["BS_LM_WER"] = lm_metrics[:, 1]

        if "CER_(BeamSearch)" in metrics_to_log:
            nolm_predictions = [
                self.text_encoder.ctc_beamsearch(prob, type="nolm")[0]["hypothesis"]
                for prob in truncated_probs[:examples_to_log]
            ]
            results["BS_predictions"] = nolm_predictions

            nolm_metrics = np.array([
                [calc_cer(target, pred) * 100, calc_wer(target, pred) * 100]
                for target, pred in zip(results["Target"], nolm_predictions)
            ])
            results["BS_CER"] = nolm_metrics[:, 0]
            results["BS_WER"] = nolm_metrics[:, 1]

        if "CER_(BeamSearch-just_test)" in metrics_to_log:
            test_predictions = [
                self.text_encoder.ctc_beamsearch(prob, type="just_test")[0]["hypothesis"]
                for prob in truncated_probs[:examples_to_log]
            ]
            results["BS_just_test_predictions"] = test_predictions

            test_metrics = np.array([
                [calc_cer(target, pred) * 100, calc_wer(target, pred) * 100]
                for target, pred in zip(results["Target"], test_predictions)
            ])
            results["BS_CER"] = test_metrics[:, 0]
            results["BS_WER"] = test_metrics[:, 1]

        if "CER_(Argmax)" in metrics_to_log:
            argmax_indices = log_probs.cpu().argmax(-1).numpy()
            trimmed_indices = [
                inds[: int(length)]
                for inds, length in zip(argmax_indices, log_probs_length.numpy())
            ]
            raw_texts = [
                self.text_encoder.decode(inds) for inds in trimmed_indices[:examples_to_log]
            ]
            decoded_texts = [
                self.text_encoder.ctc_decode(inds) for inds in trimmed_indices[:examples_to_log]
            ]

            results["Argmax_predicitons_raw"] = raw_texts
            results["Argmax_predicitons"] = decoded_texts

            argmax_metrics = np.array([
                [calc_cer(target, pred) * 100, calc_wer(target, pred) * 100]
                for target, pred in zip(results["Target"], decoded_texts)
            ])
            results["Argmax_CER"] = argmax_metrics[:, 0]
            results["Argmax_WER"] = argmax_metrics[:, 1]

        predictions_df = pd.DataFrame.from_dict(results)
        predictions_df.index = [Path(path).name for path in audio_path[:examples_to_log]]

        self.writer.add_table("predictions", predictions_df)
