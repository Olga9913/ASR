import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Класс Inferencer (аналог Trainer, но для инференса).

    Используется для обработки данных без необходимости использования оптимизаторов,
    логгеров и других компонентов, которые нужны для обучения. Основная задача — 
    выполнение инференса на данных, сохранение предсказаний и вычисление метрик.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Инициализация Inferencer.

        Аргументы:
            model (nn.Module): Модель PyTorch.
            config (DictConfig): Конфигурация запуска, содержащая настройки для инференса.
            device (str): Устройство для вычислений (например, "cpu" или "cuda").
            dataloaders (dict[DataLoader]): Словарь с DataLoader для разных наборов данных.
            text_encoder (CTCTextEncoder): Кодировщик текста для декодирования предсказаний.
            save_path (str): Путь для сохранения предсказаний и другой информации.
            metrics (dict, опционально): Словарь с метриками для инференса. Каждая метрика — 
                экземпляр src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None, опционально): Преобразования, которые 
                применяются ко всему батчу. Зависят от имени тензора.
            skip_model_load (bool, опционально): Если False, требуется указать путь к 
                предобученной модели. Если True, предполагается, что модель уже загружена.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Запуск инференса на всех наборах данных.

        Возвращает:
            part_logs (dict): Словарь с результатами для каждого набора данных.
                part_logs[part_name] содержит логи для набора данных part_name.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Обработка батча: выполнение инференса, вычисление метрик и сохранение предсказаний.

        Аргументы:
            batch_idx (int): Индекс текущего батча.
            batch (dict): Батч данных, полученный из DataLoader.
            metrics (MetricTracker): Объект для вычисления и агрегации метрик.
            part (str): Имя набора данных (например, "train", "test"). Используется для 
                создания папки для сохранения результатов.

        Возвращает:
            batch (dict): Батч данных, обновлённый с предсказаниями модели.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))


        batch_size = batch["log_probs"].shape[0]

        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            log_probs = batch["log_probs"][i].clone()
            log_probs_length = batch["log_probs_length"][i].clone()
            target_text = batch["text"][i]
            audio_path = batch["audio_path"][i]

            pred_inds = torch.argmax(log_probs.cpu(), dim=-1).numpy()
            length = log_probs_length.detach().numpy()
            target_text = self.text_encoder.normalize_text(target_text)
            prediction = self.text_encoder.ctc_decode(pred_inds[:length])


            output = {
                "prediction": prediction,
                "target": target_text,
            }

            if self.save_path is not None:
                # you can use safetensors or other lib here
                path = self.save_path / f"{str(Path(audio_path).stem)}.json"
                with open(path, "w") as f:
                    json.dump(output, f)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Запуск инференса на определённом наборе данных.

        Аргументы:
            part (str): Имя набора данных (например, "train", "test").
            dataloader (DataLoader): DataLoader для данного набора данных.

        Возвращает:
            logs (dict): Результаты метрик, вычисленных на наборе данных.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
