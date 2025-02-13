import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    
    batched_data = {} # Создаем контейнер для батчей и определяем список полей (например, "spectrogram", "text_encoded", "audio_path")
    item_keys = dataset_items[0].keys()

    for key in item_keys: # Обрабатываем каждое поле по очереди
        if key in {"spectrogram", "text_encoded"}: # Обрабатываем тензоры переменной длины (спектрограммы и закодированный текст). Они требуют паддинга, чтобы иметь одинаковый размер
            lengths = [entry[key].shape[-1] for entry in dataset_items]
            batched_data[f"{key}_length"] = torch.tensor(lengths) # Создаем тензор с длинами последовательностей

            padded_data = pad_sequence( # Паддинг последовательностей до одной длины
                [entry[key].squeeze(0).T for entry in dataset_items],
                batch_first=True
            )
            batched_data[key] = padded_data # Сохраняем паддинговые данные в словарь батчей
        else:
            batched_data[key] = [entry[key] for entry in dataset_items]

    if "spectrogram" in batched_data: # Меняем оси у спектрограммы. Нам нужен формат [batch_size, features, time] (принятый в нейросетях), поэтому транспонируем оси.
        batched_data["spectrogram"] = batched_data["spectrogram"].permute(0, 2, 1)

    return batched_data

"""
модель должна обрабатывать данные, не смотря на разную длину аудиофайлов

1) Объединяет примеры в один батч.
2) Паддит спектрограммы и текстовые последовательности.
2) Сохраняет длины последовательностей для маскирования.
2) Форматирует спектрограммы для корректной подачи в модель.
"""