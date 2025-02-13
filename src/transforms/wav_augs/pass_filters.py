import torch_audiomentations
from torch import Tensor, nn

class BandPassFilter(nn.Module):
    """
    Полосовой фильтр (Band-Pass), который пропускает только определенный диапазон частот.

    Args:
        *args, **kwargs: Аргументы, передаваемые в `torch_audiomentations.BandPassFilter`.
            - min_cutoff_freq (float): Минимальная частота среза (Гц).
            - max_cutoff_freq (float): Максимальная частота среза (Гц).
            - p (float): Вероятность применения фильтра.

    Attributes:
        band_pass_transform (torch_audiomentations.BandPassFilter): Фильтр, пропускающий сигнал в заданном диапазоне.

    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.band_pass_transform = torch_audiomentations.BandPassFilter(*args, **kwargs)

    def __call__(self, data: Tensor):
        """
        Применяет полосовой фильтр к аудиосигналу.

        Args:
            data (Tensor): Аудиосигнал (shape: [batch, time] или [time]).

        Returns:
            Tensor: Отфильтрованный аудиосигнал той же формы.
        """
        expanded_audio = data.unsqueeze(dim=1)
        filtered_audio = self.band_pass_transform(expanded_audio)
        return filtered_audio.squeeze(dim=1)


class BandStopFilter(nn.Module):
    """
    Полосозаграждающий (режекторный) фильтр (Band-Stop), который удаляет определенный диапазон частот.

    Args:
        *args, **kwargs: Аргументы, передаваемые в `torch_audiomentations.BandStopFilter`.
            - min_cutoff_freq (float): Нижняя граница удаляемого диапазона (Гц).
            - max_cutoff_freq (float): Верхняя граница удаляемого диапазона (Гц).
            - p (float): Вероятность применения фильтра.

    Attributes:
        band_stop_transform (torch_audiomentations.BandStopFilter): Фильтр, удаляющий заданный диапазон частот.

    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.band_stop_transform = torch_audiomentations.BandStopFilter(*args, **kwargs)

    def __call__(self, data: Tensor):
        """
        Применяет полосозаграждающий фильтр к аудиосигналу.

        Args:
            data (Tensor): Аудиосигнал (shape: [batch, time] или [time]).

        Returns:
            Tensor: Отфильтрованный аудиосигнал той же формы.
        """
        expanded_audio = data.unsqueeze(dim=1)
        filtered_audio = self.band_stop_transform(expanded_audio)
        return filtered_audio.squeeze(dim=1)