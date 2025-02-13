import torch_audiomentations
from torch import Tensor, nn

class AddColoredNoise(nn.Module):
    """
    Аугментация аудиосигнала путем добавления шума определенного спектрального типа
    (например, белого, розового, коричневого и т. д.).

    Args:
        *args, **kwargs: Аргументы, передаваемые в `torch_audiomentations.AddColoredNoise`.
            Среди возможных параметров: 
            - min_snr_db (float): Минимальное отношение сигнал/шум (SNR) в дБ.
            - max_snr_db (float): Максимальное отношение сигнал/шум (SNR) в дБ.
            - noise_type (str): Тип шума ('white', 'pink', 'brownian' и т. д.).

    Attributes:
        noise_transform (torch_audiomentations.AddColoredNoise): Объект, выполняющий добавление шума.

    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.noise_transform = torch_audiomentations.AddColoredNoise(*args, **kwargs)

    def forward(self, audio: Tensor):
        """
        Добавляет шум к аудиосигналу.

        Args:
            audio (Tensor): Аудиосигнал (shape: [batch, time] или [time]).

        Returns:
            Tensor: Аудиосигнал с добавленным шумом (той же формы, что и входной).
        """
        expanded_audio = audio.unsqueeze(dim=1)
        noisy_audio = self.noise_transform(expanded_audio)
        return noisy_audio.squeeze(dim=1)
