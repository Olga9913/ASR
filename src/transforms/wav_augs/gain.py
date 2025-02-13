import torch_audiomentations
from torch import Tensor, nn

class Gain(nn.Module):
    """
    Аугментация изменения громкости аудиосигнала с использованием
    `torch_audiomentations.Gain`.

    Увеличивает или уменьшает громкость входного аудиосигнала случайным образом 
    в заданном диапазоне. Может применяться к батчу аудиоданных.

    Args:
        *args: Позиционные аргументы, передаваемые в `torch_audiomentations.Gain`.
        **kwargs: Именованные аргументы, передаваемые в `torch_audiomentations.Gain`.

    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.gain_transform = torch_audiomentations.Gain(*args, **kwargs)

    def forward(self, audio: Tensor):
        """
        Применяет аугментацию изменения громкости.

        Args:
            data (Tensor): Аудиотензор формы [batch, time].

        Returns:
            Tensor: Аудиотензор той же формы, но с измененной громкостью.
        """
        expanded_audio = audio.unsqueeze(dim=1)
        processed_audio = self.gain_transform(expanded_audio)
        return processed_audio.squeeze(dim=1)

