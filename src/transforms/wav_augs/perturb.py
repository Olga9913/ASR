from torch import Tensor, nn
from torchaudio.transforms import SpeedPerturbation


class SpeedPerturb(nn.Module):
    """
    Аугментация аудиосигнала путем изменения скорости воспроизведения (Speed Perturbation).

    Изменяет скорость аудиофайла на случайное значение в заданном диапазоне, 
    не изменяя его высоту (pitch). Это полезно для увеличения разнообразия данных 
    при обучении моделей ASR (Automatic Speech Recognition).

    Args:
        *args, **kwargs: Аргументы, передаваемые в `torchaudio.transforms.SpeedPerturbation`.
            - orig_freq (int): Исходная частота дискретизации.
            - speeds (list[float]): Список коэффициентов скорости (например, [0.9, 1.0, 1.1]).

    Attributes:
        speed_transform (SpeedPerturbation): Трансформация для изменения скорости.

    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.speed_transform = SpeedPerturbation(*args, **kwargs)

    def __call__(self, data: Tensor):
        """
        Применяет изменение скорости к аудиосигналу.

        Args:
            data (Tensor): Аудиосигнал (shape: [batch, time] или [time]).

        Returns:
            Tensor: Аудиосигнал с измененной скоростью той же формы.
        """
        expanded_audio = data.unsqueeze(dim=1)
        perturbed_audio = self.speed_transform(expanded_audio)[0]
        return perturbed_audio.squeeze(dim=1)
