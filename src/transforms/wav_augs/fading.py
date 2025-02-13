import torchaudio
from torch import Tensor, nn


class Fading(nn.Module):
    """
    Аугментация аудиосигнала с применением эффектов fade-in (плавное увеличение громкости) 
    и fade-out (плавное уменьшение громкости).

    Args:
        fade_ratio (float, optional): Доля сигнала, на которую приходится fade-in и fade-out.
            Значение должно быть от 0 до 1. По умолчанию 0.5.
        *args, **kwargs: Дополнительные аргументы (не используются, но позволяют гибкость).

    Attributes:
        fade_ratio (float): Соотношение длины fade-in и fade-out к общему сигналу.
    """
    def __init__(self, fade_ratio=0.5, *args, **kwargs):
        super().__init__()
        self.fade_ratio = fade_ratio

    def forward(self, audio: Tensor):
        """
        Применяет fade-in и fade-out к входному аудиосигналу.

        Args:
            audio (Tensor): Аудиосигнал (shape: [batch, time] или [time]).

        Returns:
            Tensor: Обработанный аудиосигнал с эффектами fade-in и fade-out 
            (shape: такой же, как у входного).
        """
        fade_length = int(audio.size(-1) * self.fade_ratio)
        fade_transform = torchaudio.transforms.Fade(fade_in_len=fade_length, fade_out_len=fade_length)
        audio_expanded = audio.unsqueeze(dim=1)
        faded_audio = fade_transform(audio_expanded)
        return faded_audio.squeeze(dim=1)
