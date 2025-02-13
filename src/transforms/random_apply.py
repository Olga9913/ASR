import random
from typing import Callable

from torch import Tensor, nn

class RandomApply(nn.Module):
    """
    Применяет заданную функцию-аугментацию с вероятностью p.

    Используется в аугментации данных, когда мы хотим случайно 
    применять определенное преобразование (например, шум, изменение 
    громкости, размытие и т. д.), но не в 100% случаев.

    Args:
        p (float): Вероятность (от 0 до 1) применения аугментации.
        augmentation (Callable): Функция, которая принимает тензор 
            и возвращает его же, но с аугментацией.
    
    Raises:
        ValueError: Если p не находится в пределах [0, 1].
    """
    def __init__(self, p: float, augmentation: Callable):
        super().__init__()
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")
        self.probability = p
        self.augmentation_function = augmentation

    def __call__(self, data: Tensor) -> Tensor:
        """
        Применяет аугментацию к данным с вероятностью p.

        Args:
            data (Tensor): Входные данные (обычно аудио- или изображение-тензор).

        Returns:
            Tensor: Аугментированные данные (если сработало) или исходные (если нет).
        """
        if random.uniform(0, 1) < self.probability:
            return self.augmentation_function(data)
        return data