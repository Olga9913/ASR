import torch
from torch import nn


class BatchNormT(nn.Module):
    """
    Слой пакетной нормализации (BatchNorm) для временных данных.

    Этот слой применяет пакетную нормализацию к данным, где временная ось (T) и оси признаков (F)
    меняются местами для корректной работы BatchNorm1d. После нормализации данные возвращаются
    в исходный формат.

    Аргументы:
        num_features (int): Количество признаков (F) во входных данных.
        eps (float, опционально): Малое значение для численной стабильности. По умолчанию 1e-05.
        momentum (float, опционально): Параметр для расчёта скользящего среднего и дисперсии. По умолчанию 0.1.
        affine (bool, опционально): Если True, добавляет обучаемые параметры масштаба и смещения. По умолчанию True.
        track_running_stats (bool, опционально): Если True, отслеживает скользящее среднее и дисперсию. По умолчанию True.
        device (torch.device, опционально): Устройство для вычислений (например, CPU или GPU).
        dtype (torch.dtype, опционально): Тип данных тензоров.
    """
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x, x_length):
        """
        Прямой проход через слой пакетной нормализации.

        Аргументы:
            x (Tensor): Входные данные размерности (B, T, F), где:
                - B — размер батча,
                - T — длина последовательности (временная ось),
                - F — количество признаков.
            x_length (Tensor): Длины последовательностей в батче.

        Возвращает:
            Tuple[Tensor, Tensor]: Кортеж из:
                - Нормализованные данные размерности (B, T, F).
                - Длины последовательностей (остаются неизменными).
        """
        x = self.bn(x.transpose(1, 2))
        return x.transpose(1, 2), x_length


class ConvBlock(nn.Module):
    def __init__(self, n_conv_layers: int = 2, n_features: int = 128):
        """
        Свёрточный блок для обработки входных данных.

        Этот блок состоит из одного или нескольких свёрточных слоёв, которые применяются к входным данным
        (например, спектрограммам) для извлечения признаков. После каждого свёрточного слоя применяются
        BatchNorm и функция активации Hardtanh.

        Аргументы:
            n_conv_layers (int, опционально): Количество свёрточных слоёв. Допустимые значения: 1, 2 или 3. По умолчанию 2.
            n_features (int, опционально): Количество входных признаков (например, частотных полос в спектрограмме). По умолчанию 128.
        """
        super().__init__()
        assert n_conv_layers in [1, 2, 3]
        self.scaling = 2
        self.out_features = 0
        self.conv_block = [
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(41, 11),
                stride=(2, 2),
                padding=(20, 5),
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        ]

        self.out_features = (n_features + 20 * 2 - 41) // 2 + 1

        if n_conv_layers == 2:
            self.conv_block.extend(
                [
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=32,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm2d(32),
                    nn.Hardtanh(0, 20, inplace=True),
                ]
            )
            self.out_features = (self.out_features + 10 * 2 - 21) // 2 + 1

        if n_conv_layers == 3:
            self.conv_block.extend(
                [
                    nn.Conv2d(
                        in_channels=32,
                        out_channels=96,
                        kernel_size=(21, 11),
                        stride=(2, 1),
                        padding=(10, 5),
                    ),
                    nn.BatchNorm2d(96),
                    nn.Hardtanh(0, 20, inplace=True),
                ]
            )
            self.out_features = (self.out_features + 20 * 2 - 41) // 2 + 1
        self.conv_block = nn.Sequential(*self.conv_block)

    def forward(self, x, x_length):
        """
        Прямой проход через свёрточный блок.

        Аргументы:
            x (Tensor): Входной тензор размерности (B, 1, T, F), где:
                - B — размер батча,
                - T — длина последовательности (временная ось),
                - F — количество частотных полос (признаков).
            x_length (Tensor): Длины входных последовательностей в батче.

        Возвращает:
            Tuple[Tensor, Tensor]: Кортеж из:
                - Выходной тензор размерности (B, C, T', F'), где:
                    - C — количество выходных каналов,
                    - T' — длина последовательности после свёртки,
                    - F' — количество выходных признаков.
                - Обновлённые длины последовательностей (T').
        """
        x = self.conv_block(x)
        return x, x_length // self.scaling


class GRUBlock(nn.Module):
    """
        Блок рекуррентной нейронной сети (GRU).

    Этот блок реализует слой GRU (Gated Recurrent Unit), который используется для обработки последовательностей.
    Поддерживает двунаправленный режим и упаковку/распаковку последовательностей переменной длины.

    Аргументы:
        input_size (int): Размер входного тензора (количество признаков на каждом временном шаге).
        hidden_size (int): Размер скрытого состояния GRU.
        bias (bool, опционально): Использовать ли смещение в GRU. По умолчанию True.
        batch_first (bool, опционально): Если True, входные данные имеют формат (B, T, F), где B — размер батча.
            По умолчанию True.
        dropout (float, опционально): Вероятность dropout. По умолчанию 0 (без dropout).
        bidirectional (bool, опционально): Если True, используется двунаправленный GRU. По умолчанию True.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=True,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x, x_length):
        """
        Прямой проход через GRU с упаковкой и распаковкой последовательностей.

        Аргументы:
            x (Tensor): Входные данные размерности (B, T, F), где:
                - B — размер батча,
                - T — длина последовательности,
                - F — количество признаков.
            x_length (Tensor): Длины последовательностей в батче (для упаковки).

        Возвращает:
            Tuple[Tensor, Tensor]: Кортеж из:
                - Обработанные данные размерности (B, T, H), где H — размер скрытого состояния.
                - Длины последовательностей (остаются неизменными).
        """
        x = nn.utils.rnn.pack_padded_sequence(
            x, x_length, batch_first=True, enforce_sorted=False
        )
        x, _ = self.gru(x, None)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        if self.gru.bidirectional:
            x = x[..., : self.gru.hidden_size] + x[..., self.gru.hidden_size :]
        return x, x_length


class DeepSpeech2(nn.Module):
    """
    Модель DeepSpeech2, основанная на архитектуре из статьи http://proceedings.mlr.press/v48/amodei16.pdf.
    Архитектура состоит из сверточных слоёв для извлечения признаков и рекуррентных слоёв (GRU) для моделирования временных зависимостей.
    """

    def __init__(
        self,
        n_features: int,
        n_tokens: int,
        n_conv_layers: int = 2,
        n_rnn_layers: int = 5,
        fc_hidden: int = 512,
        **batch,
    ):
        """
        Инициализация модели DeepSpeech2.

        Аргументы:
            n_features (int): Количество входных признаков (например, частотных полос в спектрограмме).
            n_tokens (int): Количество токенов в выходном слое (например, количество символов в алфавите).
            n_conv_layers (int, опционально): Количество свёрточных слоёв. По умолчанию 2.
            n_rnn_layers (int, опционально): Количество рекуррентных слоёв (GRU). По умолчанию 5.
            fc_hidden (int, опционально): Количество нейронов в полносвязном слое. По умолчанию 512.
            **batch: Дополнительные аргументы для пакетной обработки.
        """
        super().__init__()

        self.conv_block = ConvBlock(n_conv_layers=n_conv_layers, n_features=n_features)

        rnn_input_size = n_features // 2**n_conv_layers
        rnn_input_size *= 32 if n_conv_layers < 3 else 96

        rnn_output_size = fc_hidden * 2  # bidirectional
        self.gru_layers = [
            GRUBlock(input_size=rnn_input_size, hidden_size=rnn_output_size)
        ]

        for _ in range(n_rnn_layers - 1):
            self.gru_layers.extend(
                [
                    BatchNormT(rnn_output_size),
                    GRUBlock(input_size=rnn_output_size, hidden_size=rnn_output_size),
                ]
            )
        self.gru_layers = nn.Sequential(*self.gru_layers)
        self.batch_norm = nn.BatchNorm1d(rnn_output_size)
        self.fc = nn.Linear(rnn_output_size, n_tokens, bias=False)

    def forward(self, spectrogram, **batch):
        """
        Прямой проход модели DeepSpeech2.

        Аргументы:
            spectrogram (Tensor): Входная спектрограмма размерности (B, 1, T, F), где:
                - B — размер батча,
                - T — временная длина спектрограммы,
                - F — количество частотных полос.
            **batch: Дополнительные аргументы, включая длины спектрограмм.

        Возвращает:
            dict: Словарь с выходными логарифмическими вероятностями и их длинами:
                - "log_probs": Логарифмические вероятности для каждого токена.
                - "log_probs_length": Длины выходных последовательностей.
        """

        spectrogram_length = batch["spectrogram_length"]
        spectrogram = spectrogram.unsqueeze(dim=1)
        outputs, output_lengths = self.conv_block(spectrogram, spectrogram_length)

        B, C, F, T = outputs.shape
        outputs = outputs.view(B, C * F, T).transpose(1, 2)

        for gru_layer in self.gru_layers:
            outputs, output_lengths = gru_layer(outputs, output_lengths)

        outputs = self.batch_norm(outputs.transpose(1, 2)).transpose(1, 2)

        log_probs = nn.functional.log_softmax(self.fc(outputs), dim=-1)
        return {"log_probs": log_probs, "log_probs_length": output_lengths}

    def transform_input_lengths(self, input_lengths):
        """
        Преобразует длины входных последовательностей с учётом сжатия временной оси свёрточными слоями.

        Аргументы:
            input_lengths (Tensor): Исходные длины входных последовательностей.

        Возвращает:
            output_lengths (Tensor): Новые длины после сжатия свёрточными слоями.
        """
        return input_lengths // 2

    def __str__(self):
        """
        Возвращает строковое представление модели с информацией о количестве параметров.

        Возвращает:
            str: Строка с описанием модели, общим количеством параметров и количеством обучаемых параметров.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
