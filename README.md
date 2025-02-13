# Automatic Speech Recognition. Implement and train a neural-network speech recognition system with a CTC loss.

### Установка проекта

Следуйте этим шагам для установки проекта:

1. (Опционально) Создайте и активируйте виртуальное окружение с помощью venv:
   
   ```bash
   python3 -m venv project_env
   source project_env/bin/activate
   ```
   
2. Установите зависимости:
   
   ```bash
   pip install -r requirements.txt
   ```
   
3. Установите pre-commit:
   
   ```bash
   pre-commit install
   ```

### Как запустить обучение и инференс

Для обучения модели запустите:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Где `CONFIG_NAME` – это название конфигурационного файла из `src/configs`, а `HYDRA_CONFIG_ARGUMENTS` – дополнительные аргументы.

Для запуска инференса выполните:

```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```

Перед запуском необходимо указать API-ключ для `W&B`. Сделать это можно двумя способами:

1. Через терминал:

   ```bash
   export WANDB_API_KEY=ХХХХХХХ
   echo $WANDB_API_KEY
   ```
В файле wandb.yaml пропишите:

   ```bash
   api_key: ${oc.env:WANDB_API_KEY}

   ```
2. Указать ключ напрямую в файле wandb.yaml:

   ```bash
   api_key: "ХХХХХХХ"
   ```

## Воспроизведение модели

Для обучения модели использовались два конфигурационных файла:

- `train_argmax.yaml`: обучение модели в течение **50 эпох**.
- `train.yaml`: дополнительное обучение модели в течение **100 эпох**.

### Логи обучения в WanDB

train_argmax.yaml [по ссылке.](https://wandb.ai/helgahelga-hse-university/ASR_HW/runs/8q8twjc7/logs)

train.yaml [по ссылке. ](https://wandb.ai/helgahelga-hse-university/ASR_HW/runs/843uvm4b/logs)

### Итоговое обучение модели

Финальная модель была обучена в два этапа:

1. **Первый этап** (`train_argmax.yaml`) – 50 эпох. Обучение заняло **1 час 38 минут** на A100 в Google Colab. [Отчет с графиками в WanDB](https://wandb.ai/helgahelga-hse-university/ASR_HW/runs/8q8twjc7/workspace?nw=nwuserhelgahelga)
2. **Второй этап** (`train.yaml`) – 100 эпох. Обучение заняло **7 часов 17 минут** на A100 в Google Colab. [Отчет с графиками в WanDB](https://wandb.ai/helgahelga-hse-university/ASR_HW/runs/843uvm4b/workspace?nw=nwuserhelgahelga)

### Вспомогательные функции
calc_wer_cer.py вычисляет и выводит метрики (WER и CER)
download_best_model.py загружает лучшую модель по указанному пути (-path custom/path/to/model.pth) 

### Попытки обучения и сложные моменты

Первоначально модель пыталась обучаться на локальной видеокарте **NVIDIA GeForce GTX 1050**, однако из-за низкой вычислительной мощности процесс шел **очень медленно**. Поэтому было принято решение арендовать **A100 в Google Colab**, что значительно ускорило обучение.

### Сравнение конфигураций `train.yaml` и `train_argmax.yaml`

#### 1. **Основные отличия**

| Параметр                | `train_argmax.yaml`                          | `train.yaml`                                | Объяснение различий                                                                 |
|-------------------------|---------------------------------------------|--------------------------------------------|-------------------------------------------------------------------------------------|
| **Метрики**             | `metrics: example`                          | `metrics: bs_nolm_metrics`                 | В `train.yaml` используются метрики с Beam Search без языковой модели (LM).         |
| **Токенизация**         | Стандартная (по символам)                   | `vocab_type: "bpe"`                        | В `train.yaml` используется Byte Pair Encoding (BPE) для токенизации текста.        |
| **Количество эпох**     | `n_epochs: 50`                              | `n_epochs: 100`                            | В `train.yaml` обучение длится в два раза дольше.                                    |
| **Логирование**         | `log_step: 25`                              | `log_step: 50`                             | В `train.yaml` логирование происходит реже (каждые 50 шагов вместо 25).              |
| **Мониторинг метрик**   | `monitor: "min val_WER_(Argmax)"`           | `monitor: "min val_WER_(BeamSearch)"`      | В `train.yaml` используется Beam Search для вычисления WER.                         |
| **Использование LM**    | Нет                                         | `use_lm: False`                            | В `train.yaml` явно указано, что языковая модель не используется.                   |

#### 2. **Результаты инференса**

На основе результатов инференса можно сделать следующие выводы:

В процессе инференса использовалась языковая модель с методом BeamSearch для декодирования

- **CER (Character Error Rate)**:
  - **Argmax**: 7.41%
  - **LM-BeamSearch**: 6.14%

  Модель, обученная с использованием Beam Search (как в `train.yaml`), показывает лучший результат на уровне символов (CER снижен с 7.41% до 6.14%).

- **WER (Word Error Rate)**:
  - **Argmax**: 23.72%
  - **LM-BeamSearch**: 16.48%

  На уровне слов модель с Beam Search также показывает значительное улучшение (WER снижен с 23.72% до 16.48%).

---


### Итоговые метрики на тесте

```
Best model successfully downloaded!
test: 100% 82/82 [01:03<00:00, 1.29it/s]
test_CER_(Argmax): 0.0741
test_WER_(Argmax): 0.2372
test_CER_(LM-BeamSearch): 0.0614
test_WER_(LM-BeamSearch): 0.1648
```
## Вывод
train_argmax.yaml подходит для быстрого тестирования модели, но показывает более высокие ошибки (WER = 23.72%, CER = 7.41%).

train.yaml с Beam Search и BPE обеспечивает лучшее качество (WER = 16.48%, CER = 6.14%), но требует больше времени на обучение.