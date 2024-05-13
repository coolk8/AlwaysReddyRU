# Welcome to Russian version of AlwaysReddy 🔊
*English:*
This is just a fork of great app https://github.com/ILikeAI/AlwaysReddy. Please support and star it!

*Russian:*
Сейчас это лишь форк отличного скрипта https://github.com/ILikeAI/AlwaysReddy. Поддержите оригинальный проект звездочкой!

Задача проекта минимум:
- Подключение всех известных локальных SST/TSS библиотек для русского языка в существующий код.

Задача проекта максимум:
- Создать минимального голосового помощника на русском языке, который запускается на компьютерах с 8 GB VRAM с простым интерфейса чата или однокнопочным деплоем на своих серверах.

Для этого планируется:
- Подключение всех известных локальных SST/TTS библиотек для русского языка.
- Создание STT/TTS/ LLM (7B) арен, специфичных для русского языка, чтобы выяснить лучшую по Elo рейтингу.
- Тренировка русского голоса для Piper TTS.

Полная инструкция по пользованию доступна в оригинальном репозитории.
Здесь размещен лишь частичный перевод авторского readme.


## Разделы
- [Meet AlwaysReddy](#Встречайте-AlwaysReddy)
- [Поддерживаемые серверы LLM](#Поддерживаемые-провайдеры-LLM)
- [Поддерживаемые системы TTS](#Поддерживаемые-системы-TTS)
- [Настройка](#настройка)
- [Известные проблемы](#Известные-проблемы)
- [Устранение неполадок](# Дополнительные-логи)

## Встречайте AlwaysReddy
AlwaysReddy — простой помощник LLM с идеальным пользовательским интерфейсом… Неа!
Вы взаимодействуете с ним полностью с помощью горячих клавиш, он может легко читать или записывать в буфер обмена.
Это похоже на то, что на вашем компьютере всегда работает голосовой ChatGPT: вы просто нажимаете горячую клавишу, и он выслушивает любые ваши вопросы, нет необходимости менять местами окна или вкладки, и если вы хотите добавить в него дополнительный текст, просто скопируйте текст и дважды нажмите горячую клавишу!

**демо видео работы** https://www.reddit.com/r/LocalLLaMA/comments/1ca510h/voice_chatting_with_llama_3_8b/

### Функции:
Вы взаимодействуете с AlwaysReddy с помощью горячих клавиш. Что он может:
- Голосовой чат через TTS и STT (в том числе и локальные) 
- Чтение из буфера обмена (с помощью быстрого двойного нажатия R + Ctrl + Alt + R + R). ПРИМЕЧАНИЕ. В Linux есть другая горячая клавиша!
- Написать свой текст в буфер обмена по запросу.
- Может быть запущен на 100% локально!!!

### Случаи использования:
Я часто использую AlwaysReddy для следующих целей:
- Когда я только что изучил новую концепцию, я часто объясняю ее вслух AlwaysReddy, и он сохраняет ее (примерно моими словами) в заметку.
- «Как называется Х?» Часто я знаю, как примерно что-то описать, но не могу вспомнить, как это называется. AlwaysReddy помогает быстро дать мне ответ без необходимости открывать браузер.
- «Можете ли вы проверить текст в моем буфере обмена, прежде чем я его отправлю?»
- «Судя по комментариям в моем буфере обмена, что пользователи r/LocalLLaMA думают о X?»
- Быстрые записи в журнале: я быстро перечисляю, что я сделал сегодня, и записываю запись в буфер обмена, прежде чем выключить компьютер на весь день.


## Поддерживаемые провайдеры LLM
- OpenAI
- Anthropic
- TogetherAI
- LM Studio (локальная) - [Руководство по установке](https://youtu.be/b6MPdboJEfk)
- Ollama (локальный) - [Руководство по настройке](https://youtu.be/BMYwT58rtxw?si=LHTTm85XFEJ5bMUD)

## Поддерживаемые системы TTS
- Piper TTS (локальный и быстрый) [Узнайте, как изменить модель голоса](#how-to-add-new-voices-for-piper-tts)
- API TTS OpenAI

## Настройка:

<details>
<summary>Инструкции по настройке графического процессора</summary>

## Ускорение графического процессора

Чтобы использовать ускорение графического процессора с API более быстрого шепота, выполните следующие действия:

1. Проверьте, установлен ли уже CUDA:
    - Откройте терминал или командную строку.
    - Выполните следующую команду:
     ```
     nvcc --version
     ```
   - Если установлен CUDA, вы должны увидеть вывод, похожий на:
     ```
     nvcc: NVIDIA (R) Cuda compiler driver
     Copyright (c) 2005-2021 NVIDIA Corporation
     Built on Sun_Feb_14_21:12:58_PST_2021
     Cuda compilation tools, release 11.2, V11.2.152
     Build cuda_11.2.r11.2/compiler.29618528_0
     ```
   - Запишите версию CUDA (например, 11.2 в приведенном выше примере).).

2. Если CUDA не установлен или вы хотите установить другую версию:
    - Посетите официальный веб-сайт NVIDIA CUDA Toolkit: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
    - Загрузите и установите соответствующую версию CUDA Toolkit для вашей системы.

3. Установите PyTorch с поддержкой CUDA в зависимости от вашей системы и версии CUDA. Следуйте инструкциям на официальном сайте PyTorch: [Установка PyTorch](https://pytorch.org/get-started/locally/)

    Пример команды для CUDA 11.6:
   ```
   pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
   ```

4. В файле config.py установите USE_GPU = True, чтобы включить ускорение графического процессора.

</details>

**Кстати:** Всякий раз, когда вы получаете новую версию AlwaysReddy, вам может потребоваться снова запустить сценарий установки и снова скопировать файл конфигурации, поскольку я постоянно обновляю этот проект, и часто контекст config.py меняется.

### Настройка для Windows:

1. Клонируйте этот репозиторий с помощью `git clone https://github.com/ILikeAI/AlwaysReddy`.
2. Перейдите в каталог `cd AlwaysReddy`
3. Создайте виртуальную среду с помощью `python -m venv venv`. Этот шаг важен, обязательно назовите ее именно `venv`.
4. Активируйте виртуальную среду: `venv\Scripts\activate`
5. Установите требования с помощью `pip install -r requirements.txt`. Также запустите `pip install -r fast_whisper_requirements.txt`, если вы хотите запустить whisper локально. 
6. Запустите сценарий установки с помощью `python setup.py`. Это также создаст файл запуска run_AlwaysReddy.bat.
7. Откройте файлы config.py и .env и обновите их, указав свои настройки и ключ API.
8. Запустите помощник с помощью `run_AlwaysReddy.bat` или `python main.py`. Запущенный файл автоматически активирует виртуальную среду.

Если вы получили сообщение об ошибке, сообщающее, что вам необходимо установить ffmpeg, попробуйте выполнить следующие действия: https://github.com/openai/whisper#setup.

### Настройка для Linux:
Поддержка Linux является суперэкспериментальной, но у меня она работает, свяжитесь со мной, если у вас возникнут проблемы.

1. Клонируйте этот репозиторий с помощью `git clone https://github.com/ILikeAI/AlwaysReddy`.
2. перейдите в каталог `cd AlwaysReddy`
3. Создайте виртуальную среду с помощью `python3 -m venv venv`. Этот шаг важен, обязательно назовите ее именно `venv`.
4. Активируйте виртуальную среду: `source venv/bin/activate`
5. Установите требования с помощью `pip install -r require.txt`. Также запустите `pip install -r fast_whisper_requirements.txt`, если вы хотите запустить whisper локально. 
6. Запустите сценарий установки с помощью `python3 setup.py`. Это также создаст файл запуска run_AlwaysReddy.sh.
7. Откройте файлы config.py и .env и обновите их, указав свои настройки и ключ API.
8. Запустите помощник с помощью `./run_AlwaysReddy.sh` или `python3 main.py`. Запущенный файл автоматически активирует виртуальную среду.

Обратите внимание, что в Linux мы используем библиотеку Pynput, которая не позволяет нам использовать пробел или табуляцию в наших горячих клавишах.

Если вы получили сообщение об ошибке, сообщающее, что вам необходимо установить ffmpeg, попробуйте выполнить следующие действия: https://github.com/openai/whisper#setup.

## Известные проблемы:
- В Linux он обнаруживает нажатия горячих клавиш только тогда, когда приложение находится в фокусе. Это серьезная проблема, поскольку весь смысл проекта заключается в том, чтобы он работал в фоновом режиме. Если вы хотите помочь, это будет здорово!

## Дополнительные логи:
Если у вас возникли проблемы, попробуйте удалить папку venv и начать заново.
Установите VERBOSE = True в конфигурации, чтобы получить более подробные журналы и трассировки ошибок.

## Как:
### Как использовать AlwaysReddy:
На данный момент есть только 2 основных действия:

Голосовой чат:
- Нажмите `Ctrl + Alt + R`, чтобы начать диктовку, вы можете говорить сколько угодно долго, затем снова нажмите `Ctrl + Alt + R`, чтобы остановить запись, через несколько секунд вы получите голосовой ответ.

Голосовой чат с контекстом вашего буфера обмена:
- Дважды нажмите «Ctrl + Alt + R» (или просто удерживайте «Ctrl + Alt» и дважды быстро нажмите «R»). Это предоставит ИИ содержимое вашего буфера обмена, чтобы вы могли попросить его сослаться на него, переписать его, ответить. вопросы из его содержания... как хотите!
- Очистите память помощников с помощью `Ctrl+Alt+W`.
- Отмените запись или TTS с помощью `Ctrl + Alt + E`

**Пожалуйста, дайте мне знать, если вы думаете о лучших настройках горячих клавиш по умолчанию!**

Все горячие клавиши можно редактировать в config.py.


### Как добавить новые голоса для Piper TTS:
1. Перейдите на https://huggingface.co/rhasspy/piper-voices/tree/main и выберите нужный язык.
2. Нажмите на название голоса, который хотите опробовать. Доступны модели разных размеров; Я предлагаю использовать средний размер, поскольку он довольно быстрый, но при этом звучит великолепно (для локально запускаемой модели).
3. Прослушайте образец в папке «образцы», чтобы убедиться, что голос вам нравится.
4. Загрузите файлы `.onnx` и `.json` для выбранного голоса.
5. Создайте новую папку в каталоге `piper_tts\voices` и дайте ей описательное имя. Вам нужно будет ввести имя этой папки в файл config.py. Например: `PIPER_VOICE = "default_female_voice"`.
6. Переместите два загруженных файла (`.onnx` и `.json`) в вновь созданную папку в каталоге `piper_tts\voices`.

### Как использовать STT локального whisper:
1. Откройте файл `config.py`.
2. Найдите раздел «Настройки API STT».
3. Закомментируйте строку `TRANSCRIPTION_API = "openai"`, добавив `#` в начало строки.
4. Раскомментируйте строку `TRANSCRIPTION_API = "faster-whisper"`, удалив `#` в начале строки.
5. Настройте параметры «WHISPER_MODEL» и «TRANSCRIPTION_LANGUAGE» в соответствии со своими предпочтениями.
6. Сохраните файл `config.py`.

Вот пример того, как должен выглядеть ваш файл `config.py` для локального whisper:

Доступные модели: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, big-v1, big-v2, big-v3, big, distil-large-v2, distil-medium. .en, distil-small.en, distil-large-v3

``` python
### Transcription API Settings ###

## OPENAI API TRANSCRIPTION EXAMPLE ##
# TRANSCRIPTION_API = "openai"  # this will use the hosted openai api

## Faster Whisper local transcription ###
TRANSCRIPTION_API = "FasterWhisper" # this will use the local whisper model

# Supported models: 
WHISPER_MODEL = "tiny.en" # If you prefer not to use english set it to "tiny", if the transcription quality is too low then set it to "base" but this will be a little slower

```

Примечание. Модель whisper по умолчанию предназначена только для английского языка. Попробуйте установить для WHISPER_MODEL значение 'tiny' или 'base' для других языков.

### Как поменять провайдера LLM и модель
Чтобы поменять модели, откройте файл config.py и раскомментируйте разделы API, который вы хотите использовать. Например, вот как вы будете использовать сонет Клода 3: если вы хотите использовать студию LM, вы должны закомментировать раздел Anthropic и раскомментировать раздел студии LM.

```python
### COMPLETIONS API SETTINGS ###

## LM Studio COMPLETIONS API EXAMPLE ##
# COMPLETIONS_API = "lm_studio" 
# COMPLETION_MODEL = "local-model" #This stays as local-model no matter what model you are using

## ANTHROPIC COMPLETIONS API EXAMPLE ##
COMPLETIONS_API = "anthropic" 
COMPLETION_MODEL = "claude-3-sonnet-20240229" 

## TOGETHER COMPLETIONS API EXAMPLE ##
# COMPLETIONS_API = "together"
# COMPLETION_MODEL = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT" 

## OPENAI COMPLETIONS API EXAMPLE ##
# COMPLETIONS_API = "openai"
# COMPLETION_MODEL = "gpt-4-0125-preview"
```

### Как использовать локальный TTS
Чтобы использовать локальный TTS, просто откройте файл конфигурации и установите `TTS_ENGINE="piper"`


