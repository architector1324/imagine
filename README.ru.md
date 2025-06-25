## ✨ Imagine: Легковесный API для преобразования текста в изображение с Stable Diffusion

![](logo.png)

*Изображение сгенерировано с помощью Imagine: "фото астронавта, едущего на лошади по Марсу, эпичное, кинематографичное, детализированное"*

**Imagine** - это простой, но мощный HTTP-сервер, разработанный для генерации изображений по текстовым запросам с использованием моделей Stable Diffusion и библиотеки `diffusers` от Hugging Face. Он предоставляет одну, понятную конечную точку, которая принимает JSON-запрос и возвращает сгенерированное изображение в виде строки, закодированной в Base64, что идеально подходит для быстрой интеграции в ваши приложения.

### Ключевые особенности

*   **Минималистичный API:** Одна простая в использовании конечная точка `/generate` для всех ваших потребностей в преобразовании текста в изображение.
*   **Только преобразование текста в изображение (txt2img):** Сосредоточенность на основной функциональности, без `img2img` или других сложных режимов, что обеспечивает лаконичную и целенаправленную кодовую базу.
*   **Вывод в Base64:** Удобно получайте сгенерированные изображения в виде строк Base64, что позволяет легко встраивать их непосредственно в веб-страницы, мобильные приложения или другие сервисы, не управляя файловым хранилищем.
*   **Настраиваемые параметры:** Контролируйте параметры генерации изображений, такие как `width`, `height`, `num_steps`, `guidance_scale`, `sampler`, `seed` и `negative_prompt`, через JSON-полезную нагрузку.
*   **Создан на Flask и Diffusers:** Использует надежные и популярные библиотеки Python для обеспечения надежности и простоты использования.
*   **Независимость от аппаратного обеспечения:** Поддерживает CPU, CUDA (NVIDIA), MPS (Apple Silicon) и потенциально ROCm (AMD) в зависимости от вашей настройки PyTorch.

### Почему Imagine?

**Imagine** вдохновлен философией таких инструментов, как **Ollama**, и предлагает аналогичный подход, но для моделей Stable Diffusion. Он идеально подходит для разработчиков, которым требуется легковесное решение для запуска генерации изображений Stable Diffusion **как локального сервиса в фоновом режиме**.

Это позволяет вам интегрировать возможности преобразования текста в изображение **из любого места**: будь то скрипт командной строки, Python-приложение, веб-страница или любой другой сервис. Вам больше не нужно беспокоиться о накладных расходах и сложности многофункциональных пользовательских интерфейсов; Imagine обеспечивает быстрый, API-ориентированный доступ к Stable Diffusion, разработанный для беспрепятственного развертывания и простого использования.


### Установка и использование

1.  **Клонируйте репозиторий:**
    ```bash
    git clone https://github.com/your-username/imagine.git
    cd imagine
    ```

2.  **Установите зависимости:**
    ```bash
    pip install torch diffusers transformers accelerate flask Pillow argparse requests base64
    ```
    (Для использования CUDA (NVIDIA GPU) убедитесь, что `torch` установлен правильно, например: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)

3.  **Укажите модель Stable Diffusion:**
    Отредактируйте `imagine.py` и установите `DEFAULT_MODEL` на путь к вашему файлу модели (`.safetensors` или `.ckpt`), например, `dreamshaper_8.safetensors`.

4.  **Настройте устройство (для производительности):**
    В `imagine.py` измените `DEFAULT_DEVICE = 'cpu'` на `'cuda'` (NVIDIA), `'mps'` (Apple Silicon) или оставьте `'cpu'` для CPU-вывода (медленнее).

5.  **Запустите сервер:**
    ```bash
    python imagine.py --host '0.0.0.0' -p 5000 -d cpu -f 32
    # python imagine.py
    ```
    Сервер запустится по адресу `http://0.0.0.0:5000/`.

6.  **Отправьте POST-запрос на `/generate`:**

    **Пример с использованием `curl`:**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
         -d '{
             "model": "<путь к>/dreamshaper_8.safetensors"
             "prompt": "фотография астронавта, скачущего на лошади по Марсу, эпично, кинематографично, детализированно",
             "width": 768,
             "height": 512,
             "num_steps": 25,
             "guidance": 7.0,
             "sampler": "DPM++ 2M",
             "neg": "уродливый, деформированный, размытый, низкое качество"
         }' \
         http://localhost:5000/generate | jq .
    ```

    В ответ вы получите JSON-объект, содержащий строку `img` и `seed`.

### Imagine CLI

### Описание

Простой скрипт командной строки для генерации изображений на основе текстовой подсказки с использованием библиотеки `diffusers`.

### Использование

```
usage: imagine [-m MODEL] [-o OUTPUT] [-w WIDTH] [-h HEIGHT] [-n NUM_STEPS] [-g GUIDANCE] [-s SAMPLER] [--seed SEED] [--neg NEG] [--stream STREAM] [--help] prompt [prompt ...]

SD image generator

positional arguments:
  prompt                Prompt for model

options:
  -m, --model MODEL     SD model
  -o, --output OUTPUT   Output image
  -w, --width WIDTH     Output image width
  -h, --height HEIGHT   Output image height
  -n, --num_steps NUM_STEPS
                        Number of steps
  -g, --guidance GUIDANCE
                        Guidance scale
  -s, --sampler SAMPLER
                        SD Sampler ['DDIM', 'Euler', 'Euler a', 'Heun', 'LMS', 'DPM++ 2M', 'DPM++ 2S', 'DPM++ SDE', 'DPM2', 'DPM2 a']
  --seed SEED           Seed
  --neg NEG             Negative prompt
  --stream STREAM       Stream steps samples to output image
  --help
```

#### Пример использования

```bash
./imagine-cli.py 'фотография астронавта, скачущего на лошади по Марсу, эпично, кинематографично, детализированно' -w 768 -h 512 -n 25 -g 7.0 -s 'DPM++ 2M' --neg 'уродливый, деформированный, размытый, низкое качество'
```

### Вывод

Скрипт сохранит сгенерированное изображение в текущую директорию с именем файла, основанным на подсказке или временной метке.

## Расширенное развертывание: Запуск как служба Systemd

Для постоянной и надежной работы вы можете настроить **Imagine** как службу `systemd`. Это гарантирует, что сервер будет автоматически запускаться при загрузке и перезапускаться в случае сбоев.

**1. Создайте символические ссылки и убедитесь в исполняемости:**

Убедитесь, что как ваш серверный скрипт (`imagine.py`), так и утилита командной строки (`imagine-cli.py`) являются исполняемыми и имеют символические ссылки на общедоступную директорию `PATH`, такую как `/usr/bin/`.

```bash
# Сделайте скрипты исполняемыми
chmod +x /path/to/imagine/imagine.py
chmod +x /path/to/imagine/imagine-cli.py

# Создайте символические ссылки
# Замените `/path/to/imagine` на фактический путь к директории вашего проекта, если он отличается
sudo ln -s /path/to/imagine/imagine.py /usr/bin/imagine
sudo ln -s /path/to/imagine/imagine-cli.py /usr/bin/imagine-cli
```

**2. Создайте файл службы Systemd:**

Создайте файл с именем `imagine.service` в `/etc/systemd/system/`:

```bash
sudo nano /etc/systemd/system/imagine.service
```

Вставьте следующее содержимое в файл:

```ini
[Unit]
Description=Imagine: Сервер генерации изображений Stable Diffusion
After=network.target syslog.target

[Service]
# ЗАМЕНИТЕ 'arch' НА ВАШЕ ИМЯ ПОЛЬЗОВАТЕЛЯ LINUX!
# Запускайте сервис от имени вашего текущего пользователя.
# Это упрощает управление разрешениями, так как скрипт и модель, вероятно, находятся в вашей домашней директории.
User=arch

# Команда для выполнения при запуске сервиса.
# Убедитесь, что /usr/bin/imagine указывает на ваш основной серверный скрипт (imagine.py).
ExecStart=/usr/bin/imagine

# Перезапускать сервис, если он аварийно завершится
Restart=on-failure
RestartSec=5s

# Стандартный вывод и ошибки будут направлены в журнал systemd для удобной отладки
StandardOutput=journal
StandardError=journal

# Тип сервиса: simple (по умолчанию) или forking
Type=simple

[Install]
# Этот юнит должен запускаться, когда система достигает состояния multi-user.target (нормальная загрузка)
WantedBy=multi-user.target
```

**Важные примечания:**
*   **Замените `arch` на ваше фактическое имя пользователя Linux!**
*   Убедитесь, что символическая ссылка `/usr/bin/imagine` корректно указывает на ваш **серверный скрипт** (`imagine.py`).

**3. Включите и запустите службу:**

После сохранения файла `imagine.service`:

```bash
# Перезагрузите systemd, чтобы он распознал новый сервис
sudo systemctl daemon-reload

# Включите сервис для автоматического запуска при загрузке
sudo systemctl enable imagine.service

# Запустите сервис немедленно
sudo systemctl start imagine.service
```

**4. Проверьте статус службы и логи:**

Чтобы проверить, что служба работает корректно:

```bash
sudo systemctl status imagine.service
```

Чтобы просмотреть логи в реальном времени для отладки:

```bash
journalctl -u imagine.service -f
```

