# Задаём базовый образ
FROM python:3.9
# Задаём рабочую директорию контейнера
WORKDIR /usr/src/app
# Копируем содержимое папки app в рабочую директорию контейнера
COPY ./app ./
# Копируем файл requirements.txt в рабочую директорию контейнера
COPY ./requirements.txt ./
# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
# Исполняем команду для запуска скрипта
CMD [ "python", "./server.py" ]
