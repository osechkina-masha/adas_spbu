# adas_spbu

## Оптимизация разметки ground truth data пешеходов

Данный трекер интергирован с [СVAT](https://github.com/opencv/cvat) - Computer Vision Annotation Tool.

Для того, чтобы им воспользоваться, необходимо:

Операционная система: Ubuntu 18.04+

Склонировать репозиторий:

```{r, engine='bash', count_lines}
git clone https://github.com/zhitm/cvat_with_mytracker.git
```

Установить docker и docker compose, а затем собрать сvat:

```{r, engine='bash', count_lines}
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

Установить nuclio версии 1.8.14

```{r, engine='bash', count_lines}
wget https://github.com/nuclio/nuclio/releases/download/1.8.14/nuctl-1.8.14-linux-amd64
sudo chmod +x nuctl-<version>-linux-amd64
sudo ln -sf $(pwd)/nuctl-<version>-linux-amd64 /usr/local/bin/nuctl
```

Создать проект cvat в nuclio

```{r, engine='bash', count_lines}
nuctl create project cvat
```

Сделать мой скрипт для запуска трекера исполняемым и запустить его:

```{r, engine='bash', count_lines}
sudo chmod +x connectMyTracker.sh
./connectMyTracker.sh
```

Ждать, пока он закончит работу, а затем открыть localhost:8080 в браузере. Можно пользоваться трекером после создания
таски с видео.
