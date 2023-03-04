FROM python
RUN pip install python-telegram-bot==13.15 openai
RUN apt-get update && apt-get -y install ffmpeg libavcodec-extra && pip install pydub==0.25.1
