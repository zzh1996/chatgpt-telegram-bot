FROM python:3.11
RUN pip install python-telegram-bot==20.2 openai==1.61.0 aioboto3 beautifulsoup4 lxml tiktoken==0.8.0 yt-dlp
