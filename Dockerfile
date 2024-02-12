FROM python:3.11
RUN pip install Telethon==1.32.1 cryptg==0.4.0 aiohttp[speedups] pyjwt
