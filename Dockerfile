FROM python:3.12
RUN pip install Telethon==1.38.1 cryptg==0.4.0 aiohttp[speedups] pyjwt
