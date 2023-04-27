FROM python:3.11
RUN pip install python-telegram-bot==20.2 git+https://github.com/zzh1996/openai-python.git@fix
