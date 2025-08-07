import asyncio
import os
import logging
import shelve
import datetime
import time
import traceback
import hashlib
import base64
import copy
from collections import defaultdict
from richtext import RichText
import openai
import httpx
from telethon import TelegramClient, events, errors, functions, types
import signal
from contextlib import asynccontextmanager

def debug_signal_handler(signal, frame):
    breakpoint()

signal.signal(signal.SIGUSR1, debug_signal_handler)

ADMIN_ID = 71863318

GPT_35_PROMPT = 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.\nKnowledge cutoff: 2021-09\nCurrent date: {current_date}'
GPT_4_PROMPT = 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2021-09\nCurrent date: {current_date}'
GPT_4_PROMPT_2 = 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2023-04\nCurrent date: {current_date}'
GPT_4_TURBO_PROMPT = 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2023-12\nCurrent date: {current_date}\n\nImage input capabilities: Enabled\nPersonality: v2'
GPT_4O_PROMPT = 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2023-10\nCurrent date: {current_date}'

MODELS = [
    {'prefix': '$', 'model': 'gpt-5-chat-latest', 'prompt_template': ''},

    {'prefix': '5$', 'model': 'gpt-5-2025-08-07', 'prompt_template': ''},
    {'prefix': '5m$', 'model': 'gpt-5-mini-2025-08-07', 'prompt_template': ''},
    {'prefix': '5n$', 'model': 'gpt-5-nano-2025-08-07', 'prompt_template': ''},
    {'prefix': '5c$', 'model': 'gpt-5-chat-latest', 'prompt_template': ''},

    {'prefix': 'o3p$', 'model': 'o3-pro', 'prompt_template': ''},
    {'prefix': 'o3-pro$', 'model': 'o3-pro', 'prompt_template': ''},
    {'prefix': 'o3-pro-2025-06-10$', 'model': 'o3-pro-2025-06-10', 'prompt_template': ''},
    {'prefix': 'o3$', 'model': 'o3', 'prompt_template': ''},
    {'prefix': 'o3-2025-04-16$', 'model': 'o3-2025-04-16', 'prompt_template': ''},
    {'prefix': 'o4m$', 'model': 'o4-mini', 'prompt_template': ''},
    {'prefix': 'o4-mini-2025-04-16$', 'model': 'o4-mini-2025-04-16', 'prompt_template': ''},

    {'prefix': '41$', 'model': 'gpt-4.1', 'prompt_template': ''},
    {'prefix': 'gpt-4.1-2025-04-14$', 'model': 'gpt-4.1-2025-04-14', 'prompt_template': ''},
    {'prefix': '41m$', 'model': 'gpt-4.1-mini', 'prompt_template': ''},
    {'prefix': 'gpt-4.1-mini-2025-04-14$', 'model': 'gpt-4.1-mini-2025-04-14', 'prompt_template': ''},
    {'prefix': '41n$', 'model': 'gpt-4.1-nano', 'prompt_template': ''},
    {'prefix': 'gpt-4.1-nano-2025-04-14$', 'model': 'gpt-4.1-nano-2025-04-14', 'prompt_template': ''},

    {'prefix': '4oms$', 'model': 'gpt-4o-mini-search-preview', 'prompt_template': ''},
    {'prefix': 'gpt-4o-mini-search-preview-2025-03-11$', 'model': 'gpt-4o-mini-search-preview-2025-03-11', 'prompt_template': ''},
    {'prefix': '4os$', 'model': 'gpt-4o-search-preview', 'prompt_template': ''},
    {'prefix': 'gpt-4o-search-preview-2025-03-11$', 'model': 'gpt-4o-search-preview-2025-03-11', 'prompt_template': ''},

    {'prefix': '45$', 'model': 'gpt-4.5-preview', 'prompt_template': ''},
    {'prefix': 'gpt-4.5-preview$', 'model': 'gpt-4.5-preview', 'prompt_template': ''},
    {'prefix': 'gpt-4.5-preview-2025-02-27$', 'model': 'gpt-4.5-preview-2025-02-27', 'prompt_template': ''},

    {'prefix': '4o$', 'model': 'gpt-4o-2024-11-20', 'prompt_template': GPT_4O_PROMPT},
    {'prefix': '4om$', 'model': 'gpt-4o-mini-2024-07-18', 'prompt_template': GPT_4O_PROMPT},
    {'prefix': '4$', 'model': 'gpt-4-turbo-2024-04-09', 'prompt_template': GPT_4_TURBO_PROMPT},
    {'prefix': '3$', 'model': 'gpt-3.5-turbo-0125', 'prompt_template': GPT_35_PROMPT},
    {'prefix': 'o1$', 'model': 'o1', 'prompt_template': ''},
    {'prefix': 'o1m$', 'model': 'o1-mini', 'prompt_template': ''},
    {'prefix': 'o1p$', 'model': 'o1-preview', 'prompt_template': ''},
    {'prefix': 'o3m$', 'model': 'o3-mini', 'prompt_template': ''},

    {'prefix': 'o1-preview$', 'model': 'o1-preview', 'prompt_template': ''},
    {'prefix': 'o1-preview-2024-09-12$', 'model': 'o1-preview-2024-09-12', 'prompt_template': ''},
    {'prefix': 'o1-mini$', 'model': 'o1-mini', 'prompt_template': ''},
    {'prefix': 'o1-mini-2024-09-12$', 'model': 'o1-mini-2024-09-12', 'prompt_template': ''},
    {'prefix': 'o1-2024-12-17$', 'model': 'o1-2024-12-17', 'prompt_template': ''},
    {'prefix': 'o3-mini$', 'model': 'o3-mini', 'prompt_template': ''},
    {'prefix': 'o3-mini-2025-01-31$', 'model': 'o3-mini-2025-01-31', 'prompt_template': ''},
    {'prefix': 'o1-pro$', 'model': 'o1-pro', 'prompt_template': ''},
    {'prefix': 'o1-pro-2025-03-19$', 'model': 'o1-pro-2025-03-19', 'prompt_template': ''},

    {'prefix': 'chatgpt-4o-latest$', 'model': 'chatgpt-4o-latest', 'prompt_template': GPT_4O_PROMPT},
    {'prefix': 'chatgpt$', 'model': 'chatgpt-4o-latest', 'prompt_template': GPT_4O_PROMPT},

    {'prefix': 'gpt-4o-mini-2024-07-18$', 'model': 'gpt-4o-mini-2024-07-18', 'prompt_template': GPT_4O_PROMPT},
    {'prefix': 'gpt-4o-mini$', 'model': 'gpt-4o-mini', 'prompt_template': GPT_4O_PROMPT},

    {'prefix': 'gpt-4o-2024-05-13$', 'model': 'gpt-4o-2024-05-13', 'prompt_template': GPT_4O_PROMPT},
    {'prefix': 'gpt-4o-2024-08-06$', 'model': 'gpt-4o-2024-08-06', 'prompt_template': GPT_4O_PROMPT},
    {'prefix': 'gpt-4o-2024-11-20$', 'model': 'gpt-4o-2024-11-20', 'prompt_template': GPT_4O_PROMPT},
    {'prefix': 'gpt-4o$', 'model': 'gpt-4o', 'prompt_template': GPT_4O_PROMPT},

    {'prefix': 'gpt-4-turbo-2024-04-09$', 'model': 'gpt-4-turbo-2024-04-09', 'prompt_template': GPT_4_TURBO_PROMPT},
    {'prefix': 'gpt-4-0125-preview$', 'model': 'gpt-4-0125-preview', 'prompt_template': GPT_4_TURBO_PROMPT},
    {'prefix': 'gpt-4-1106-preview$', 'model': 'gpt-4-1106-preview', 'prompt_template': GPT_4_PROMPT_2},
    {'prefix': 'gpt-4-1106-vision-preview$', 'model': 'gpt-4-1106-vision-preview', 'prompt_template': GPT_4_PROMPT_2},
    {'prefix': 'gpt-4-0613$', 'model': 'gpt-4-0613', 'prompt_template': GPT_4_PROMPT},
    {'prefix': 'gpt-4-32k-0613$', 'model': 'gpt-4-32k-0613', 'prompt_template': GPT_4_PROMPT},

    {'prefix': 'gpt-3.5-turbo-0125$', 'model': 'gpt-3.5-turbo-0125', 'prompt_template': GPT_35_PROMPT},
    {'prefix': 'gpt-3.5-turbo-1106$', 'model': 'gpt-3.5-turbo-1106', 'prompt_template': GPT_35_PROMPT},
    {'prefix': 'gpt-3.5-turbo-0613$', 'model': 'gpt-3.5-turbo-0613', 'prompt_template': GPT_35_PROMPT},
    {'prefix': 'gpt-3.5-turbo-16k-0613$', 'model': 'gpt-3.5-turbo-16k-0613', 'prompt_template': GPT_35_PROMPT},
    {'prefix': 'gpt-3.5-turbo-0301$', 'model': 'gpt-3.5-turbo-0301', 'prompt_template': GPT_35_PROMPT},
]
DEFAULT_MODEL = 'gpt-4-0613' # For compatibility with the old database format

SHOW_COST_THRESHOLD = 0.1
PRICING = {
    'o1': (15, 60, 7.5, True),
    'o1-2024-12-17': (15, 60, 7.5, True),
    'o1-preview': (15, 60, 7.5, True),
    'o1-preview-2024-09-12': (15, 60, 7.5, True),
    'o1-mini': (1.1, 4.4, 0.55, True),
    'o1-mini-2024-09-12': (1.1, 4.4, 0.55, True),
    'o3-mini': (1.1, 4.4, 0.55, True),
    'o3-mini-2025-01-31': (1.1, 4.4, 0.55, True),
    'gpt-4.5-preview': (75, 150, 37.5, True),
    'gpt-4.5-preview-2025-02-27': (75, 150, 37.5, True),
    'o1-pro': (150, 600, 150, True),
    'o1-pro-2025-03-19': (150, 600, 150, True),
    'o3': (2, 8, 0.5, True),
    'o3-2025-04-16': (2, 8, 0.5, True),
    'o4-mini': (1.1, 4.4, 0.275, True),
    'o4-mini-2025-04-16': (1.1, 4.4, 0.275, True),
    'o3-pro': (20, 80, 20, True),
    'o3-pro-2025-06-10': (20, 80, 20, True),

    'chatgpt-4o-latest': (5, 15, 5, False),
    'gpt-3.5-turbo-0125': (0.5, 1.5, 0.5, False),
    'gpt-3.5-turbo-0301': (1.5, 2, 1.5, False),
    'gpt-3.5-turbo-0613': (1.5, 2, 1.5, False),
    'gpt-3.5-turbo-1106': (1, 2, 1, False),
    'gpt-3.5-turbo-16k-0613': (3, 4, 3, False),
    'gpt-4-0125-preview': (10, 30, 10, False),
    'gpt-4-0613': (30, 60, 30, False),
    'gpt-4-1106-preview': (10, 30, 10, False),
    'gpt-4-1106-vision-preview': (10, 30, 10, False),
    'gpt-4-32k-0613': (60, 120, 60, False),
    'gpt-4-turbo-2024-04-09': (10, 30, 10, False),
    'gpt-4o': (2.5, 10, 1.25, False),
    'gpt-4o-2024-05-13': (5, 15, 5, False),
    'gpt-4o-2024-08-06': (2.5, 10, 1.25, False),
    'gpt-4o-2024-11-20': (2.5, 10, 1.25, False),
    'gpt-4o-mini': (0.15, 0.6, 0.075, False),
    'gpt-4o-mini-2024-07-18': (0.15, 0.6, 0.075, False),
    'gpt-4o-mini-search-preview': (0.15, 0.6, 0.15, False),
    'gpt-4o-mini-search-preview-2025-03-11': (0.15, 0.6, 0.15, False),
    'gpt-4o-search-preview': (2.5, 10, 2.5, False),
    'gpt-4o-search-preview-2025-03-11': (2.5, 10, 2.5, False),

    'gpt-5-chat-latest': (1.25, 10, 0.125, False),
    'gpt-5-2025-08-07': (1.25, 10, 0.125, True),
    'gpt-5-mini-2025-08-07': (0.25, 2, 0.025, True),
    'gpt-5-nano-2025-08-07': (0.05, 0.4, 0.005, True),
}

def get_prompt(model):
    for m in MODELS:
        if m['model'] == model:
            return m['prompt_template'].replace('{current_date}', (datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=8)).strftime('%Y-%m-%d'))
    raise ValueError('Model not found')

aclient = openai.AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=0,
    timeout=60,
)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

TELEGRAM_LENGTH_LIMIT = 4096
TELEGRAM_MIN_INTERVAL = 3
OPENAI_MAX_RETRY = 3
OPENAI_RETRY_INTERVAL = 10
SEND_BATCH_DELAY = 1
TEXT_FILE_SIZE_LIMIT = 1_000_000
PDF_FILE_SIZE_LIMIT = 32_000_000
TRIGGERS_LIMIT = 20

class PendingReplyManager:
    def __init__(self):
        self.messages = {}

    def add(self, reply_id):
        assert reply_id not in self.messages
        self.messages[reply_id] = asyncio.Event()

    def remove(self, reply_id):
        if reply_id not in self.messages:
            return
        self.messages[reply_id].set()
        del self.messages[reply_id]

    async def wait_for(self, reply_id):
        if reply_id not in self.messages:
            return
        logging.info('PendingReplyManager waiting for %r', reply_id)
        await self.messages[reply_id].wait()
        logging.info('PendingReplyManager waiting for %r finished', reply_id)

class RateLimitManager:
    def __init__(self):
        self.locks = defaultdict(asyncio.Lock)
        self.last_timestamps = defaultdict(int)

    @asynccontextmanager
    async def session(self, chat_id):
        lock = self.locks[chat_id]
        await lock.acquire()
        handle = _RateLimitSessionHandle(self, chat_id)
        try:
            yield handle
        finally:
            handle._invalidate()
            remaining_time = self.get_remaining_time(chat_id)
            if remaining_time > 0:
                asyncio.get_running_loop().call_later(remaining_time, lock.release)
            else:
                lock.release()

    def get_remaining_time(self, chat_id):
        return max(self.last_timestamps[chat_id] + TELEGRAM_MIN_INTERVAL - time.time(), 0)

def ensure_interval(func):
    async def new_func(self, *args, **kwargs):
        remaining_time = self.manager.get_remaining_time(self.chat_id)
        if remaining_time > 0:
            await asyncio.sleep(remaining_time)
        result = await func(self, *args, **kwargs)
        self.manager.last_timestamps[self.chat_id] = time.time()
        return result
    return new_func

def check_alive(func):
    async def new_func(self, *args, **kwargs):
        if not self._alive:
            raise RuntimeError("Session invalid")
        return await func(self, *args, **kwargs)
    return new_func

def retry(max_retry=30, interval=10):
    def decorator(func):
        async def new_func(*args, **kwargs):
            for _ in range(max_retry - 1):
                try:
                    return await func(*args, **kwargs)
                except errors.FloodWaitError as e:
                    logging.exception(e)
                    await asyncio.sleep(interval)
            return await func(*args, **kwargs)
        return new_func
    return decorator

class _RateLimitSessionHandle:
    def __init__(self, manager, chat_id):
        self.manager = manager
        self.chat_id = chat_id
        self._alive = True

    def _invalidate(self):
        self._alive = False

    @check_alive
    @retry()
    @ensure_interval
    async def send_message(self, text, reply_to_message_id):
        logging.info('Sending message: chat_id=%r, reply_to_message_id=%r, text=%r', self.chat_id, reply_to_message_id, text)
        text = RichText(text)
        text, entities = text.to_telegram()
        msg = await bot.send_message(
            self.chat_id,
            text,
            reply_to=reply_to_message_id,
            link_preview=False,
            formatting_entities=entities,
        )
        logging.info('Message sent: chat_id=%r, reply_to_message_id=%r, message_id=%r', self.chat_id, reply_to_message_id, msg.id)
        return msg.id

    @check_alive
    @retry()
    @ensure_interval
    async def edit_message(self, text, message_id):
        logging.info('Editing message: chat_id=%r, message_id=%r, text=%r', self.chat_id, message_id, text)
        text = RichText(text)
        text, entities = text.to_telegram()
        try:
            await bot.edit_message(
                self.chat_id,
                message_id,
                text,
                link_preview=False,
                formatting_entities=entities,
            )
        except errors.MessageNotModifiedError as e:
            logging.info('Message not modified: chat_id=%r, message_id=%r', self.chat_id, message_id)
        else:
            logging.info('Message edited: chat_id=%r, message_id=%r', self.chat_id, message_id)

    @check_alive
    @retry()
    @ensure_interval
    async def delete_message(self, message_id):
        logging.info('Deleting message: chat_id=%r, message_id=%r', self.chat_id, message_id)
        await bot.delete_messages(
            self.chat_id,
            message_id,
        )
        logging.info('Message deleted: chat_id=%r, message_id=%r', self.chat_id, message_id)

async def send_message(chat_id, text, reply_to_message_id):
    async with BotReplyMessages(chat_id, reply_to_message_id, '') as b:
        b.update(text)
        await b.finalize()

def is_whitelist(chat_id):
    whitelist = db['whitelist']
    return chat_id in whitelist

def add_whitelist(chat_id):
    whitelist = db['whitelist']
    whitelist.add(chat_id)
    db['whitelist'] = whitelist

def del_whitelist(chat_id):
    whitelist = db['whitelist']
    whitelist.discard(chat_id)
    db['whitelist'] = whitelist

def get_whitelist():
    return db['whitelist']

def only_admin(func):
    async def new_func(message):
        if message.sender_id != ADMIN_ID:
            await send_message(message.chat_id, 'Only admin can use this command', message.id)
            return
        await func(message)
    return new_func

def only_private(func):
    async def new_func(message):
        if message.chat_id != message.sender_id:
            await send_message(message.chat_id, 'This command only works in private chat', message.id)
            return
        await func(message)
    return new_func

def only_whitelist(func):
    async def new_func(message):
        if not is_whitelist(message.chat_id):
            if message.chat_id == message.sender_id:
                await send_message(message.chat_id, 'This chat is not in whitelist', message.id)
            return
        await func(message)
    return new_func

def save_photo(photo_blob): # TODO: change to async
    h = hashlib.sha256(photo_blob).hexdigest()
    dir = f'photos/{h[:2]}/{h[2:4]}'
    path = f'{dir}/{h}'
    if not os.path.isfile(path):
        os.makedirs(dir, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(photo_blob)
    return h

def load_photo(h):
    dir = f'photos/{h[:2]}/{h[2:4]}'
    path = f'{dir}/{h}'
    with open(path, 'rb') as f:
        return f.read()

def save_file(file_blob):
    h = hashlib.sha256(file_blob).hexdigest()
    dir = f'files/{h[:2]}/{h[2:4]}'
    path = f'{dir}/{h}'
    if not os.path.isfile(path):
        os.makedirs(dir, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(file_blob)
    return h

def load_file(h):
    dir = f'files/{h[:2]}/{h[2:4]}'
    path = f'{dir}/{h}'
    with open(path, 'rb') as f:
        return f.read()

async def completion(chat_history, model, chat_id, msg_id, task_id): # chat_history = [user, ai, user, ai, ..., user]
    assert len(chat_history) % 2 == 1
    system_prompt = get_prompt(model)
    messages=[{"role": "system", "content": system_prompt}] if system_prompt else []
    roles = ["user", "assistant"]
    role_id = 0
    for msg in chat_history:
        messages.append({"role": roles[role_id], "content": msg})
        role_id = 1 - role_id
    def remove_blobs(messages):
        new_messages = copy.deepcopy(messages)
        for message in new_messages:
            if 'content' in message:
                if isinstance(message['content'], list):
                    for obj in message['content']:
                        if obj['type'] == 'image_url':
                            obj['image_url']['url'] = obj['image_url']['url'][:50] + '...'
                        elif obj['type'] == 'file':
                            obj['file']['file_data'] = obj['file']['file_data'][:50] + '...'
        return new_messages

    def convert_to_responses_api_input_format(messages):
        system_prompt = None
        new_messages = []
        for message in messages:
            if message['role'] == 'system':
                system_prompt = message['content']
                continue
            elif message['role'] == 'user':
                new_message_content = []
                if isinstance(message['content'], list):
                    for item in message['content']:
                        if item['type'] == 'text':
                            new_message_content.append({'type': 'input_text', 'text': item['text']})
                        elif item['type'] == 'image_url':
                            new_message_content.append({'type': 'input_image', 'image_url': item['image_url']['url'], 'detail': item['image_url']['detail']})
                        elif item['type'] == 'file':
                            new_message_content.append({'type': 'input_file', 'filename': item['file']['filename'], 'file_data': item['file']['file_data']})
                        else:
                            raise ValueError('Unknown message type in chat history')
                else:
                    new_message_content.append({'type': 'input_text', 'text': message['content']})
                new_messages.append({'role': message['role'], 'content': new_message_content})
            elif message['role'] == 'assistant':
                if isinstance(message['content'], list):
                    assert len(message['content']) == 1 and message['content'][0]['type'] == 'text'
                    new_messages.append({'role': message['role'], 'content': message['content'][0]['text']})
                else:
                    new_messages.append({'role': message['role'], 'content': message['content']})
            else:
                raise ValueError('Unknown role in chat history')
        return new_messages, system_prompt

    is_reasoning_model = model.startswith('o') or model.startswith('gpt-5')
    support_stream = True # As of 2025-02-14, o1 supports streaming
    support_reasoning_effort = model in ['o1', 'o1-2024-12-17', 'o3-mini', 'o3-mini-2025-01-31', 'o1-pro', 'o1-pro-2025-03-19', 'o3', 'o3-2025-04-16', 'o4-mini', 'o4-mini-2025-04-16', 'o3-pro', 'o3-pro-2025-06-10', 'gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']
    support_reasoning_summary = model in ['o1', 'o1-2024-12-17', 'o3-mini', 'o3-mini-2025-01-31', 'o3', 'o3-2025-04-16', 'o4-mini', 'o4-mini-2025-04-16', 'o3-pro', 'o3-pro-2025-06-10', 'gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']
    is_search_model = 'search' in model
    is_responses_api = model.startswith('o1-pro') or support_reasoning_summary or model.startswith('gpt-5')
    if not is_responses_api:
        kwargs = {'model': model}
        if support_stream:
            kwargs['stream'] = True
            kwargs['stream_options'] = {"include_usage": True}
        if is_reasoning_model:
            kwargs['timeout'] = httpx.Timeout(timeout=3600, connect=15)
        if support_reasoning_effort:
            kwargs['reasoning_effort'] = 'high'
        if is_search_model:
            kwargs['web_search_options'] = {'search_context_size': 'high'}
        logging.info('Request (chat_id=%r, msg_id=%r, task_id=%r, model=%r, args=%r): %s', chat_id, msg_id, task_id, model, kwargs, remove_blobs(messages))
        kwargs['messages'] = messages
        stream = await aclient.chat.completions.create(**kwargs)
        if not support_stream:
            async def to_aiter(x):
                yield x
            stream = to_aiter(stream)

        finished = False
        async for response in stream:
            logging.info('Response (chat_id=%r, msg_id=%r, task_id=%r): %s', chat_id, msg_id, task_id, response)
            if model in PRICING and response.usage is not None:
                usage_text = f"Prompt tokens: {response.usage.prompt_tokens}\n"
                cached_prompt_tokens = 0
                if response.usage.prompt_tokens_details is not None:
                    if response.usage.prompt_tokens_details.cached_tokens is not None:
                        if response.usage.prompt_tokens_details.cached_tokens > 0:
                            cached_prompt_tokens = response.usage.prompt_tokens_details.cached_tokens
                            usage_text += f"Cached prompt tokens: {cached_prompt_tokens}\n"
                if response.usage.completion_tokens_details is not None:
                    if response.usage.completion_tokens_details.reasoning_tokens is not None:
                        if response.usage.completion_tokens_details.reasoning_tokens > 0:
                            usage_text += f"Reasoning tokens: {response.usage.completion_tokens_details.reasoning_tokens}\n"
                usage_text += f"Completion tokens: {response.usage.completion_tokens}\n"
                input_price, output_price, cached_price, always_show_cost = PRICING[model]
                cost = ((response.usage.prompt_tokens - cached_prompt_tokens) * input_price + response.usage.completion_tokens * output_price + cached_prompt_tokens * cached_price) / 1e6
                usage_text += f"Cost: ${cost:.2f}\n"
                if always_show_cost or cost >= SHOW_COST_THRESHOLD:
                    yield {'type': 'info', 'text': usage_text}
            if finished:
                assert len(response.choices) == 0
                continue
            obj = response.choices[0]
            if hasattr(obj, 'delta'):
                delta = obj.delta
            else:
                delta = obj.message
            if delta.role is not None:
                if delta.role != 'assistant':
                    raise ValueError("Role error")
            if delta.content is not None:
                yield {'type': 'text', 'text': delta.content}
            if obj.finish_reason is not None or ('finish_details' in obj.model_extra and obj.finish_details is not None):
                if hasattr(obj, 'delta'):
                    assert all(item is None for item in [
                        delta.content,
                        delta.function_call,
                        delta.role,
                        delta.tool_calls,
                    ])
                finish_reason = obj.finish_reason
                if 'finish_details' in obj.model_extra and obj.finish_details is not None:
                    assert finish_reason is None
                    finish_reason = obj.finish_details['type']
                if finish_reason == 'length':
                    yield {'type': 'error', 'text': '[!] Error: Output truncated due to limit\n'}
                elif finish_reason == 'stop':
                    pass
                elif finish_reason is not None:
                    if obj.finish_reason is not None:
                        yield {'type': 'error', 'text': f'[!] Error: finish_reason="{finish_reason}"\n'}
                    else:
                        yield {'type': 'error', 'text': f'[!] Error: finish_details="{obj.finish_details}"\n'}
                finished = True
    else:
        kwargs = {'model': model}
        kwargs['stream'] = True
        if is_reasoning_model:
            kwargs['timeout'] = httpx.Timeout(timeout=3600, connect=15)
        if support_reasoning_effort:
            if 'reasoning' not in kwargs:
                kwargs['reasoning'] = {}
            kwargs['reasoning']['effort'] = 'high'
        if support_reasoning_summary:
            if 'reasoning' not in kwargs:
                kwargs['reasoning'] = {}
            kwargs['reasoning']['summary'] = 'detailed'
        logging.info('Request (chat_id=%r, msg_id=%r, task_id=%r, model=%r, args=%r): %s', chat_id, msg_id, task_id, model, kwargs, remove_blobs(messages))
        input_, instructions = convert_to_responses_api_input_format(messages)
        kwargs['input'] = input_
        if instructions is not None:
            kwargs['instructions'] = instructions
        stream = await aclient.responses.create(**kwargs)

        async for response in stream:
            logging.info('Response (chat_id=%r, msg_id=%r, task_id=%r): %s', chat_id, msg_id, task_id, response)
            if response.type == 'response.created':
                pass
            elif response.type == 'response.in_progress':
                pass
            elif response.type == 'response.output_item.added':
                pass
            elif response.type == 'response.output_item.done':
                pass
            elif response.type == 'response.content_part.added':
                assert response.part.type == 'output_text'
            elif response.type == 'response.content_part.done':
                pass
            elif response.type == 'response.completed':
                if model in PRICING:
                    input_tokens = response.response.usage.input_tokens
                    cached_tokens = response.response.usage.input_tokens_details.cached_tokens
                    output_tokens = response.response.usage.output_tokens
                    reasoning_tokens = response.response.usage.output_tokens_details.reasoning_tokens
                    usage_text = f"Input tokens: {input_tokens}\n"
                    if cached_tokens > 0:
                        usage_text += f"Cached tokens: {cached_tokens}\n"
                    usage_text += f"Output tokens: {output_tokens}\n"
                    if reasoning_tokens > 0:
                        usage_text += f"Reasoning tokens: {reasoning_tokens}\n"
                    input_price, output_price, cached_price, always_show_cost = PRICING[model]
                    cost = ((input_tokens - cached_tokens) * input_price + output_tokens * output_price + cached_tokens * cached_price) / 1e6
                    usage_text += f"Cost: ${cost:.2f}\n"
                    if always_show_cost or cost >= SHOW_COST_THRESHOLD:
                        yield {'type': 'info', 'text': usage_text}
            elif response.type == 'response.failed':
                error_code = response.response.error.code
                error_message = response.response.error.message
                yield {'type': 'error', 'text': f'[!] Error: {error_code}: {error_message}\n'}
            elif response.type == 'response.incomplete':
                yield {'type': 'error', 'text': f'[!] Error: Incomplete response: {response.response.incomplete_details.reason}\n'}
            elif response.type == 'error':
                yield {'type': 'error', 'text': f'[!] Error: {response}\n'}
            elif response.type == 'response.output_text.delta':
                yield {'type': 'text', 'text': response.delta}
            elif response.type == 'response.output_text.done':
                pass
            elif response.type ==  'response.reasoning_summary_part.added':
                assert response.part['type'] == 'summary_text'
            elif response.type == 'response.reasoning_summary_text.delta':
                yield {'type': 'reasoning', 'text': response.delta}
            elif response.type == 'response.reasoning_summary_text.done':
                pass
            elif response.type == 'response.reasoning_summary_part.done':
                yield {'type': 'reasoning_delimiter'}
            else:
                raise ValueError(f"Unknown response type: {response.type}")

def construct_chat_history(chat_id, msg_id):
    messages = []
    should_be_bot = False
    model = DEFAULT_MODEL
    has_image = False
    while True:
        key = repr((chat_id, msg_id))
        if key not in db:
            logging.error('History message not found (chat_id=%r, msg_id=%r)', chat_id, msg_id)
            return None, None
        is_bot, message, reply_id, *params = db[key]
        if params:
            if params[0] is not None:
                model = params[0]
        if is_bot != should_be_bot:
            logging.error('Role does not match (chat_id=%r, msg_id=%r)', chat_id, msg_id)
            return None, None
        if isinstance(message, list):
            new_message = []
            for obj in message:
                if obj['type'] == 'text':
                    new_message.append(obj)
                elif obj['type'] == 'image':
                    blob = load_photo(obj['hash'])
                    blob_base64 = base64.b64encode(blob).decode()
                    image_url = 'data:image/jpeg;base64,' + blob_base64
                    new_message.append({'type': 'image_url', 'image_url': {'url': image_url, 'detail': 'high'}})
                    has_image = True
                elif obj['type'] == 'file':
                    blob = load_file(obj['file']['hash'])
                    blob_base64 = base64.b64encode(blob).decode()
                    file_url = 'data:application/pdf;base64,' + blob_base64
                    new_message.append({'type': 'file', 'file': {'filename': obj['file']['filename'], 'file_data': file_url}})
                else:
                    raise ValueError('Unknown message type in chat history')
            message = new_message
        messages.append(message)
        should_be_bot = not should_be_bot
        if reply_id is None:
            break
        msg_id = reply_id
    if len(messages) % 2 != 1:
        logging.error('First message not from user (chat_id=%r, msg_id=%r)', chat_id, msg_id)
        return None, None
    return messages[::-1], model

@only_admin
async def add_whitelist_handler(message):
    if is_whitelist(message.chat_id):
        await send_message(message.chat_id, 'Already in whitelist', message.id)
        return
    add_whitelist(message.chat_id)
    await send_message(message.chat_id, 'Whitelist added', message.id)

@only_admin
async def del_whitelist_handler(message):
    if not is_whitelist(message.chat_id):
        await send_message(message.chat_id, 'Not in whitelist', message.id)
        return
    del_whitelist(message.chat_id)
    await send_message(message.chat_id, 'Whitelist deleted', message.id)

@only_admin
@only_private
async def get_whitelist_handler(message):
    await send_message(message.chat_id, str(get_whitelist()), message.id)

@only_whitelist
async def list_models_handler(message):
    text = ''
    for m in MODELS:
        text += 'Prefix: "' + RichText.Code(m["prefix"]) + '", model: ' + RichText.Code(m["model"]) + '\n'
    await send_message(message.chat_id, text, message.id)

class BotReplyMessages:
    def __init__(self, chat_id, orig_msg_id, prefix):
        self.prefix = prefix
        self.msg_len = TELEGRAM_LENGTH_LIMIT - len(prefix)
        assert self.msg_len > 0
        self.chat_id = chat_id
        self.orig_msg_id = orig_msg_id
        self.replied_msgs = []
        self.text = ''
        self.pending_text = None
        self.timer_task = None
        self.update_task = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, type, value, tb):
        await self.finalize()
        for msg_id, _ in self.replied_msgs:
            pending_reply_manager.remove((self.chat_id, msg_id))

    async def _update(self, session, text):
        slices = []
        while len(text) > self.msg_len:
            slices.append(text[:self.msg_len])
            text = text[self.msg_len:]
        if text:
            slices.append(text)
        if not slices:
            slices = [''] # deal with empty message

        for i in range(min(len(slices), len(self.replied_msgs))):
            msg_id, msg_text = self.replied_msgs[i]
            if slices[i] != msg_text:
                await session.edit_message(self.prefix + slices[i], msg_id)
                self.replied_msgs[i] = (msg_id, slices[i])
        if len(slices) > len(self.replied_msgs):
            for i in range(len(self.replied_msgs), len(slices)):
                if i == 0:
                    reply_to = self.orig_msg_id
                else:
                    reply_to, _ = self.replied_msgs[i - 1]
                msg_id = await session.send_message(self.prefix + slices[i], reply_to)
                self.replied_msgs.append((msg_id, slices[i]))
                pending_reply_manager.add((self.chat_id, msg_id))
        if len(self.replied_msgs) > len(slices):
            for i in range(len(slices), len(self.replied_msgs)):
                msg_id, _ = self.replied_msgs[i]
                await session.delete_message(msg_id)
                pending_reply_manager.remove((self.chat_id, msg_id))
            self.replied_msgs = self.replied_msgs[:len(slices)]

    async def _process_updates(self):
        try:
            while self.pending_text is not None:
                async with rate_limit_manager.session(self.chat_id) as s:
                    pending_text = self.pending_text
                    self.pending_text = None
                    await self._update(s, pending_text)
        finally:
            self.update_task = None

    async def _timer(self):
        await asyncio.sleep(SEND_BATCH_DELAY)
        self.pending_text = self.text
        if self.update_task is None:
            self.update_task = asyncio.create_task(self._process_updates())
        self.timer_task = None

    def update(self, text):
        self.text = text
        if self.timer_task is None:
            self.timer_task = asyncio.create_task(self._timer())

    async def finalize(self):
        if self.timer_task is not None:
            self.timer_task.cancel()
            self.timer_task = None
        self.pending_text = self.text
        if self.update_task is None:
            self.update_task = asyncio.create_task(self._process_updates())
        while self.timer_task is not None or self.update_task is not None:
            if self.timer_task is not None:
                await self.timer_task
            if self.update_task is not None:
                await self.update_task

@only_whitelist
async def reply_handler(message):
    chat_id = message.chat_id
    sender_id = message.sender_id
    msg_id = message.id
    text = message.message
    logging.info('New message: chat_id=%r, sender_id=%r, msg_id=%r, text=%r, photo=%s, document=%s', chat_id, sender_id, msg_id, text, message.photo, message.document)
    reply_to_id = None
    models = None
    extra_photo_message = None
    extra_document_message = None
    if not text and message.photo is None and message.document is None: # unknown media types
        return
    if message.is_reply:
        if message.reply_to.quote_text is not None:
            return
        reply_to_message = await message.get_reply_message()
        if reply_to_message.sender_id == bot_id: # user reply to bot message
            reply_to_id = message.reply_to.reply_to_msg_id
            await pending_reply_manager.wait_for((chat_id, reply_to_id))
        elif reply_to_message.photo is not None: # user reply to a photo
            extra_photo_message = reply_to_message
        elif reply_to_message.document is not None: # user reply to a document
            extra_document_message = reply_to_message
        else:
            return
    if not message.is_reply or extra_photo_message is not None or extra_document_message is not None: # new message
        if '$' not in text:
            if chat_id == sender_id: # if in private chat, send hint
                await send_message(chat_id, '[!] Error: Please start a new conversation with $ or reply to a bot message', msg_id)
            return
        prefix, text = text.split('$', 1)
        if '\n' in prefix:
            if chat_id == sender_id:
                await send_message(chat_id, '[!] Error: Please start a new conversation with $ or reply to a bot message', msg_id)
            return
        triggers = prefix.split(',')
        models = []
        for t in triggers:
            for m in MODELS:
                if m['prefix'] == t.strip() + '$':
                    models.append(m['model'])
                    break
        if models and len(triggers) > TRIGGERS_LIMIT:
            await send_message(chat_id, f'[!] Error: Too many triggers (limit: {TRIGGERS_LIMIT})', msg_id)
            return
        if chat_id == sender_id and len(models) != len(triggers):
            await send_message(chat_id, '[!] Error: Unknown trigger in prefix', msg_id)
            return
        if not models:
            return

    photo_message = message if message.photo is not None else extra_photo_message
    photo_hash = None
    if photo_message is not None:
        if photo_message.grouped_id is not None:
            await send_message(chat_id, '[!] Error: Grouped photos are not yet supported, but will be supported soon', msg_id)
            return
        photo_blob = await photo_message.download_media(bytes)
        photo_hash = save_photo(photo_blob)

    document_message = message if message.document is not None else extra_document_message
    document_text = None
    file_hash = None
    file_name = None
    if document_message is not None:
        if document_message.grouped_id is not None:
            await send_message(chat_id, '[!] Error: Grouped files are not yet supported, but will be supported soon', msg_id)
            return
        if document_message.document.mime_type == 'application/pdf':
            if document_message.document.size > PDF_FILE_SIZE_LIMIT:
                await send_message(chat_id, '[!] Error: PDF file too large', msg_id)
                return
            document_blob = await document_message.download_media(bytes)
            file_hash = save_file(document_blob)
            file_name = document_message.document.attributes[0].file_name
        elif document_message.document.mime_type.startswith('text/'):
            if document_message.document.size > TEXT_FILE_SIZE_LIMIT:
                await send_message(chat_id, '[!] Error: Text file too large', msg_id)
                return
            document_blob = await document_message.download_media(bytes)
            try:
                document_text = document_blob.decode()
                assert all(c != '\x00' for c in document_text)
            except:
                await send_message(chat_id, '[!] Error: Text file is not valid UTF-8', msg_id)
                return
        else:
            await send_message(chat_id, '[!] Error: Unknown file type', msg_id)
            return

    if photo_hash:
        new_message = [{'type': 'image', 'hash': photo_hash}]
        if text:
            new_message.append({'type': 'text', 'text': text})
    elif document_text:
        if text:
            new_message = document_text + '\n\n' + text
        else:
            new_message = document_text
    elif file_hash:
        new_message = [{'type': 'file', 'file': {'filename': file_name, 'hash': file_hash}}]
        if text:
            new_message.append({'type': 'text', 'text': text})
    else:
        new_message = text

    db[repr((chat_id, msg_id))] = (False, new_message, reply_to_id, None)

    chat_history, model = construct_chat_history(chat_id, msg_id)
    if chat_history is None:
        await send_message(chat_id, f"[!] Error: Unable to proceed with this conversation. Potential causes: the message replied to may be incomplete, contain an error, be a system message, or not exist in the database.", msg_id)
        return

    models = models if models is not None else [model]
    async with asyncio.TaskGroup() as tg:
        for task_id, m in enumerate(models):
            tg.create_task(process_request(chat_id, msg_id, chat_history, m, task_id))

def render_reply(reply, info, error, reasoning, is_generating):
    result = ''
    for part in reasoning:
        if part:
            result += RichText.Quote(part, not is_generating) + '\n'
    result += RichText.from_markdown(reply)
    if info:
        result += '\n' + RichText.Quote(info)
    if error:
        result += '\n' + RichText.Quote(RichText.Bold(error))
    if is_generating:
        result += '\n' + RichText.Italic('[!Generating...]')
    return result

async def process_request(chat_id, msg_id, chat_history, model, task_id):
    error_cnt = 0
    while True:
        reply = ''
        info = ''
        error = ''
        reasoning = []
        async with BotReplyMessages(chat_id, msg_id, f'[{model}] ') as replymsgs:
            try:
                replymsgs.update(render_reply(reply, info, error, reasoning, True))
                stream = completion(chat_history, model, chat_id, msg_id, task_id)
                async for delta in stream:
                    if delta['type'] == 'text':
                        reply += delta['text']
                    elif delta['type'] == 'error':
                        error += delta['text']
                    elif delta['type'] == 'info':
                        info += delta['text']
                    elif delta['type'] == 'reasoning':
                        if not reasoning:
                            reasoning.append('')
                        reasoning[-1] += delta['text']
                    elif delta['type'] == 'reasoning_delimiter':
                        if reasoning and reasoning[-1]:
                            reasoning.append('')
                    replymsgs.update(render_reply(reply, info, error, reasoning, True))
                replymsgs.update(render_reply(reply, info, error, reasoning, False))
                await replymsgs.finalize()
                for message_id, _ in replymsgs.replied_msgs:
                    db[repr((chat_id, message_id))] = (True, reply, msg_id, model)
                return
            except Exception as e:
                error_cnt += 1
                logging.exception('Error (chat_id=%r, msg_id=%r, model=%r, task_id=%r, cnt=%r): %s', chat_id, msg_id, model, task_id, error_cnt, e)
                will_retry = not isinstance (e, openai.BadRequestError) and error_cnt <= OPENAI_MAX_RETRY
                error += f'[!] Error: {traceback.format_exception_only(e)[-1].strip()}\n'
                if will_retry:
                    error += f'Retrying ({error_cnt}/{OPENAI_MAX_RETRY})...\n'
                replymsgs.update(render_reply(reply, info, error, reasoning, False))
                if will_retry:
                    await asyncio.sleep(OPENAI_RETRY_INTERVAL)
                if not will_retry:
                    break

async def ping(message):
    await send_message(message.chat_id, f'chat_id={message.chat_id} user_id={message.sender_id} is_whitelisted={is_whitelist(message.chat_id)}', message.id)

async def main():
    global bot_id, pending_reply_manager, rate_limit_manager, db, bot

    logFormatter = logging.Formatter("%(asctime)s %(process)d %(levelname)s %(message)s")

    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(__file__ + ".log")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    with shelve.open('db') as db:
        # db[(chat_id, msg_id)] = (is_bot, text, reply_id, model)
        # compatible old db format: db[(chat_id, msg_id)] = (is_bot, text, reply_id)
        # db['whitelist'] = set(whitelist_chat_ids)
        if 'whitelist' not in db:
            db['whitelist'] = {ADMIN_ID}
        bot_id = int(TELEGRAM_BOT_TOKEN.split(':')[0])
        pending_reply_manager = PendingReplyManager()
        rate_limit_manager = RateLimitManager()
        async with await TelegramClient('bot', TELEGRAM_API_ID, TELEGRAM_API_HASH).start(bot_token=TELEGRAM_BOT_TOKEN) as bot:
            bot.parse_mode = None
            me = await bot.get_me()
            @bot.on(events.NewMessage)
            async def process(event):
                if event.message.chat_id is None:
                    return
                if event.message.sender_id is None or event.message.sender_id == bot_id:
                    return
                if event.message.message is None:
                    return
                text = event.message.message
                if text == '/ping' or text == f'/ping@{me.username}':
                    await ping(event.message)
                elif text == '/list_models' or text == f'/list_models@{me.username}':
                    await list_models_handler(event.message)
                elif text == '/add_whitelist' or text == f'/add_whitelist@{me.username}':
                    await add_whitelist_handler(event.message)
                elif text == '/del_whitelist' or text == f'/del_whitelist@{me.username}':
                    await del_whitelist_handler(event.message)
                elif text == '/get_whitelist' or text == f'/get_whitelist@{me.username}':
                    await get_whitelist_handler(event.message)
                else:
                    await reply_handler(event.message)
            assert await bot(functions.bots.SetBotCommandsRequest(
                scope=types.BotCommandScopeDefault(),
                lang_code='en',
                commands=[types.BotCommand(command, description) for command, description in [
                    ('ping', 'Test bot connectivity'),
                    ('list_models', 'List supported models'),
                    ('add_whitelist', 'Add this group to whitelist (only admin)'),
                    ('del_whitelist', 'Delete this group from whitelist (only admin)'),
                    ('get_whitelist', 'List groups in whitelist (only admin)'),
                ]]
            ))
            await bot.run_until_disconnected()

asyncio.run(main())
