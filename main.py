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
from richtext import RichText, RichTextParts
from google import genai
from google.genai import types as gtypes
from google.genai import errors as gerrors
from telethon import TelegramClient, events, errors, functions, types
import signal
import re
from contextlib import asynccontextmanager
import shutil

def debug_signal_handler(signal, frame):
    breakpoint()

signal.signal(signal.SIGUSR1, debug_signal_handler)

ADMIN_ID = 71863318

MODELS = [
    {'prefix': 'g$', 'model': 'gemini-2.5-pro-preview-05-06'},
    {'prefix': 'gf$', 'model': 'gemini-2.5-flash-preview-05-20'},
    {'prefix': 'g2$', 'model': 'gemini-2.0-pro-exp-02-05'},
    {'prefix': 'g2f$', 'model': 'gemini-2.0-flash'},
    {'prefix': 'gfl$', 'model': 'gemini-2.0-flash-lite'},
    {'prefix': 'g15$', 'model': 'gemini-1.5-pro-latest'},
    {'prefix': 'g1$', 'model': 'gemini-1.0-pro-latest', 'vision_model': 'gemini-pro-vision'},
    {'prefix': 'gt$', 'model': 'gemini-2.0-flash-thinking-exp-01-21'},
    {'prefix': 'ge$', 'model': 'gemma-3-27b-it'},
    {'prefix': 'gi$', 'model': 'gemini-2.0-flash-exp-image-generation'},

    {'prefix': 'gemini-2.5-pro-preview-05-06$', 'model': 'gemini-2.5-pro-preview-05-06'},
    {'prefix': 'gemini-2.5-pro-preview-03-25$', 'model': 'gemini-2.5-pro-preview-03-25'},
    {'prefix': 'gemini-2.5-flash-preview-04-17$', 'model': 'gemini-2.5-flash-preview-04-17'},
    {'prefix': 'gemini-2.5-flash-preview-05-20$', 'model': 'gemini-2.5-flash-preview-05-20'},

    {'prefix': 'gemini-2.0-flash$', 'model': 'gemini-2.0-flash'},
    {'prefix': 'gemini-2.0-flash-lite$', 'model': 'gemini-2.0-flash-lite'},
    {'prefix': 'gemini-2.0-pro-exp-02-05$', 'model': 'gemini-2.0-pro-exp-02-05'},

    {'prefix': 'gemini-exp-1114$', 'model': 'gemini-exp-1114'},
    {'prefix': 'gemini-exp-1121$', 'model': 'gemini-exp-1121'},
    {'prefix': 'gemini-exp-1206$', 'model': 'gemini-exp-1206'},
    {'prefix': 'learnlm-1.5-pro-experimental$', 'model': 'learnlm-1.5-pro-experimental'},

    {'prefix': 'gemini-2.0-flash-exp$', 'model': 'gemini-2.0-flash-exp'},
    {'prefix': 'gemini-2.0-flash-thinking-exp-1219$', 'model': 'gemini-2.0-flash-thinking-exp-1219'},
    {'prefix': 'gemini-2.0-flash-thinking-exp-01-21$', 'model': 'gemini-2.0-flash-thinking-exp-01-21'},

    {'prefix': 'gemini-1.5-pro-latest$', 'model': 'gemini-1.5-pro-latest'},
    {'prefix': 'gemini-1.5-pro$', 'model': 'gemini-1.5-pro'},
    {'prefix': 'gemini-1.5-pro-001$', 'model': 'gemini-1.5-pro-001'},
    {'prefix': 'gemini-1.5-pro-002$', 'model': 'gemini-1.5-pro-002'},
    {'prefix': 'gemini-1.5-pro-exp-0801$', 'model': 'gemini-1.5-pro-exp-0801'},
    {'prefix': 'gemini-1.5-pro-exp-0827$', 'model': 'gemini-1.5-pro-exp-0827'},

    {'prefix': 'gemini-1.5-flash-latest$', 'model': 'gemini-1.5-flash-latest'},
    {'prefix': 'gemini-1.5-flash$', 'model': 'gemini-1.5-flash'},
    {'prefix': 'gemini-1.5-flash-001$', 'model': 'gemini-1.5-flash-001'},
    {'prefix': 'gemini-1.5-flash-002$', 'model': 'gemini-1.5-flash-002'},
    {'prefix': 'gemini-1.5-flash-exp-0827$', 'model': 'gemini-1.5-flash-exp-0827'},
    {'prefix': 'gemini-1.5-flash-8b-exp-0827$', 'model': 'gemini-1.5-flash-8b-exp-0827'},
    {'prefix': 'gemini-1.5-flash-8b-exp-0924$', 'model': 'gemini-1.5-flash-8b-exp-0924'},

    {'prefix': 'gemini-1.0-pro-latest$', 'model': 'gemini-1.0-pro-latest'},
    {'prefix': 'gemini-1.0-pro$', 'model': 'gemini-1.0-pro'},
    {'prefix': 'gemini-1.0-pro-001$', 'model': 'gemini-1.0-pro-001'},
]
DEFAULT_MODEL = 'gemini-1.5-pro-latest' # For compatibility with the old database format

def PRICING(model, input_tokens, output_tokens, audio_tokens):
    if model.startswith('gemini-2.5-pro-preview-'):
        if input_tokens <= 200_000: # exact conditions is not sure
            return 1.25e-6 * input_tokens + 10e-6 * output_tokens
        else:
            return 2.5e-6 * input_tokens + 15e-6 * output_tokens
    elif model.startswith('gemini-2.5-flash-preview-'):
        return 0.15e-6 * (input_tokens - audio_tokens) + 3.5e-6 * output_tokens + 1e-6 * audio_tokens
    elif model == 'gemini-2.0-flash':
        return 0.1e-6 * (input_tokens - audio_tokens) + 0.4e-6 * output_tokens + 0.7e-6 * audio_tokens
    elif model == 'gemini-2.0-flash-lite':
        return 0.075e-6 * input_tokens + 0.3e-6 * output_tokens

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

TELEGRAM_LENGTH_LIMIT = 4096
TELEGRAM_MIN_INTERVAL = 3
OPENAI_MAX_RETRY = 3
OPENAI_RETRY_INTERVAL = 30
SEND_BATCH_DELAY = 1
TEXT_FILE_SIZE_LIMIT = 10_000_000
FILE_SIZE_LIMIT = 32_000_000
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
    async def send_message(self, message, reply_to_message_id):
        logging.info('Sending message: chat_id=%r, reply_to_message_id=%r, text=%r', self.chat_id, reply_to_message_id, message)
        message = RichTextParts(message)
        if len(message.parts) == 1 and message.parts[0]['type'] == 'richtext':
            text, entities = message.parts[0]['content'].to_telegram()
            file = None
        elif len(message.parts) == 2 and message.parts[0]['type'] == 'richtext' and message.parts[1]['type'] == 'image':
            text, entities = message.parts[0]['content'].to_telegram()
            file = load_photo_filename(message.parts[1]['content'])
        else:
            raise ValueError('Invalid format')
        msg = await bot.send_message(
            self.chat_id,
            text,
            reply_to=reply_to_message_id,
            link_preview=False,
            formatting_entities=entities,
            file=file,
        )
        logging.info('Message sent: chat_id=%r, reply_to_message_id=%r, message_id=%r', self.chat_id, reply_to_message_id, msg.id)
        return msg.id

    @check_alive
    @retry()
    @ensure_interval
    async def edit_message(self, message, message_id):
        logging.info('Editing message: chat_id=%r, message_id=%r, text=%r', self.chat_id, message_id, message)
        message = RichTextParts(message)
        if len(message.parts) == 1 and message.parts[0]['type'] == 'richtext':
            text, entities = message.parts[0]['content'].to_telegram()
            file = None
        elif len(message.parts) == 2 and message.parts[0]['type'] == 'richtext' and message.parts[1]['type'] == 'image':
            text, entities = message.parts[0]['content'].to_telegram()
            file = load_photo_filename(message.parts[1]['content'])
        else:
            raise ValueError('Invalid format')
        # Cannot remove an image using edit_message
        try:
            await bot.edit_message(
                self.chat_id,
                message_id,
                text,
                link_preview=False,
                formatting_entities=entities,
                file=file,
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

def load_photo_filename(h):
    dir = f'photos/{h[:2]}/{h[2:4]}'
    path = f'{dir}/{h}'
    new_path = path + '.png'
    if not os.path.exists(new_path):
        shutil.copy(path, new_path)
    return new_path

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
    tools = ''
    if '+' in model:
        model, tools = model.split('+', 1)
    tools = set(tools)

    assert len(chat_history) % 2 == 1
    messages=[]
    roles = ["user", "model"]
    role_id = 0
    for msg in chat_history:
        messages.append({"role": roles[role_id], "parts": msg})
        role_id = 1 - role_id
    def remove_blobs(messages):
        new_messages = copy.deepcopy(messages)
        for message in new_messages:
            if 'parts' in message:
                if isinstance(message['parts'], list):
                    for obj in message['parts']:
                        if isinstance(obj, dict) and 'data' in obj:
                            obj['data'] = '...'
        return new_messages
    logging.info('Request (chat_id=%r, msg_id=%r, task_id=%r): %s', chat_id, msg_id, task_id, remove_blobs(messages))

    contents = []
    for message in messages:
        parts = []
        for part in message['parts']:
            if isinstance(part, str):
                if message['role'] == 'user':
                    youtube_re = r'@https://(?:(?:(?:www\.|m\.)?youtube\.com)|youtu\.be)/[a-zA-Z0-9./?=&%_~#:-]+'
                    youtube_urls = re.findall(youtube_re, part)
                    for url in youtube_urls:
                        url = url[1:]
                        parts.append(gtypes.Part(file_data=gtypes.FileData(file_uri=url)))
                parts.append(gtypes.Part.from_text(text=part))
            else:
                parts.append(gtypes.Part.from_bytes(data=part['data'], mime_type=part['mime_type']))
        contents.append(gtypes.Content(
            role=message['role'],
            parts=parts,
        ))

    is_reasoning_model = model in [
        'gemini-2.0-flash-thinking-exp-01-21',
        'gemini-2.0-pro-exp-02-05',
        'gemini-2.5-flash-preview-04-17',
        'gemini-2.5-flash-preview-05-20',
        'gemini-2.5-pro-preview-03-25',
        'gemini-2.5-pro-preview-05-06',
    ]
    is_image_generation_model = model == 'gemini-2.0-flash-exp-image-generation'

    config=gtypes.GenerateContentConfig(
        safety_settings=[
            gtypes.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='OFF'),
            gtypes.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='OFF'),
            gtypes.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='OFF'),
            gtypes.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='OFF'),
            gtypes.SafetySetting(category='HARM_CATEGORY_CIVIC_INTEGRITY', threshold='OFF'),
        ],
        # media_resolution='MEDIA_RESOLUTION_HIGH', # Media resolution is not enabled for api version v1beta
        http_options=gtypes.HttpOptions(timeout=600000),
    )

    if is_reasoning_model:
        config.thinking_config = gtypes.ThinkingConfig(include_thoughts=True, thinking_budget=24576)

    if is_image_generation_model:
        config.response_modalities = ["image", "text"]

    config_tools = []
    for tool in tools:
        if tool == 'c':
            config_tools.append(gtypes.Tool(code_execution=gtypes.ToolCodeExecution))
        elif tool == 's':
            config_tools.append(gtypes.Tool(google_search=gtypes.GoogleSearch))
    config.tools = config_tools

    def remove_response_blobs(response):
        if response.candidates is not None and len(response.candidates) == 1:
            obj = response.candidates[0]
            if obj.content is not None and obj.content.parts is not None and len(obj.content.parts) == 1:
                if obj.content.parts[0].inline_data is not None:
                    response_new = response.model_copy(deep=True)
                    response_new.candidates[0].content.parts[0].inline_data = b'...'
                    return response_new
        return response

    stream = await client.aio.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    )
    async for response in stream:
        logging.info('Response (chat_id=%r, msg_id=%r, task_id=%r): %r', chat_id, msg_id, task_id, remove_response_blobs(response))
        response: gtypes.GenerateContentResponse
        if response.candidates is not None:
            assert len(response.candidates) == 1
            obj = response.candidates[0]
            if obj.content is not None:
                if obj.content.role is not None:
                    assert obj.content.role == 'model'
                if obj.content.parts is not None:
                    assert len(obj.content.parts) == 1
                    if obj.content.parts[0].text is not None:
                        if obj.content.parts[0].thought is not None and obj.content.parts[0].thought:
                            yield {'type': 'reasoning', 'text': obj.content.parts[0].text}
                        else:
                            yield {'type': 'text', 'text': obj.content.parts[0].text}
                    assert obj.content.parts[0].video_metadata is None
                    if obj.content.parts[0].code_execution_result is not None:
                        markdown = f'\n```\n{obj.content.parts[0].code_execution_result.output}\n```\n'
                        yield {'type': 'text', 'text': markdown}
                    if obj.content.parts[0].executable_code is not None:
                        code = obj.content.parts[0].executable_code
                        markdown = f'\n```{code.language.lower()}\n{code.code}\n```\n'
                        yield {'type': 'text', 'text': markdown}
                    assert obj.content.parts[0].file_data is None
                    assert obj.content.parts[0].function_call is None
                    assert obj.content.parts[0].function_response is None
                    if obj.content.parts[0].inline_data is not None:
                        yield {'type': 'image', 'data': obj.content.parts[0].inline_data.data}
            # assert obj.citation_metadata is None # TODO: show citations when uploading file
            # assert obj.grounding_metadata is None # TODO: show grounding when using google search
            assert obj.finish_message is None
            if obj.finish_reason is not None:
                if obj.finish_reason == 'STOP':
                    pass
                else:
                    yield {'type': 'error', 'text': f'[!] Error: finish_reason="{obj.finish_reason}"\n'}
        if response.usage_metadata is not None:
            usage = response.usage_metadata
            usage_text = ''
            input_tokens = 0
            output_tokens = 0
            thinking_tokens = 0
            if usage.prompt_token_count is not None:
                usage_text += f'Prompt tokens: {usage.prompt_token_count}\n'
                input_tokens += usage.prompt_token_count
            if usage.tool_use_prompt_token_count is not None:
                usage_text += f'Tool use prompt tokens: {usage.tool_use_prompt_token_count}\n'
                output_tokens += usage.tool_use_prompt_token_count
            if usage.cached_content_token_count is not None:
                usage_text += f'Cached tokens: {usage.cached_content_token_count}\n'
            if usage.thoughts_token_count is not None:
                usage_text += f'Thought tokens: {usage.thoughts_token_count}\n'
                thinking_tokens += usage.thoughts_token_count
            if usage.candidates_token_count is not None:
                usage_text += f'Output tokens: {usage.candidates_token_count}\n'
                output_tokens += usage.candidates_token_count
            audio_tokens = 0
            for mod in usage.prompt_tokens_details:
                if mod.modality == 'AUDIO':
                    audio_tokens += mod.token_count
            if audio_tokens > 0:
                usage_text += f'Audio tokens: {audio_tokens}\n'
            cost = PRICING(model, input_tokens, output_tokens + thinking_tokens, audio_tokens)
            if cost:
                usage_text += f'Cost (Estimated): ${cost:.2f}\n'
            yield {'type': 'info', 'text': usage_text}

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
        assert isinstance(message, list)
        new_message = []
        for obj in message:
            if obj['type'] == 'text':
                new_message.append(obj['text'])
            elif obj['type'] == 'image':
                blob = load_photo(obj['hash'])
                new_message.append({'mime_type': 'image/jpeg', 'data': blob})
                has_image = True
            elif obj['type'] == 'file':
                blob = load_file(obj['file']['hash'])
                new_message.append({'mime_type': obj['file']['mime_type'], 'data': blob})
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
    if has_image:
        for m in MODELS:
            if m['model'] == model and 'vision_model' in m:
                model = m['vision_model']
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
        text += f'Prefix: "{m["prefix"]}", model: {m["model"]}\n'
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
        slices = RichTextParts(text).to_slices(self.msg_len)
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
                elif m['prefix'] == t.strip().split('+', 1)[0] + '$':
                    tools = t.strip().split('+', 1)[1]
                    if not set(tools) <= set('sc') or not tools:
                        await send_message(chat_id, '[!] Error: Unknown tools in prefix', msg_id)
                        return
                    models.append(m['model'] + '+' + tools)
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
    file_mime_type = None
    if document_message is not None:
        if document_message.grouped_id is not None:
            await send_message(chat_id, '[!] Error: Grouped files are not yet supported, but will be supported soon', msg_id)
            return
        if document_message.document.mime_type in [
            'application/pdf',
            'image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif',
            'video/mp4', 'video/mov', 'video/avi', 'video/x-flv', 'video/mpg', 'video/webm', 'video/wmv', 'video/3gpp',
            'audio/mpeg', 'audio/aiff', 'audio/aac', 'audio/ogg', 'audio/flac',
        ]:
            if document_message.document.size > FILE_SIZE_LIMIT:
                await send_message(chat_id, '[!] Error: File too large', msg_id)
                return
            document_blob = await document_message.download_media(bytes)
            file_hash = save_file(document_blob)
            for attr in document_message.document.attributes:
                if isinstance(attr, types.DocumentAttributeFilename):
                    file_name = attr.file_name
                    break
            file_mime_type = document_message.document.mime_type
            if file_mime_type == 'audio/mpeg':
                file_mime_type = 'audio/mp3' # fix Gemini mp3 mime type
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
            new_message = [{'type': 'text', 'text': document_text + '\n\n' + text}]
        else:
            new_message = [{'type': 'text', 'text': document_text}]
    elif file_hash:
        new_message = [{'type': 'file', 'file': {'filename': file_name, 'hash': file_hash, 'mime_type': file_mime_type}}]
        if text:
            new_message.append({'type': 'text', 'text': text})
    else:
        new_message = [{'type': 'text', 'text': text}]

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
    result = RichTextParts()
    for part in reply:
        if part['type'] == 'text':
            result += RichText.from_markdown(part['text'])
        elif part['type'] == 'image':
            result += RichTextParts.Image(part['hash'])
        else:
            raise ValueError(f"Unknown type: {part['type']}")
    if reasoning:
        result = RichText.Quote(reasoning.strip(), not is_generating) + '\n' + result
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
        reply = []
        info = ''
        error = ''
        reasoning = ''
        async with BotReplyMessages(chat_id, msg_id, f'[{model}] ') as replymsgs:
            try:
                replymsgs.update(render_reply(reply, info, error, reasoning, True))
                stream = completion(chat_history, model, chat_id, msg_id, task_id)
                async for delta in stream:
                    if delta['type'] == 'text':
                        if delta['text'] == '':
                            continue
                        if reply and reply[-1]['type'] == 'text':
                            reply[-1]['text'] += delta['text']
                        else:
                            reply.append({'type': 'text', 'text': delta['text']})
                    elif delta['type'] == 'image':
                        photo_hash = save_photo(delta['data'])
                        reply.append({'type': 'image', 'hash': photo_hash})
                    elif delta['type'] == 'error':
                        error += delta['text']
                    elif delta['type'] == 'info':
                        info = delta['text']
                    elif delta['type'] == 'reasoning':
                        reasoning += delta['text']
                    replymsgs.update(render_reply(reply, info, error, reasoning, True))
                replymsgs.update(render_reply(reply, info, error, reasoning, False))
                await replymsgs.finalize()
                for message_id, _ in replymsgs.replied_msgs:
                    db[repr((chat_id, message_id))] = (True, reply, msg_id, model)
                return
            except Exception as e:
                error_cnt += 1
                logging.exception('Error (chat_id=%r, msg_id=%r, model=%r, task_id=%r, cnt=%r): %s', chat_id, msg_id, model, task_id, error_cnt, e)
                will_retry = not isinstance (e, gerrors.ClientError) and error_cnt <= OPENAI_MAX_RETRY
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
