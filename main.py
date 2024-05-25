import asyncio
import os
import logging
import shelve
import time
import traceback
import hashlib
import base64
import copy
from collections import defaultdict
from richtext import RichText
from anthropic import AsyncAnthropic
from telethon import TelegramClient, events, errors, functions, types
import signal

def debug_signal_handler(signal, frame):
    breakpoint()

signal.signal(signal.SIGUSR1, debug_signal_handler)

ADMIN_ID = 71863318

MODELS = [
    {'prefix': 'cs$$$', 'model': 'claude-3-haiku-20240307'},
    {'prefix': 'cs$$', 'model': 'claude-3-sonnet-20240229'},
    {'prefix': 'cs$', 'model': 'claude-3-opus-20240229'},
]

aclient = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_retries=0,
    timeout=15,
)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

TELEGRAM_LENGTH_LIMIT = 4096
TELEGRAM_MIN_INTERVAL = 3
OPENAI_MAX_RETRY = 3
OPENAI_RETRY_INTERVAL = 10
FIRST_BATCH_DELAY = 1
TEXT_FILE_SIZE_LIMIT = 100_000

telegram_last_timestamp = defaultdict(lambda: None)
telegram_rate_limit_lock = defaultdict(asyncio.Lock)

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

def within_interval(chat_id):
    global telegram_last_timestamp
    if telegram_last_timestamp[chat_id] is None:
        return False
    remaining_time = telegram_last_timestamp[chat_id] + TELEGRAM_MIN_INTERVAL - time.time()
    return remaining_time > 0

def ensure_interval(interval=TELEGRAM_MIN_INTERVAL):
    def decorator(func):
        async def new_func(*args, **kwargs):
            chat_id = args[0]
            async with telegram_rate_limit_lock[chat_id]:
                global telegram_last_timestamp
                if telegram_last_timestamp[chat_id] is not None:
                    remaining_time = telegram_last_timestamp[chat_id] + interval - time.time()
                    if remaining_time > 0:
                        await asyncio.sleep(remaining_time)
                result = await func(*args, **kwargs)
                telegram_last_timestamp[chat_id] = time.time()
                return result
        return new_func
    return decorator

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

async def completion(messages, model, chat_id, msg_id, system_prompt):
    def remove_image(messages):
        new_messages = copy.deepcopy(messages)
        for message in new_messages:
            if 'content' in message:
                if isinstance(message['content'], list):
                    for obj in message['content']:
                        if obj['type'] == 'image':
                            obj['source']['data'] = obj['source']['data'][:50] + '...'
        return new_messages
    logging.info('Request (chat_id=%r, msg_id=%r, system_prompt=%r): %s', chat_id, msg_id, system_prompt, remove_image(messages))
    stream = await aclient.messages.create(model=model, messages=messages, stream=True, max_tokens=4096, system=system_prompt)
    async for event in stream:
        logging.info('Response (chat_id=%r, msg_id=%r): %s', chat_id, msg_id, event)
        if event.type == 'message_start':
            assert event.message.role == 'assistant'
            assert event.message.content == []
            assert event.message.stop_reason is None
        elif event.type == 'content_block_delta':
            assert event.index == 0
            assert event.delta.type == 'text_delta'
            yield event.delta.text
        elif event.type == 'content_block_start':
            assert event.index == 0
            assert event.content_block.text == ''
        elif event.type == 'content_block_stop':
            assert event.index == 0
        elif event.type == 'message_delta':
            stop_reason = event.delta.stop_reason
            if stop_reason is not None:
                if stop_reason == 'end_turn':
                    pass
                else:
                    yield f'\n\n[!] Error: stop_reason="{stop_reason}"'

def construct_chat_history(chat_id, msg_id):
    model = None
    messages = []
    system_prompt = ''
    while True:
        key = repr((chat_id, msg_id))
        if key not in db:
            logging.error('History message not found (chat_id=%r, msg_id=%r)', chat_id, msg_id)
            return None, None
        msgs, reply_id = db[key]
        new_messages = []
        for i in msgs:
            if "model" in i:
                model = i["model"]
            elif i['role'] == 'system':
                system_prompt = i['content']
            else:
                message = i['content']
                if isinstance(message, list):
                    new_message = []
                    for obj in message:
                        if obj['type'] == 'text':
                            new_message.append(obj)
                        elif obj['type'] == 'image':
                            blob = load_photo(obj['hash'])
                            blob_base64 = base64.b64encode(blob).decode()
                            new_message.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': blob_base64}})
                        else:
                            raise ValueError('Unknown message type in chat history')
                    message = new_message
                new_messages.append({"role": i["role"], "content": message})
        messages = new_messages + messages
        if reply_id is None:
            break
        msg_id = reply_id
    if model is None:
        return None, None, None

    # merge messages of the same role, supporting https://docs.anthropic.com/en/docs/prefill-claudes-response
    new_messages = []
    for i in messages:
        if new_messages and i["role"] == new_messages[-1]["role"] and isinstance(i["content"], str) and isinstance(new_messages[-1]["content"], str):
            new_messages[-1]["content"] += i["content"]
        else:
            new_messages.append(i)
    messages = new_messages

    return messages, model, system_prompt

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

@retry()
@ensure_interval()
async def send_message(chat_id, text, reply_to_message_id):
    logging.info('Sending message: chat_id=%r, reply_to_message_id=%r, text=%r', chat_id, reply_to_message_id, text)
    text = RichText(text)
    text, entities = text.to_telegram()
    msg = await bot.send_message(
        chat_id,
        text,
        reply_to=reply_to_message_id,
        link_preview=False,
        formatting_entities=entities,
    )
    logging.info('Message sent: chat_id=%r, reply_to_message_id=%r, message_id=%r', chat_id, reply_to_message_id, msg.id)
    return msg.id

@retry()
@ensure_interval()
async def edit_message(chat_id, text, message_id):
    logging.info('Editing message: chat_id=%r, message_id=%r, text=%r', chat_id, message_id, text)
    text = RichText(text)
    text, entities = text.to_telegram()
    try:
        await bot.edit_message(
            chat_id,
            message_id,
            text,
            link_preview=False,
            formatting_entities=entities,
        )
    except errors.MessageNotModifiedError as e:
        logging.info('Message not modified: chat_id=%r, message_id=%r', chat_id, message_id)
    else:
        logging.info('Message edited: chat_id=%r, message_id=%r', chat_id, message_id)

@retry()
@ensure_interval()
async def delete_message(chat_id, message_id):
    logging.info('Deleting message: chat_id=%r, message_id=%r', chat_id, message_id)
    await bot.delete_messages(
        chat_id,
        message_id,
    )
    logging.info('Message deleted: chat_id=%r, message_id=%r', chat_id, message_id)

class BotReplyMessages:
    def __init__(self, chat_id, orig_msg_id, prefix):
        self.prefix = prefix
        self.msg_len = TELEGRAM_LENGTH_LIMIT - len(prefix)
        assert self.msg_len > 0
        self.chat_id = chat_id
        self.orig_msg_id = orig_msg_id
        self.replied_msgs = []
        self.text = ''

    async def __aenter__(self):
        return self

    async def __aexit__(self, type, value, tb):
        await self.finalize()
        for msg_id, _ in self.replied_msgs:
            pending_reply_manager.remove((self.chat_id, msg_id))

    async def _force_update(self, text):
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
                await edit_message(self.chat_id, self.prefix + slices[i], msg_id)
                self.replied_msgs[i] = (msg_id, slices[i])
        if len(slices) > len(self.replied_msgs):
            for i in range(len(self.replied_msgs), len(slices)):
                if i == 0:
                    reply_to = self.orig_msg_id
                else:
                    reply_to, _ = self.replied_msgs[i - 1]
                msg_id = await send_message(self.chat_id, self.prefix + slices[i], reply_to)
                self.replied_msgs.append((msg_id, slices[i]))
                pending_reply_manager.add((self.chat_id, msg_id))
        if len(self.replied_msgs) > len(slices):
            for i in range(len(slices), len(self.replied_msgs)):
                msg_id, _ = self.replied_msgs[i]
                await delete_message(self.chat_id, msg_id)
                pending_reply_manager.remove((self.chat_id, msg_id))
            self.replied_msgs = self.replied_msgs[:len(slices)]

    async def update(self, text):
        self.text = text
        if not within_interval(self.chat_id):
            await self._force_update(self.text)

    async def finalize(self):
        await self._force_update(self.text)

@only_whitelist
async def reply_handler(message):
    chat_id = message.chat_id
    sender_id = message.sender_id
    msg_id = message.id
    text = message.message
    logging.info('New message: chat_id=%r, sender_id=%r, msg_id=%r, text=%r, photo=%s, document=%s', chat_id, sender_id, msg_id, text, message.photo, message.document)
    reply_to_id = None
    extra_photo_message = None
    extra_document_message = None
    if not text and message.photo is None and message.document is None: # unknown media types
        return
    new_messages = None
    if message.is_reply:
        if message.reply_to.quote_text is not None:
            return
        reply_to_message = await message.get_reply_message()
        if reply_to_message.sender_id == bot_id: # user reply to bot message
            reply_to_id = message.reply_to.reply_to_msg_id
            msgs = []
            splits = text.split('$$$')
            for i, text in enumerate(splits):
                if i % 2 == 0:
                    role = 'user'
                else:
                    role = 'assistant'
                msgs.append({"role": role, "content": text})
            await pending_reply_manager.wait_for((chat_id, reply_to_id))
            new_messages = msgs
        elif reply_to_message.photo is not None: # user reply to a photo
            extra_photo_message = reply_to_message
        elif reply_to_message.document is not None: # user reply to a document
            extra_document_message = reply_to_message
        else:
            return
    if not message.is_reply or extra_photo_message is not None or extra_document_message is not None: # new message
        matched_prefix = None
        for m in MODELS:
            if text.startswith(m['prefix']):
                text = text[len(m['prefix']):]
                model = m['model']
                matched_prefix = m['prefix']
                break
        else: # not reply or new message to bot
            if chat_id == sender_id: # if in private chat, send hint
                await send_message(chat_id, 'Please start a new conversation with $ or reply to a bot message', msg_id)
            return
        msgs = [{"model": model}]
        splits = text.split('$$$')
        if len(splits) < 2:
            await send_message(chat_id, f'Usage: {matched_prefix}system$$$user$$$assistant$$$...$$$user', msg_id)
            return
        for i, text in enumerate(splits):
            if i == 0:
                role = 'system'
            elif i % 2:
                role = 'user'
            else:
                role = 'assistant'
            msgs.append({"role": role, "content": text})
        new_messages = msgs

    photo_message = message if message.photo is not None else extra_photo_message
    photo_hash = None
    if photo_message is not None:
        if photo_message.grouped_id is not None:
            await send_message(chat_id, 'Grouped photos are not yet supported, but will be supported soon', msg_id)
            return
        photo_blob = await photo_message.download_media(bytes)
        photo_hash = save_photo(photo_blob)

    document_message = message if message.document is not None else extra_document_message
    document_text = None
    if document_message is not None:
        if document_message.grouped_id is not None:
            await send_message(chat_id, 'Grouped files are not yet supported, but will be supported soon', msg_id)
            return
        if document_message.document.size > TEXT_FILE_SIZE_LIMIT:
            await send_message(chat_id, 'File too large', msg_id)
            return
        document_blob = await document_message.download_media(bytes)
        try:
            document_text = document_blob.decode()
            assert all(c != '\x00' for c in document_text)
        except:
            await send_message(chat_id, 'File is not text file or not valid UTF-8', msg_id)
            return

    if photo_hash:
        if len(new_messages) > 1:
            await send_message(chat_id, 'Photo not supported in multiple messages', msg_id)
            return
        new_messages[0]['content'] = [{'type': 'text', 'text': new_messages[0]['content']}, {'type': 'image', 'hash': photo_hash}]
    elif document_text:
        if len(new_messages) > 1:
            await send_message(chat_id, 'File not supported in multiple messages', msg_id)
            return
        if new_messages[0]['content']:
            new_messages[0]['content'] = document_text + '\n\n' + new_messages[0]['content']
        else:
            new_messages[0]['content'] = document_text

    for i in new_messages:
        if 'content' in i:
            new_message = i['content']
            if (isinstance(new_message, str) and len(new_message) == 0) or (isinstance(new_message, list) and sum(len(m['text']) for m in new_message if m['type'] == 'text') == 0):
                await send_message(chat_id, f"[!] Error: Input text should not be empty", msg_id)
                return

    db[repr((chat_id, msg_id))] = new_messages, reply_to_id

    chat_history, model, system_prompt = construct_chat_history(chat_id, msg_id)
    if chat_history is None:
        await send_message(chat_id, f"[!] Error: Unable to proceed with this conversation. Potential causes: the message replied to may be incomplete, contain an error, be a system message, or not exist in the database.", msg_id)
        return

    error_cnt = 0
    while True:
        reply = ''
        async with BotReplyMessages(chat_id, msg_id, f'[{model}] ') as replymsgs:
            try:
                stream = completion(chat_history, model, chat_id, msg_id, system_prompt)
                first_update_timestamp = None
                async for delta in stream:
                    reply += delta
                    if first_update_timestamp is None:
                        first_update_timestamp = time.time()
                    if time.time() >= first_update_timestamp + FIRST_BATCH_DELAY:
                        await replymsgs.update(RichText.from_markdown(reply) + ' [!Generating...]')
                await replymsgs.update(RichText.from_markdown(reply))
                await replymsgs.finalize()
                for message_id, _ in replymsgs.replied_msgs:
                    db[repr((chat_id, message_id))] = [{"role": "assistant", "content": reply}], msg_id
                return
            except Exception as e:
                error_cnt += 1
                logging.exception('Error (chat_id=%r, msg_id=%r, cnt=%r): %s', chat_id, msg_id, error_cnt, e)
                will_retry = error_cnt <= OPENAI_MAX_RETRY
                error_msg = f'[!] Error: {traceback.format_exception_only(e)[-1].strip()}'
                if will_retry:
                    error_msg += f'\nRetrying ({error_cnt}/{OPENAI_MAX_RETRY})...'
                if reply:
                    error_msg = reply + '\n\n' + error_msg
                await replymsgs.update(error_msg)
                if will_retry:
                    await asyncio.sleep(OPENAI_RETRY_INTERVAL)
                if not will_retry:
                    break

async def ping(message):
    await send_message(message.chat_id, f'chat_id={message.chat_id} user_id={message.sender_id} is_whitelisted={is_whitelist(message.chat_id)}', message.id)

async def main():
    global bot_id, pending_reply_manager, db, bot

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
        # db[(chat_id, msg_id)] = (msgs, reply_id)
        # msgs = [{"model": "gpt-4"}, {"role": "system/user/assistant", "content": "foo"}, ...]
        # db['whitelist'] = set(whitelist_chat_ids)
        if 'whitelist' not in db:
            db['whitelist'] = {ADMIN_ID}
        bot_id = int(TELEGRAM_BOT_TOKEN.split(':')[0])
        pending_reply_manager = PendingReplyManager()
        async with await TelegramClient('bot', TELEGRAM_API_ID, TELEGRAM_API_HASH).start(bot_token=TELEGRAM_BOT_TOKEN) as bot:
            bot.parse_mode = None
            me = await bot.get_me()
            @bot.on(events.NewMessage)
            async def process(event):
                if event.message.chat_id is None:
                    return
                if event.message.sender_id is None:
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
