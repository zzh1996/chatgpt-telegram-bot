import asyncio
import os
import logging
import shelve
import datetime
import time
import json
import traceback
from collections import defaultdict
import openai
from telethon import TelegramClient, events, errors, functions, types

ADMIN_ID = 71863318

MODELS = [
    {'prefix': '$$', 'model': 'gpt-3.5-turbo-1106', 'prompt_template': 'You are ChatGPT Telegram bot. ChatGPT is a large language model trained by OpenAI. This Telegram bot is developed by zzh whose username is zzh1996. Answer as concisely as possible. Knowledge cutoff: Sep 2021. Current Beijing Time: {current_time}"'},
    {'prefix': '3$', 'model': 'gpt-3.5-turbo', 'prompt_template': 'You are ChatGPT Telegram bot. ChatGPT is a large language model trained by OpenAI. This Telegram bot is developed by zzh whose username is zzh1996. Answer as concisely as possible. Knowledge cutoff: Sep 2021. Current Beijing Time: {current_time}"'},
    {'prefix': '$', 'model': 'gpt-4-1106-preview', 'prompt_template': 'You are ChatGPT Telegram bot. ChatGPT is a large language model trained by OpenAI, based on the GPT-4 architecture. This Telegram bot is developed by zzh whose username is zzh1996. Answer as concisely as possible. Knowledge cutoff: Apr 2023. Current Beijing Time: {current_time}"'},
    {'prefix': '4$', 'model': 'gpt-4', 'prompt_template': 'You are ChatGPT Telegram bot. ChatGPT is a large language model trained by OpenAI, based on the GPT-4 architecture. This Telegram bot is developed by zzh whose username is zzh1996. Answer as concisely as possible. Knowledge cutoff: Sep 2021. Current Beijing Time: {current_time}"'},
]
DEFAULT_MODEL = 'gpt-4' # For compatibility with the old database format

def get_prompt(model):
    for m in MODELS:
        if m['model'] == model:
            return m['prompt_template'].replace('{current_time}', (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'))
    raise ValueError('Model not found')

openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID"))
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")

TELEGRAM_LENGTH_LIMIT = 4096
TELEGRAM_MIN_INTERVAL = 3
OPENAI_MAX_RETRY = 3
OPENAI_RETRY_INTERVAL = 10
FIRST_BATCH_DELAY = 1

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

async def completion(chat_history, model, chat_id, msg_id): # chat_history = [user, ai, user, ai, ..., user]
    assert len(chat_history) % 2 == 1
    messages=[{"role": "system", "content": get_prompt(model)}]
    roles = ["user", "assistant"]
    role_id = 0
    for msg in chat_history:
        messages.append({"role": roles[role_id], "content": msg})
        role_id = 1 - role_id
    logging.info('Request (chat_id=%r, msg_id=%r): %s', chat_id, msg_id, messages)
    stream = await openai.ChatCompletion.acreate(model=model, messages=messages, stream=True, request_timeout=15)
    finished = False
    async for response in stream:
        logging.info('Response (chat_id=%r, msg_id=%r): %s', chat_id, msg_id, json.dumps(response, ensure_ascii=False))
        assert not finished
        obj = response['choices'][0]
        if obj['finish_reason'] is not None:
            assert not obj['delta']
            if obj['finish_reason'] == 'length':
                yield ' [!Output truncated due to limit]'
            elif obj['finish_reason'] == 'stop':
                pass
            else:
                yield f'\n\n[!] Error: finish_reason="{obj["finish_reason"]}"'
            finished = True
        if 'role' in obj['delta']:
            if obj['delta']['role'] != 'assistant':
                raise ValueError("Role error")
        if 'content' in obj['delta']:
            yield obj['delta']['content']

def construct_chat_history(chat_id, msg_id):
    messages = []
    should_be_bot = False
    model = DEFAULT_MODEL
    while True:
        key = repr((chat_id, msg_id))
        if key not in db:
            logging.error('History message not found (chat_id=%r, msg_id=%r)', chat_id, msg_id)
            return None, None
        is_bot, text, reply_id, *params = db[key]
        if params:
            model = params[0]
        if is_bot != should_be_bot:
            logging.error('Role does not match (chat_id=%r, msg_id=%r)', chat_id, msg_id)
            return None, None
        messages.append(text)
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
        text += f'Prefix: "{m["prefix"]}", model: {m["model"]}\n'
    await send_message(message.chat_id, text, message.id)

@retry()
@ensure_interval()
async def send_message(chat_id, text, reply_to_message_id):
    logging.info('Sending message: chat_id=%r, reply_to_message_id=%r, text=%r', chat_id, reply_to_message_id, text)
    msg = await bot.send_message(
        chat_id,
        text,
        reply_to=reply_to_message_id,
        link_preview=False,
    )
    logging.info('Message sent: chat_id=%r, reply_to_message_id=%r, message_id=%r', chat_id, reply_to_message_id, msg.id)
    return msg.id

@retry()
@ensure_interval()
async def edit_message(chat_id, text, message_id):
    logging.info('Editing message: chat_id=%r, message_id=%r, text=%r', chat_id, message_id, text)
    try:
        await bot.edit_message(
            chat_id,
            message_id,
            text,
            link_preview=False,
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
    logging.info('New message: chat_id=%r, sender_id=%r, msg_id=%r, text=%r', chat_id, sender_id, msg_id, text)
    reply_to_id = None
    model = DEFAULT_MODEL
    if message.is_reply:
        reply_to_message = await message.get_reply_message()
        if reply_to_message.sender_id == bot_id: # user reply to bot message
            reply_to_id = message.reply_to.reply_to_msg_id
            await pending_reply_manager.wait_for((chat_id, reply_to_id))
        else:
            return
    else: # new message
        for m in MODELS:
            if text.startswith(m['prefix']):
                text = text[len(m['prefix']):]
                model = m['model']
                break
        else: # not reply or new message to bot
            if chat_id == sender_id: # if in private chat, send hint
                await send_message(chat_id, 'Please start a new conversation with $ or reply to a bot message', msg_id)
            return
    db[repr((chat_id, msg_id))] = (False, text, reply_to_id, model)

    chat_history, model = construct_chat_history(chat_id, msg_id)
    if chat_history is None:
        await send_message(chat_id, f"[!] Error: Unable to proceed with this conversation. Potential causes: the message replied to may be incomplete, contain an error, be a system message, or not exist in the database.", msg_id)
        return

    error_cnt = 0
    while True:
        reply = ''
        async with BotReplyMessages(chat_id, msg_id, f'[{model}] ') as replymsgs:
            try:
                stream = completion(chat_history, model, chat_id, msg_id)
                first_update_timestamp = None
                async for delta in stream:
                    reply += delta
                    if first_update_timestamp is None:
                        first_update_timestamp = time.time()
                    if time.time() >= first_update_timestamp + FIRST_BATCH_DELAY:
                        await replymsgs.update(reply + ' [!Generating...]')
                await replymsgs.update(reply)
                await replymsgs.finalize()
                for message_id, _ in replymsgs.replied_msgs:
                    db[repr((chat_id, message_id))] = (True, reply, msg_id, model)
                return
            except Exception as e:
                error_cnt += 1
                logging.exception('Error (chat_id=%r, msg_id=%r, cnt=%r): %s', chat_id, msg_id, error_cnt, e)
                will_retry = not isinstance (e, openai.InvalidRequestError) and error_cnt <= OPENAI_MAX_RETRY
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
        # db[(chat_id, msg_id)] = (is_bot, text, reply_id, model)
        # compatible old db format: db[(chat_id, msg_id)] = (is_bot, text, reply_id)
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
