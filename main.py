import asyncio
import os
import logging
import shelve
import datetime
import time
import json
import traceback
import openai
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import RetryAfter, NetworkError, BadRequest

ADMIN_ID = 71863318
DEFAULT_MODEL = "gpt-4o"

openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

TELEGRAM_LENGTH_LIMIT = 4096
TELEGRAM_MIN_INTERVAL = 3
OPENAI_MAX_RETRY = 3
OPENAI_RETRY_INTERVAL = 10
FIRST_BATCH_DELAY = 1

telegram_last_timestamp = None
telegram_rate_limit_lock = asyncio.Lock()

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

def within_interval():
    global telegram_last_timestamp
    if telegram_last_timestamp is None:
        return False
    remaining_time = telegram_last_timestamp + TELEGRAM_MIN_INTERVAL - time.time()
    return remaining_time > 0

def ensure_interval(interval=TELEGRAM_MIN_INTERVAL):
    def decorator(func):
        async def new_func(*args, **kwargs):
            async with telegram_rate_limit_lock:
                global telegram_last_timestamp
                if telegram_last_timestamp is not None:
                    remaining_time = telegram_last_timestamp + interval - time.time()
                    if remaining_time > 0:
                        await asyncio.sleep(remaining_time)
                result = await func(*args, **kwargs)
                telegram_last_timestamp = time.time()
                return result
        return new_func
    return decorator

def retry(max_retry=30, interval=10):
    def decorator(func):
        async def new_func(*args, **kwargs):
            for _ in range(max_retry - 1):
                try:
                    return await func(*args, **kwargs)
                except (RetryAfter, NetworkError) as e:
                    if isinstance(e, BadRequest):
                        raise
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
    async def new_func(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message is None:
            return
        if update.message.from_user.id != ADMIN_ID:
            await send_message(update.effective_chat.id, 'Only admin can use this command', update.message.message_id)
            return
        await func(update, context)
    return new_func

def only_private(func):
    async def new_func(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message is None:
            return
        if update.effective_chat.id != update.message.from_user.id:
            await send_message(update.effective_chat.id, 'This command only works in private chat', update.message.message_id)
            return
        await func(update, context)
    return new_func

def only_whitelist(func):
    async def new_func(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.message is None:
            return
        if not is_whitelist(update.effective_chat.id):
            if update.effective_chat.id == update.message.from_user.id:
                await send_message(update.effective_chat.id, 'This chat is not in whitelist', update.message.message_id)
            return
        await func(update, context)
    return new_func

async def completion(messages, model, chat_id, msg_id):
    logging.info('Request (chat_id=%r, msg_id=%r, model=%r): %s', chat_id, msg_id, model, messages)
    stream = await openai.ChatCompletion.acreate(model=model, messages=messages, stream=True, request_timeout=15)
    async for response in stream:
        logging.info('Response (chat_id=%r, msg_id=%r): %s', chat_id, msg_id, json.dumps(response, ensure_ascii=False))
        obj = response['choices'][0]
        if obj['finish_reason'] is not None:
            assert not obj['delta']
            if obj['finish_reason'] == 'length':
                yield ' [!Output truncated due to limit]'
            return
        if 'role' in obj['delta']:
            if obj['delta']['role'] != 'assistant':
                raise ValueError("Role error")
        if 'content' in obj['delta']:
            yield obj['delta']['content']

def construct_chat_history(chat_id, msg_id):
    model = None
    messages = []
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
            else:
                new_messages.append(i)
        messages = new_messages + messages
        if reply_id is None:
            break
        msg_id = reply_id
    if model is None:
        return None, None
    return messages, model

@only_admin
async def add_whitelist_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_whitelist(update.effective_chat.id):
        await send_message(update.effective_chat.id, 'Already in whitelist', update.message.message_id)
        return
    add_whitelist(update.effective_chat.id)
    await send_message(update.effective_chat.id, 'Whitelist added', update.message.message_id)

@only_admin
async def del_whitelist_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_whitelist(update.effective_chat.id):
        await send_message(update.effective_chat.id, 'Not in whitelist', update.message.message_id)
        return
    del_whitelist(update.effective_chat.id)
    await send_message(update.effective_chat.id, 'Whitelist deleted', update.message.message_id)

@only_admin
@only_private
async def get_whitelist_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_message(update.effective_chat.id, str(get_whitelist()), update.message.message_id)

@retry()
@ensure_interval()
async def send_message(chat_id, text, reply_to_message_id):
    logging.info('Sending message: chat_id=%r, reply_to_message_id=%r, text=%r', chat_id, reply_to_message_id, text)
    msg = await application.bot.send_message(
        chat_id,
        text,
        reply_to_message_id=reply_to_message_id,
        disable_web_page_preview=True,
    )
    logging.info('Message sent: chat_id=%r, reply_to_message_id=%r, message_id=%r', chat_id, reply_to_message_id, msg.message_id)
    return msg.message_id

@retry()
@ensure_interval()
async def edit_message(chat_id, text, message_id):
    logging.info('Editing message: chat_id=%r, message_id=%r, text=%r', chat_id, message_id, text)
    try:
        await application.bot.edit_message_text(
            text,
            chat_id=chat_id,
            message_id=message_id,
            disable_web_page_preview=True,
        )
    except BadRequest as e:
        if e.message == 'Message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message':
            logging.info('Message not modified: chat_id=%r, message_id=%r', chat_id, message_id)
        else:
            raise
    else:
        logging.info('Message edited: chat_id=%r, message_id=%r', chat_id, message_id)

@retry()
@ensure_interval()
async def delete_message(chat_id, message_id):
    logging.info('Deleting message: chat_id=%r, message_id=%r', chat_id, message_id)
    try:
        await application.bot.delete_message(
            chat_id,
            message_id,
        )
    except BadRequest as e:
        if e.message == 'Message to delete not found':
            logging.info('Message to delete not found: chat_id=%r, message_id=%r', chat_id, message_id)
        else:
            raise
    else:
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
        if not within_interval():
            await self._force_update(self.text)

    async def finalize(self):
        await self._force_update(self.text)

@only_whitelist
async def reply_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    sender_id = update.message.from_user.id
    msg_id = update.message.message_id
    text = update.message.text
    logging.info('New message: chat_id=%r, sender_id=%r, msg_id=%r, text=%r', chat_id, sender_id, msg_id, text)
    reply_to_message = update.message.reply_to_message
    if reply_to_message is not None and update.message.reply_to_message.from_user.id == bot_id: # user reply to bot message
        reply_to_id = reply_to_message.message_id
        await pending_reply_manager.wait_for((chat_id, reply_to_id))
        msgs = []
        splits = text.split('$$$')
        if len(splits) % 2 == 0:
            await send_message(update.effective_chat.id, 'Usage: user$$$assistant$$$...$$$user', update.message.message_id)
            return
        for i, text in enumerate(splits):
            if i % 2 == 0:
                role = 'user'
            else:
                role = 'assistant'
            msgs.append({"role": role, "content": text})
        db[repr((chat_id, msg_id))] = msgs, reply_to_id
    elif text.startswith('s$'): # new message
        model = DEFAULT_MODEL
        if text.startswith('s$$'):
            text = text[3:]
            model = "gpt-3.5-turbo"
        else:
            text = text[2:]
        msgs = [{"model": model}]
        splits = text.split('$$$')
        if len(splits) % 2 or len(splits) < 2:
            await send_message(update.effective_chat.id, 'Usage: s$system$$$user$$$assistant$$$...$$$user', update.message.message_id)
            return
        for i, text in enumerate(splits):
            if i == 0:
                role = 'system'
            elif i % 2:
                role = 'user'
            else:
                role = 'assistant'
            msgs.append({"role": role, "content": text})
        db[repr((chat_id, msg_id))] = msgs, None
    else: # not reply or new message to bot
        if update.effective_chat.id == update.message.from_user.id: # if in private chat, send hint
            await send_message(update.effective_chat.id, 'Please start a new conversation with $ or reply to a bot message', update.message.message_id)
        return

    chat_history, model = construct_chat_history(chat_id, msg_id)
    if chat_history is None:
        await send_message(update.effective_chat.id, f"[!] Error: Unable to proceed with this conversation. Potential causes: the message replied to may be incomplete, contain an error, be a system message, or not exist in the database.", update.message.message_id)
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
                    db[repr((chat_id, message_id))] = [{"role": "assistant", "content": reply}], msg_id
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

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_message(update.effective_chat.id, f'chat_id={update.effective_chat.id} user_id={update.message.from_user.id} is_whitelisted={is_whitelist(update.effective_chat.id)}', update.message.message_id)

if __name__ == '__main__':
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
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), reply_handler))
        application.add_handler(CommandHandler('ping', ping))
        application.add_handler(CommandHandler('add_whitelist', add_whitelist_handler))
        application.add_handler(CommandHandler('del_whitelist', del_whitelist_handler))
        application.add_handler(CommandHandler('get_whitelist', get_whitelist_handler))
        application.run_polling()
