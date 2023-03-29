import os
import logging
import shelve
import datetime
import time
import openai
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
from telegram.error import RetryAfter, NetworkError, BadRequest
from requests.exceptions import RequestException

ADMIN_ID = 71863318
DEFAULT_MODEL = "gpt-4"
def PROMPT():
    s = "You are ChatGPT Telegram bot. ChatGPT is a large language model trained by OpenAI. This Telegram bot is developed by zzh whose username is zzh1996. Answer as concisely as possible. Knowledge cutoff: Sep 2021. Current Beijing Time: {current_time}"
    return s.replace('{current_time}', (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M'))

openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

TELEGRAM_LENGTH_LIMIT = 4096
TELEGRAM_TIMEOUT = 20
TELEGRAM_MIN_INTERVAL = 3
TELEGRAM_LAST_TIMESTAMP = None

def within_interval():
    global TELEGRAM_LAST_TIMESTAMP
    if TELEGRAM_LAST_TIMESTAMP is None:
        return False
    remaining_time = TELEGRAM_LAST_TIMESTAMP + TELEGRAM_MIN_INTERVAL - time.time()
    return remaining_time > 0

def ensure_interval(interval=TELEGRAM_MIN_INTERVAL):
    def decorator(func):
        def new_func(*args, **kwargs):
            global TELEGRAM_LAST_TIMESTAMP
            if TELEGRAM_LAST_TIMESTAMP is not None:
                remaining_time = TELEGRAM_LAST_TIMESTAMP + interval - time.time()
                if remaining_time > 0:
                    time.sleep(remaining_time)
            result = func(*args, **kwargs)
            TELEGRAM_LAST_TIMESTAMP = time.time()
            return result
        return new_func
    return decorator

def retry(max_retry=30, interval=10):
    def decorator(func):
        def new_func(*args, **kwargs):
            for _ in range(max_retry - 1):
                try:
                    return func(*args, **kwargs)
                except (RetryAfter, NetworkError) as e:
                    if isinstance(e, BadRequest):
                        raise
                    logging.exception(e)
                    time.sleep(interval)
            return func(*args, **kwargs)
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
    def new_func(update, context):
        if update.message is None:
            return
        if update.message.from_user.id != ADMIN_ID:
            update.message.reply_text('Only admin can use this command')
            return
        func(update, context)
    return new_func

def only_private(func):
    def new_func(update, context):
        if update.message is None:
            return
        if update.effective_chat.id != update.message.from_user.id:
            update.message.reply_text('This command only works in private chat')
            return
        func(update, context)
    return new_func

def only_whitelist(func):
    def new_func(update, context):
        if not is_whitelist(update.effective_chat.id):
            if update.message is None:
                return
            if update.effective_chat.id == update.message.from_user.id:
                update.message.reply_text('This chat is not in whitelist')
            return
        func(update, context)
    return new_func

def completion(chat_history, model): # chat_history = [user, ai, user, ai, ..., user]
    assert len(chat_history) % 2 == 1
    messages=[{"role": "system", "content": PROMPT()}]
    roles = ["user", "assistant"]
    role_id = 0
    for msg in chat_history:
        messages.append({"role": roles[role_id], "content": msg})
        role_id = 1 - role_id
    logging.info('Request: %s', messages)
    stream = openai.ChatCompletion.create(model=model, messages=messages, stream=True, request_timeout=(30, 30))
    for response in stream:
        logging.info('Response: %s', response)
        obj = response['choices'][0]
        if obj['finish_reason'] is not None:
            assert not obj['delta']
            if obj['finish_reason'] == 'length':
                yield ' [!长度超限]'
            return
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
            logging.error('History message not found')
            return None, None
        is_bot, text, reply_id, *params = db[key]
        if params:
            model = params[0]
        if is_bot != should_be_bot:
            logging.error('Role does not match')
            return None, None
        messages.append(text)
        should_be_bot = not should_be_bot
        if reply_id is None:
            break
        msg_id = reply_id
    if len(messages) % 2 != 1:
        logging.error('First message not from user')
        return None, None
    return messages[::-1], model

@only_admin
def add_whitelist_handler(update, context):
    if is_whitelist(update.effective_chat.id):
        update.message.reply_text('Already in whitelist')
        return
    add_whitelist(update.effective_chat.id)
    update.message.reply_text('Whitelist added')

@only_admin
def del_whitelist_handler(update, context):
    if not is_whitelist(update.effective_chat.id):
        update.message.reply_text('Not in whitelist')
        return
    del_whitelist(update.effective_chat.id)
    update.message.reply_text('Whitelist deleted')

@only_admin
@only_private
def get_whitelist_handler(update, context):
    update.message.reply_text(str(get_whitelist()))

@retry()
@ensure_interval()
def send_message(chat_id, text, reply_to_message_id):
    msg = updater.bot.send_message(
        chat_id,
        text,
        reply_to_message_id=reply_to_message_id,
        disable_web_page_preview=True,
        timeout=TELEGRAM_TIMEOUT,
    )
    logging.info('Send message: chat_id=%r, reply_to_message_id=%r, text=%r', chat_id, reply_to_message_id, text)
    return msg.message_id

@retry()
@ensure_interval()
def edit_message(chat_id, text, message_id):
    updater.bot.edit_message_text(
        text,
        chat_id=chat_id,
        message_id=message_id,
        disable_web_page_preview=True,
        timeout=TELEGRAM_TIMEOUT,
    )
    logging.info('Edit message: chat_id=%r, message_id=%r, text=%r', chat_id, message_id, text)

@retry()
@ensure_interval()
def delete_message(chat_id, message_id):
    updater.bot.delete_message(
        chat_id,
        message_id,
        timeout=TELEGRAM_TIMEOUT,
    )
    logging.info('Delete message: chat_id=%r, message_id=%r', chat_id, message_id)

class BotReplyMessages:
    def __init__(self, chat_id, orig_msg_id, prefix):
        self.prefix = prefix
        self.msg_len = TELEGRAM_LENGTH_LIMIT - len(prefix)
        assert self.msg_len > 0
        self.chat_id = chat_id
        self.orig_msg_id = orig_msg_id
        self.replied_msgs = []
        self.text = ''

    def _force_update(self, text):
        slices = []
        while len(text) > self.msg_len:
            slices.append(text[:self.msg_len])
            text = text[self.msg_len:]
        if text:
            slices.append(text)

        for i in range(min(len(slices), len(self.replied_msgs))):
            msg_id, msg_text = self.replied_msgs[i]
            if slices[i] != msg_text:
                edit_message(self.chat_id, self.prefix + slices[i], msg_id)
                self.replied_msgs[i] = (msg_id, slices[i])
        if len(slices) > len(self.replied_msgs):
            for i in range(len(self.replied_msgs), len(slices)):
                if i == 0:
                    reply_to = self.orig_msg_id
                else:
                    reply_to, _ = self.replied_msgs[i - 1]
                msg_id = send_message(self.chat_id, self.prefix + slices[i], reply_to)
                self.replied_msgs.append((msg_id, slices[i]))
        if len(self.replied_msgs) > len(slices):
            for i in range(len(slices), len(self.replied_msgs)):
                msg_id, _ = self.replied_msgs[i]
                delete_message(self.chat_id, msg_id)
            self.replied_msgs = self.replied_msgs[:len(slices)]

    def update(self, text):
        self.text = text
        if not within_interval():
            self._force_update(self.text)

    def finalize(self):
        self._force_update(self.text)

@only_whitelist
def reply_handler(update, context):
    chat_id = update.effective_chat.id
    sender_id = update.message.from_user.id
    msg_id = update.message.message_id
    text = update.message.text
    logging.info('New message: chat=%r, sender=%r, id=%r, msg=%r', chat_id, sender_id, msg_id, text)
    reply_to_message = update.message.reply_to_message
    reply_to_id = None
    model = DEFAULT_MODEL
    if reply_to_message is not None and update.message.reply_to_message.from_user.id == bot_id: # user reply to bot message
        reply_to_id = reply_to_message.message_id
    elif text.startswith('$'): # new message
        if text.startswith('$'):
            if text.startswith('$$'):
                text = text[2:]
                model = "gpt-3.5-turbo"
            else:
                text = text[1:]
    else: # not reply or new message to bot
        if update.effective_chat.id == update.message.from_user.id: # if in private chat, send hint
            update.message.reply_text('Please start a new conversation with $ or reply to a bot message')
        return
    db[repr((chat_id, msg_id))] = (False, text, reply_to_id, model)

    chat_history, model = construct_chat_history(chat_id, msg_id)
    if chat_history is None:
        update.message.reply_text(f"[!] Error: Can't fetch this conversation, please start a new one.")
        return

    reply = ''
    replymsgs = BotReplyMessages(chat_id, msg_id, f'[{model}] ')
    try:
        cnt = 0
        while True:
            try:
                stream = completion(chat_history, model)
                break
            except openai.OpenAIError as e:
                if e.http_status != 500:
                    raise
                cnt += 1
                if cnt == 5:
                    raise
                time.sleep(5)
        for delta in stream:
            reply += delta
            replymsgs.update(reply + ' [!正在生成]')
        replymsgs.update(reply)
        replymsgs.finalize()
    except (openai.OpenAIError, RequestException) as e:
        logging.exception('OpenAI Error: %s', e)
        error_msg = f'[!] OpenAI Error: {e}'
        if reply:
            error_msg = reply + '\n\n' + error_msg
        replymsgs.update(error_msg)
        replymsgs.finalize()
        return

    for message_id, _ in replymsgs.replied_msgs:
        db[repr((chat_id, message_id))] = (True, reply, msg_id, model)

def ping(update, context):
    update.message.reply_text(f'chat_id={update.effective_chat.id} user_id={update.message.from_user.id} is_whitelisted={is_whitelist(update.effective_chat.id)}')

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
        # db[(chat_id, msg_id)] = (is_bot, text, reply_id)
        # db['whitelist'] = set(whitelist_chat_ids)
        if 'whitelist' not in db:
            db['whitelist'] = {ADMIN_ID}
        updater = Updater(token=TELEGRAM_BOT_TOKEN, use_context=True)
        bot_id = updater.bot.get_me().id
        logging.info("Bot ID: %s", bot_id)
        dispatcher = updater.dispatcher
        dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), reply_handler))
        dispatcher.add_handler(CommandHandler('ping', ping))
        dispatcher.add_handler(CommandHandler('add_whitelist', add_whitelist_handler))
        dispatcher.add_handler(CommandHandler('del_whitelist', del_whitelist_handler))
        dispatcher.add_handler(CommandHandler('get_whitelist', get_whitelist_handler))
        updater.start_polling()
        updater.idle()
