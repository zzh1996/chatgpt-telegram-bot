import asyncio
import os
import logging
import shelve
import datetime
import time
import json
import traceback
import uuid
import openai
import tiktoken
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import RetryAfter, NetworkError, BadRequest

from plugins.calculator import Calculator
from plugins.browsing import Browsing
from plugins.search import Search
from plugins.youtube import Youtube

ADMIN_ID = 71863318
DEFAULT_MODEL = "gpt-4-1106-preview"
TRIGGER = 'p$'
PLUGINS = [Search, Browsing, Youtube, Calculator]

def PROMPT(model):
    s = "You are ChatGPT Telegram bot with the abilities to search on Google, read web pages, get YouTube transcripts and use a calculator. You must use these abilities to answer user's questions. You shouldn't answer or calculate by yourself without these tools. Always search by keywords and don't repeat user's question in search query. Search for \"Bill Gates\" instead of \"Who is Bill Gates\" when user asks \"Who is Bill Gates?\". You should read web pages after searching if they contain information you need. Summarize results instead of verbatim repetition. Always respond in the same language as the user's questions, even though the search and web pages may be in other languages. If the user asks in Chinese, please answer in Chinese, and if the user asks in English, please answer in English. Answer as concisely as possible. Knowledge cutoff: Apr 2023. Current Beijing Time: {current_time}"
    return s.replace('{current_time}', (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'))

openai.api_key = os.getenv("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

TELEGRAM_LENGTH_LIMIT = 4096
TELEGRAM_MIN_INTERVAL = 3
OPENAI_MAX_RETRY = 3
OPENAI_RETRY_INTERVAL = 10
FIRST_BATCH_DELAY = 1
FUNCTION_CALLS_LIMIT = 5
FUNCTION_CALLS_MAX_TOKENS = 32768

telegram_last_timestamp = None
telegram_rate_limit_lock = asyncio.Lock()

def dump_and_show(chat_logs, chat_uuid):
    dir = f'chatlogs/{chat_uuid[:2]}/{chat_uuid[2:4]}'
    path = f'{dir}/{chat_uuid[4:]}.json'
    os.makedirs(dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(chat_logs, f, ensure_ascii=False, indent=2)
    logging.info('Chat logs saved to %s', path)
    return f'https://ai.sqrt-1.me/#{chat_uuid}'

class PluginManager:
    def __init__(self, plugin_classes):
        self.plugins = [c() for c in plugin_classes]
        self.function_registry = {}
        self.function_prompt = []
        for plugin in self.plugins:
            for func in plugin.functions:
                self.function_prompt.append(func)
                self.function_registry[func['name']] = getattr(plugin, func['name'])

    async def call(self, func_name, params):
        if func_name not in self.function_registry:
            return json.dumps({'error': f'Function {func_name} not found'})
        func = self.function_registry[func_name]
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**params)
            else:
                result = func(**params)
            return {'result': result}
        except Exception as e:
            logging.exception('Error during function call')
            return {'error': traceback.format_exception_only(e)[-1].strip()}

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

async def completion(messages, model, chat_id, msg_id, functions, function_call=None):
    logging.info('Request (chat_id=%r, msg_id=%r, model=%r, functions=%r): %s', chat_id, msg_id, model, functions, messages)
    params = {
        'model': model,
        'messages': messages,
        'stream': True,
        'request_timeout': 15,
        'functions': functions,
    }
    if function_call is not None:
        params['function_call'] = function_call
    stream = await openai.ChatCompletion.acreate(**params)
    function_call = {'name': None, 'arguments': ''}
    async for response in stream:
        logging.info('Response (chat_id=%r, msg_id=%r): %s', chat_id, msg_id, json.dumps(response, ensure_ascii=False))
        obj = response['choices'][0]
        if obj['finish_reason'] is not None:
            assert not obj['delta']
            if obj['finish_reason'] == 'length':
                yield ' [!Output truncated due to limit]'
            if obj['finish_reason'] == 'function_call':
                yield function_call
            return
        if 'role' in obj['delta']:
            if obj['delta']['role'] != 'assistant':
                raise ValueError("Role error")
        if 'content' in obj['delta'] and obj['delta']['content'] is not None:
            yield obj['delta']['content']
        if 'function_call' in obj['delta']:
            if 'name' in obj['delta']['function_call']:
                assert function_call['name'] is None
                assert obj['delta']['function_call']['arguments'] == ''
                function_call['name'] = obj['delta']['function_call']['name']
            if 'arguments' in obj['delta']['function_call']:
                function_call['arguments'] += obj['delta']['function_call']['arguments']

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
        msgs = [{"role": "user", "content": text}]
        db[repr((chat_id, msg_id))] = msgs, reply_to_id
    elif text.startswith(TRIGGER): # new message
        model = DEFAULT_MODEL
        text = text[len(TRIGGER):]
        msgs = [{"model": model}]
        msgs.append({"role": "system", "content": PROMPT(model)})
        msgs.append({"role": "user", "content": text})
        db[repr((chat_id, msg_id))] = msgs, None
    else: # not reply or new message to bot
        if update.effective_chat.id == update.message.from_user.id: # if in private chat, send hint
            await send_message(update.effective_chat.id, f'Please start a new conversation with {TRIGGER} or reply to a bot message', update.message.message_id)
        return

    chat_history, model = construct_chat_history(chat_id, msg_id)
    if chat_history is None:
        await send_message(update.effective_chat.id, f"[!] Error: Unable to proceed with this conversation. Potential causes: the message replied to may be incomplete, contain an error, be a system message, or not exist in the database.", update.message.message_id)
        return

    new_messages = []
    function_calls_counter = 0
    continue_round = True
    last_msg_id = msg_id
    chat_uuid = str(uuid.uuid4())
    while continue_round:
        continue_round = False
        error_cnt = 0
        while True:
            reply = ''
            async with BotReplyMessages(chat_id, last_msg_id, f'[{model}] ') as replymsgs:
                try:
                    function_call = None
                    if function_calls_counter >= FUNCTION_CALLS_LIMIT:
                        function_call = "none" # tell model not to use function calls anymore
                    stream = completion(chat_history + new_messages, model, chat_id, last_msg_id, plugin_manager.function_prompt, function_call)
                    first_update_timestamp = None
                    function_call = None
                    async for delta in stream:
                        if isinstance(delta, str): # text
                            reply += delta
                            if first_update_timestamp is None:
                                first_update_timestamp = time.time()
                            if time.time() >= first_update_timestamp + FIRST_BATCH_DELAY:
                                await replymsgs.update(reply + ' [!Generating...]')
                        else: # function call
                            function_call = delta
                    msg = {"role": "assistant", "content": reply}
                    if function_call is not None:
                        msg['function_call'] = function_call
                    enc = tiktoken.encoding_for_model(model)
                    estimated_input_tokens = len(enc.encode(json.dumps(chat_history + new_messages, ensure_ascii=False)))
                    estimated_output_tokens = len(enc.encode(json.dumps(msg, ensure_ascii=False)))
                    estimated_dollars = estimated_input_tokens * 1e-5 + estimated_output_tokens * 3e-5
                    new_messages.append(msg)
                    if function_call is not None:
                        try:
                            arguments = json.loads(function_call["arguments"])
                        except Exception as e:
                            logging.exception('Error decoding json, json: %s', function_call["arguments"])
                            reply += f'\n\n[!] Function call arguments not valid JSON\nFunction name: {function_call["name"]}\nFunction arguments: {function_call["arguments"]}'
                        else:
                            reply += f'\n\n[+] Function call: {function_call["name"]}({json.dumps(arguments, ensure_ascii=False)})'
                            if function_calls_counter >= FUNCTION_CALLS_LIMIT:
                                reply += f'\n\n[!] Error: Function calls exceeded limit {FUNCTION_CALLS_LIMIT}, won\'t continue.'
                            else:
                                reply += f'\nExecuting function call...'
                                await replymsgs.update(reply)
                                call_response = await plugin_manager.call(function_call['name'], arguments)
                                # call_response can be appended to new_messages safely when there's an error
                                # because continue_round will be False so new_messages won't be saved into db
                                call_response_json = json.dumps(call_response, ensure_ascii=False)
                                new_messages.append({"role": "function", "name": function_call['name'], "content": call_response_json})
                                if 'error' in call_response:
                                    reply += f'\n\n[!] Error: Function call error: {call_response["error"]}'
                                else:
                                    enc = tiktoken.encoding_for_model(model)
                                    response_tokens = len(enc.encode(call_response_json))
                                    reply += f' Done! (Response {response_tokens} tokens)'
                                    if response_tokens <= 100:
                                        reply += '\nResponse: '+ call_response_json
                                    if response_tokens > FUNCTION_CALLS_MAX_TOKENS:
                                        reply += f'\n\n[!] Error: Function call result exceeded token limit {FUNCTION_CALLS_MAX_TOKENS}, won\'t continue.'
                                    else:
                                        function_calls_counter += 1
                                        continue_round = True
                    if estimated_dollars >= 0.1:
                        reply += f'\n\n[$] Estimated tokens: {estimated_input_tokens / 1000:.1f}K input + {estimated_output_tokens/1000:.1f}K output\n[$] Estimated cost（本次请求成本）: ${estimated_dollars:.2f} / ¥{estimated_dollars * 7.28:.2f}\n[$] Please note the costs 请注意费用消耗'
                    link = dump_and_show(chat_history + new_messages, chat_uuid)
                    reply += f'\n\n[+] Chat logs: {link}'
                    await replymsgs.update(reply)
                    await replymsgs.finalize()
                    last_msg_id = replymsgs.replied_msgs[-1][0]
                    if function_call is None:
                        # update db inside "async with" to ensure db is ready when the pending reply lock is released
                        db[repr((chat_id, last_msg_id))] = new_messages, msg_id
                    break
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
                        return

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_message(update.effective_chat.id, f'chat_id={update.effective_chat.id} user_id={update.message.from_user.id} is_whitelisted={is_whitelist(update.effective_chat.id)} trigger={TRIGGER}', update.message.message_id)

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
        plugin_manager = PluginManager(PLUGINS)
        application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).concurrent_updates(True).build()
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), reply_handler))
        application.add_handler(CommandHandler('ping', ping))
        application.add_handler(CommandHandler('add_whitelist', add_whitelist_handler))
        application.add_handler(CommandHandler('del_whitelist', del_whitelist_handler))
        application.add_handler(CommandHandler('get_whitelist', get_whitelist_handler))
        application.run_polling()
