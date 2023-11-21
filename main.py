import asyncio
import os
import logging
import shelve
import time
import json
import traceback
import hashlib
import string
from collections import defaultdict
import openai
from telethon import TelegramClient, events, errors, functions, types

ADMIN_ID = 71863318

MODEL = 'gpt-4-1106-preview'

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
    async def new_func(message, *args):
        if message.sender_id != ADMIN_ID:
            await send_message(message.chat_id, 'Only admin can use this command', message.id)
            return
        await func(message, *args)
    return new_func

def only_private(func):
    async def new_func(message, *args):
        if message.chat_id != message.sender_id:
            await send_message(message.chat_id, 'This command only works in private chat', message.id)
            return
        await func(message, *args)
    return new_func

def only_whitelist(func):
    async def new_func(message, *args):
        if not is_whitelist(message.chat_id):
            if message.chat_id == message.sender_id:
                await send_message(message.chat_id, 'This chat is not in whitelist', message.id)
            return
        await func(message, *args)
    return new_func

async def completion(messages, chat_id, msg_id):
    logging.info('Request (chat_id=%r, msg_id=%r): %s', chat_id, msg_id, messages)
    stream = await openai.ChatCompletion.acreate(model=MODEL, messages=messages, stream=True, request_timeout=15)
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
    trigger, gpt_id = None, None
    messages = []
    while True:
        key = repr((chat_id, msg_id))
        if key not in db:
            logging.error('History message not found (chat_id=%r, msg_id=%r)', chat_id, msg_id)
            return None, None
        msgs, reply_id = db[key]
        new_messages = []
        for i in msgs:
            if 'gpt' in i:
                trigger, gpt_id = i['gpt']
            else:
                new_messages.append(i)
        messages = new_messages + messages
        if reply_id is None:
            break
        msg_id = reply_id
    if trigger is None or gpt_id is None:
        return None, None
    return messages, (trigger, gpt_id)

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

create_gpt_help_text = '创建新的 GPT\n使用方法：/create [system prompt]\n例如：\n/create 你是一个中英翻译机器人，请把用户输入的内容从中文翻译成英文，或者从英文翻译成中文。请不要输出任何额外信息。'

async def create_gpt(message, text):
    text = text.strip()
    if not text:
        await send_message(message.chat_id, create_gpt_help_text, message.id)
        return
    gpt_id = hashlib.sha256(text.encode()).hexdigest()[:32]
    db[repr(('gpts', gpt_id))] = text
    await send_message(message.chat_id, f'创建成功！\n你可以使用 /set 命令来把此 GPT 绑定为本群的 Trigger\n你的 GPT ID: {gpt_id}\n例如：\n/set {gpt_id} translator\n之后可以用 translator$ 这个前缀来使用\n你也可以用 {gpt_id}$ 前缀来直接使用\n请注意：此 GPT ID 在不同群中都可以使用，但每个群要单独绑定 Trigger\n\n你的 system prompt：\n{text}', message.id)

set_trigger_help_text = '把 GPT 绑定为本群的 Trigger\n使用方法：/set [GPT ID] [Trigger] [描述]\n其中描述会被 /list 功能展示，可以为空\n例如：\n/set 976648131458b1780b2adfb5c48eede6 translator 这是一个翻译机器人\n然后可以用 translator$ 这个前缀来使用对应的 GPT\n注意：Trigger 仅限大小写字母和数字，长度为 2 到 20'

@only_whitelist
async def set_trigger(message, text):
    splits = text.strip().split()
    if len(splits) in [2, 3]:
        if len(splits) == 2:
            splits.append('')
        gpt_id, trigger, description = splits
        if len(description) > 50 or '\n' in description:
            await send_message(message.chat_id, f'GPT 的描述不能超过 50 个字符，并且不能包含换行符', message.id)
            return
        if trigger.endswith('$'):
            trigger = trigger[:-1]
        if repr(('gpts', gpt_id)) not in db:
            await send_message(message.chat_id, f'GPT ID "{gpt_id}" 不存在', message.id)
            return
        charset = string.ascii_letters + string.digits
        if not all(c in charset for c in trigger) or len(trigger) not in range(2, 21):
            await send_message(message.chat_id, f'Trigger 仅限大小写字母和数字，长度为 2 到 20', message.id)
            return
        if repr(('gpts_triggers', message.chat_id)) in db:
            group_triggers = db[repr(('gpts_triggers', message.chat_id))]
        else:
            group_triggers = []
        for i, (author, _, trigger_, _) in enumerate(group_triggers):
            if trigger_ == trigger: # 已存在
                if author != message.sender_id and message.sender_id != ADMIN_ID:
                    await send_message(message.chat_id, f'这个 Trigger 被别人（uid={author}）占用了，你可以让 TA 删除或者找管理员删除', message.id)
                    return
                group_triggers[i] = (message.sender_id, gpt_id, trigger, description)
                break
        else:
            group_triggers.append((message.sender_id, gpt_id, trigger, description))
        db[repr(('gpts_triggers', message.chat_id))] = group_triggers
        await send_message(message.chat_id, f'Trigger 已设置！\n从现在起，在本群中你可以使用 {trigger}$ 开头的消息来使用这个 GPT\nTrigger: {trigger}\nGPT ID：{gpt_id}\n描述：{description}', message.id)
        return
    await send_message(message.chat_id, set_trigger_help_text, message.id)

del_trigger_help_text = '删除本群的 Trigger\n使用方法：/unset [Trigger]\n例如：\n/unset translator'

@only_whitelist
async def del_trigger(message, text):
    trigger = text.strip()
    if trigger.endswith('$'):
        trigger = trigger[:-1]
    if trigger:
        if repr(('gpts_triggers', message.chat_id)) in db:
            group_triggers = db[repr(('gpts_triggers', message.chat_id))]
        else:
            group_triggers = []
        for i, (author, _, trigger_, _) in enumerate(group_triggers):
            if trigger_ == trigger:
                if author != message.sender_id and message.sender_id != ADMIN_ID:
                    await send_message(message.chat_id, f'你无权删除这个 Trigger，你可以让拥有者（uid={author}）删除或者找管理员删除', message.id)
                    return
                del group_triggers[i]
                db[repr(('gpts_triggers', message.chat_id))] = group_triggers
                await send_message(message.chat_id, f'Trigger {trigger} 已删除！', message.id)
                return
        else:
            await send_message(message.chat_id, f'本群根本不存在 "{trigger}" 这个 Trigger！', message.id)
            return
    await send_message(message.chat_id, del_trigger_help_text, message.id)

list_trigger_help_text = '列出本群的所有 Trigger\n使用方法：/list'

@only_whitelist
async def list_trigger(message):
    if repr(('gpts_triggers', message.chat_id)) not in db:
        group_triggers = []
    else:
        group_triggers = db[repr(('gpts_triggers', message.chat_id))]
    if not group_triggers:
        await send_message(message.chat_id, f'本群还没有 Trigger，请使用 /create 命令创建 GPT 后使用 /set 命令绑定 Trigger', message.id)
        return
    text = '本群的 Trigger 列表：\n\n'
    for _, _, trigger, description in group_triggers:
        text += f'[{trigger}] {description}\n'
    async with BotReplyMessages(message.chat_id, message.id, '') as b:
        await b.update(text)
        await b.finalize()

list_trigger_full_help_text = '列出本群的所有 Trigger，包含 GPT ID 和添加人\n使用方法：/list_full'

@only_whitelist
async def list_trigger_full(message):
    if repr(('gpts_triggers', message.chat_id)) not in db:
        group_triggers = []
    else:
        group_triggers = db[repr(('gpts_triggers', message.chat_id))]
    if not group_triggers:
        await send_message(message.chat_id, f'本群还没有 Trigger，请使用 /create 命令创建 GPT 后使用 /set 命令绑定 Trigger', message.id)
        return
    text = '本群的 Trigger 列表：\n\n'
    for author, gpt_id, trigger, description in group_triggers:
        text += f'Trigger: {trigger}\nGPT ID: {gpt_id}\n添加人：uid={author}\n描述：{description}\n\n'
    async with BotReplyMessages(message.chat_id, message.id, '') as b:
        await b.update(text)
        await b.finalize()

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
    msgs = []
    if message.is_reply:
        reply_to_message = await message.get_reply_message()
        if reply_to_message.sender_id == bot_id: # user reply to bot message
            reply_to_id = message.reply_to.reply_to_msg_id
            await pending_reply_manager.wait_for((chat_id, reply_to_id))
            msgs.append({"role": "user", "content": text})
        else: # user reply to user message
            return
    else: # new message
        splits = text.split('$', 1)
        if len(splits) != 2:
            return
        trigger, text = splits
        gpt_id = None
        if repr(('gpts', trigger)) in db:
            gpt_id = trigger
        else:
            if repr(('gpts_triggers', chat_id)) in db:
                group_triggers = db[repr(('gpts_triggers', chat_id))]
            else:
                group_triggers = []
            for _, gpt_id_, trigger_, _ in group_triggers:
                if trigger_ == trigger:
                    gpt_id = gpt_id_
        if gpt_id is None:
            return
        system_prompt = db[repr(('gpts', gpt_id))]
        msgs.append({"gpt": (trigger, gpt_id)})
        msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": text})

    db[repr((chat_id, msg_id))] = (msgs, reply_to_id)

    chat_history, t = construct_chat_history(chat_id, msg_id)
    if chat_history is None:
        await send_message(chat_id, f"[!] Error: Unable to proceed with this conversation. Potential causes: the message replied to may be incomplete, contain an error, be a system message, or not exist in the database.", msg_id)
        return

    trigger, gpt_id = t
    if trigger == gpt_id:
        prefix = f'[{gpt_id}] '
    else:
        if repr(('gpts_triggers', chat_id)) in db:
            group_triggers = db[repr(('gpts_triggers', chat_id))]
        else:
            group_triggers = []
        for _, gpt_id_, trigger_, _ in group_triggers:
            if trigger_ == trigger and gpt_id_ == gpt_id: # not changed
                prefix = f'[{trigger}] '
                break
        else: # trigger no longer exists
            prefix = f'[{gpt_id}] (was {trigger})\n'

    error_cnt = 0
    while True:
        reply = ''
        async with BotReplyMessages(chat_id, msg_id, prefix) as replymsgs:
            try:
                stream = completion(chat_history, chat_id, msg_id)
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

async def ping(message):
    await send_message(message.chat_id, f'chat_id={message.chat_id} user_id={message.sender_id} is_whitelisted={is_whitelist(message.chat_id)}', message.id)

async def help(message):
    help_message = '\n\n'.join([create_gpt_help_text, set_trigger_help_text, del_trigger_help_text, list_trigger_help_text, list_trigger_full_help_text])
    await send_message(message.chat_id, help_message, message.id)

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
        # msgs = [{"role": "system/user/assistant", "content": "foo"}]
        # db['whitelist'] = set(whitelist_chat_ids)
        # db[('gpts', gpt_id)] = system_prompt
        # db[('gpts_triggers', chat_id)] = [(author, gpt_id, trigger, description)]
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
                elif text == '/add_whitelist' or text == f'/add_whitelist@{me.username}':
                    await add_whitelist_handler(event.message)
                elif text == '/del_whitelist' or text == f'/del_whitelist@{me.username}':
                    await del_whitelist_handler(event.message)
                elif text == '/get_whitelist' or text == f'/get_whitelist@{me.username}':
                    await get_whitelist_handler(event.message)
                elif text.startswith(f'/create@{me.username}'):
                    await create_gpt(event.message, text[len(f'/create@{me.username}'):])
                elif text.startswith('/create'):
                    await create_gpt(event.message, text[len('/create'):])
                elif text.startswith(f'/set@{me.username}'):
                    await set_trigger(event.message, text[len(f'/set@{me.username}'):])
                elif text.startswith('/set'):
                    await set_trigger(event.message, text[len('/set'):])
                elif text.startswith(f'/unset@{me.username}'):
                    await del_trigger(event.message, text[len(f'/unset@{me.username}'):])
                elif text.startswith('/unset'):
                    await del_trigger(event.message, text[len('/unset'):])
                elif text == '/list' or text == f'/list@{me.username}':
                    await list_trigger(event.message)
                elif text == '/list_full' or text == f'/list_full@{me.username}':
                    await list_trigger_full(event.message)
                elif text == '/help' or text == f'/help@{me.username}':
                    await help(event.message)
                else:
                    await reply_handler(event.message)
            assert await bot(functions.bots.SetBotCommandsRequest(
                scope=types.BotCommandScopeDefault(),
                lang_code='en',
                commands=[types.BotCommand(command, description) for command, description in [
                    ('ping', 'Test bot connectivity'),
                    ('add_whitelist', 'Add this group to whitelist (only admin)'),
                    ('del_whitelist', 'Delete this group from whitelist (only admin)'),
                    ('get_whitelist', 'List groups in whitelist (only admin)'),
                    ('create', 'Create a new GPT'),
                    ('set', 'Bind a GPT ID to a Trigger for this group'),
                    ('unset', 'Delete a Trigger for this group'),
                    ('list', 'List Triggers in this group'),
                    ('list_full', 'List Triggers in this group (full)'),
                    ('help', 'Show help message'),
                ]]
            ))
            await bot.run_until_disconnected()

asyncio.run(main())
