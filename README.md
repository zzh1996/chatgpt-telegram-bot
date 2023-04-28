# chatgpt-telegram-bot

![Demo](https://user-images.githubusercontent.com/10773481/235257879-ce91d3d3-769a-4089-addd-2ac9eea26f35.gif)

## Features

1. **Real-time output streaming** - Receive AI model outputs while they are being generated using the message editing feature of Telegram.
2. **Model switching** - Easily change between models (e.g., GPT-4 and GPT-3.5-Turbo) using distinct message prefixes.
3. **Concurrent requests** - The bot can process multiple user queries simultaneously.
4. **Long message support** - For outputs exceeding Telegram's single message limit, the bot replies using multiple messages.
5. **Automatic retry** - The bot retries automatically in case of OpenAI API or Telegram API errors.
6. **Persistent and forkable chat history** - Chat logs are stored in a database, allowing users to reply to historical messages anytime.
7. **Message queueing** - Replies to generating messages are added to a processing queue.
8. **Whitelist management** - Bot administrators can enable or disable the bot in specific groups.

## Usage

1. Obtain an OpenAI API key from https://platform.openai.com/account/api-keys
2. Create a Telegram bot by using BotFather via https://t.me/BotFather
3. Set **OpenAI API key** and **Telegram Bot Token** in the `docker-compose.yml` file.
4. In the `main.py` file, specify the **Telegram admin user ID** (you can send `/ping` to your bot to obtain your user ID) and the **prompt text**.
5. If you want to use the bot in a group, you must **disable privacy mode** in the bot settings.
6. Run `docker-compose up --build -d`
7. You can interact with your bot by initiating a new conversation with a message that starts with `$`. For example, you can type `$Hello`. By default, the bot uses the GPT-4 model. To switch to the GPT-3.5-Turbo model, simply start the message with `$$`.
8. To continue a conversation thread, reply to a previous message sent by the bot.
9. You can reply multiple times to the same message to "fork" the thread.
10. In a group, only the bot admin can use the `/add_whitelist` command to whitelist the group. To remove the group from the whitelist, the admin can use the `/del_whitelist` command.
11. Finally, in a private chat with the bot, you can use the `/get_whitelist` command to get a list of all whitelisted groups.

By following these steps, you can effectively use the ChatGPT Telegram bot for your needs.

Note: the above instructions are written by ChatGPT.
