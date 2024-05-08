# chatgpt-telegram-bot

![Demo](https://user-images.githubusercontent.com/10773481/235257879-ce91d3d3-769a-4089-addd-2ac9eea26f35.gif)

## Branches

- **master** - The OpenAI GPT models. Stable version.
- **dev** - The OpenAI GPT models. Development version.
- **system-prompt** - The OpenAI GPT models. Allows users to specify a system prompt arbitrarily.
- **gemini** - The Google Gemini models.
- **claude** - The Anthropic Claude models.
- **glm** - The Zhipu AI GLM-4 models.
- **gpts** - Allows customization of different prefixes corresponding to different system prompts, similar to ChatGPT's GPTs functionality.
- **plugins** - Utilizes the Function calling feature of GPT-4 to support Google and Bing searches, web browsing, YouTube subtitle downloads, Wolfram Alpha, and a calculator plugin.

## Features

1. **Real-time output streaming** - Receive AI model outputs while they are being generated using the message editing feature of Telegram.
1. **Model switching** - Easily change between models (e.g., GPT-4 and GPT-3.5-Turbo) using distinct message prefixes.
1. **Concurrent requests** - The bot can process multiple user queries simultaneously.
1. **Long message support** - For outputs exceeding Telegram's single message limit, the bot replies using multiple messages.
1. **Automatic retry** - The bot retries automatically in case of OpenAI API or Telegram API errors.
1. **Persistent and forkable chat history** - Chat logs are stored in a database, allowing users to reply to historical messages anytime.
1. **Message queueing** - Replies to generating messages are added to a processing queue.
1. **Whitelist management** - Bot administrators can enable or disable the bot in specific groups.
1. **Multimodal Support** - Supports images and text files as input.
1. **Rich Text Rendering** - The bot's output can be displayed as rich text in Telegram, such as code blocks and bold text.

## Usage

1. Obtain an OpenAI API key from https://platform.openai.com/account/api-keys
1. Create a Telegram bot by using BotFather via https://t.me/BotFather
1. Get `api_id` and `api_hash` from https://my.telegram.org/apps
1. Set **OpenAI API key** and **Telegram Bot Token** and **Telegram API ID and Hash** in the `docker-compose.yml` file.
1. In the `main.py` file, specify the **Telegram admin user ID** (you can send `/ping` to your bot to obtain your user ID) and the **prompt text**.
1. If you want to use the bot in a group, you must **disable privacy mode** in the bot settings.
1. Run `docker-compose up --build -d`
1. You can interact with your bot by initiating a new conversation with a message that starts with `$`. For example, you can type `$Hello`. By default, the bot uses the GPT-4 model. To switch to the GPT-3.5-Turbo model, simply start the message with `$$`.
1. To continue a conversation thread, reply to a previous message sent by the bot.
1. You can reply multiple times to the same message to "fork" the thread.
1. In a group, only the bot admin can use the `/add_whitelist` command to whitelist the group. To remove the group from the whitelist, the admin can use the `/del_whitelist` command.
1. Finally, in a private chat with the bot, you can use the `/get_whitelist` command to get a list of all whitelisted groups.

By following these steps, you can effectively use the ChatGPT Telegram bot for your needs.

Note: the above instructions are written by ChatGPT.
