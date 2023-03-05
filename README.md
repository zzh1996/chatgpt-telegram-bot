# chatgpt-telegram-bot

To use ChatGPT Telegram bot, please follow the instructions below.

1. Obtain an OpenAI API key from https://platform.openai.com/account/api-keys
2. Create a Telegram bot by using BotFather via https://t.me/BotFather
3. Set **OpenAI API key** and **Telegram Bot Token** in the `docker-compose.yml` file.
4. In the `main.py` file, specify the **Telegram admin user ID** (you can send `/ping` to your bot to obtain your user ID) and the **prompt text**.
5. If you want to use the bot in a group, you must **disable privacy mode** in the bot settings.
6. Run `docker-compose up --build -d`
7. You can interact with your bot by initiating a new conversation with a message that starts with `$`. For example, you can type `$Hello`.
8. To continue a conversation thread, reply to a previous message sent by the bot.
9. You can reply multiple times to the same message to "fork" the thread.
10. In a group, only the bot admin can use the `/add_whitelist` command to whitelist the group. To remove the group from the whitelist, the admin can use the `/del_whitelist` command.
11. Finally, in a private chat with the bot, you can use the `/get_whitelist` command to get a list of all whitelisted groups.

By following these steps, you can effectively use the ChatGPT Telegram bot for your needs.

Note: the above instructions are written by ChatGPT.
