import asyncio
import sys
import aiohttp
import os

class WolframAlpha:
    functions = [{
        "name": "wolfram_alpha",
        "description": "Use Wolfram Alpha",
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The query. English only.",
                },
            },
            "required": ["input"],
            "additionalProperties": False,
        },
        "strict": True,
    }]

    def __init__(self):
        self.app_id = os.environ['WOLFRAM_ALPHA_APP_ID']

    async def wolfram_alpha(self, input):
        api_url = 'https://www.wolframalpha.com/api/v1/llm-api'
        params = {
            'appid': self.app_id,
            'input': input,
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            async with session.get(api_url, params=params) as response:
                if response.status != 501:
                    response.raise_for_status()
                result = await response.text()
                return result

async def main():
    w = WolframAlpha()
    input = 'integral sin(x)/x'
    if len(sys.argv) > 1:
        input = sys.argv[1]
    print((await w.wolfram_alpha(input))['result'])

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
