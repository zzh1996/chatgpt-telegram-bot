import aiohttp
import asyncio
import os
import json

class Search:
    functions = [{
        "name": "search",
        "description": "Search on Google and get the search results. Use concise keywords as query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        }
    }]

    def __init__(self):
        self.key = os.environ['GOOGLE_SEARCH_KEY']
        self.cx = os.environ['GOOGLE_SEARCH_CX']

    async def search(self, query):
        api_url = 'https://www.googleapis.com/customsearch/v1'
        params = {
            'key': self.key,
            'cx': self.cx,
            'q': query,
        }
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.get(api_url, params=params) as response:
                response.raise_for_status()
                results = await response.json()
                return [{'title': item['title'], 'link': item['link'], 'snippet': item['snippet']} for item in results['items']]

async def main():
    s = Search()
    print(json.dumps(await s.search('ChatGPT'), ensure_ascii=False))

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
