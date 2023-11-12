import aiohttp
import asyncio
import os
import json
import sys

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
                # print(json.dumps(results, ensure_ascii=False, indent=2))
                if 'items' not in results:
                    return 'No results found. Please change the keyword.'
                ret = []
                for item in results['items']:
                    obj = {}
                    if 'title' in item:
                        obj['title'] = item['title']
                    if 'link' in item:
                        obj['link'] = item['link']
                    if 'snippet' in item:
                        obj['snippet'] = item['snippet']
                    if len(obj):
                        ret.append(obj)
                return ret

async def main():
    s = Search()
    keyword = 'ChatGPT'
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
    print(json.dumps(await s.search(keyword), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
