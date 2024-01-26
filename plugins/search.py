import aiohttp
import asyncio
import os
import json
import sys

class Search:
    functions = [{
        "name": "search",
        "description": "Search on Google and Bing and get the search results. Use concise keywords as query.",
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
        query = query.strip()
        if not query:
            return {'error': 'query is empty'}
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
                google_results = []
                if 'items' in results:
                    for item in results['items']:
                        obj = {}
                        if 'title' in item:
                            obj['title'] = item['title']
                        if 'link' in item:
                            obj['link'] = item['link']
                        if 'snippet' in item:
                            obj['snippet'] = item['snippet']
                        if len(obj):
                            google_results.append(obj)

        subscription_key = os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
        endpoint = os.environ['BING_SEARCH_V7_ENDPOINT'] + "/v7.0/search"
        params = {'q': query}
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.get(endpoint, headers=headers, params=params) as response:
                response.raise_for_status()
                results = await response.json()
                # print(json.dumps(results, ensure_ascii=False, indent=2))
                bing_results = []
                if 'webPages' in results and 'value' in results['webPages']:
                    for item in results['webPages']['value']:
                        obj = {}
                        if 'name' in item:
                            obj['title'] = item['name']
                        if 'url' in item:
                            obj['link'] = item['url']
                        if 'snippet' in item:
                            obj['snippet'] = item['snippet']
                        if len(obj):
                            bing_results.append(obj)

        return {'google_results': google_results, 'bing_results': bing_results}

async def main():
    s = Search()
    keyword = 'ChatGPT'
    if len(sys.argv) > 1:
        keyword = sys.argv[1]
    print(json.dumps(await s.search(keyword), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
