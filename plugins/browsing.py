import asyncio
import aioboto3
import json
from bs4 import BeautifulSoup
import os

class Browsing:
    functions = [{
        "name": "open_url",
        "description": "Open a given URL and fetch the entire page as text",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to open",
                },
            },
            "required": ["url"],
        }
    }]

    def __init__(self):
        self.session = aioboto3.Session(
            aws_access_key_id=os.environ['aws_access_key_id'],
            aws_secret_access_key=os.environ['aws_secret_access_key'],
            region_name=os.environ['aws_region_name'],
        )

    async def open_url(self, url):
        if not (url.startswith('http://') or url.startswith('https://')):
            return {'error': 'URL not starting with http:// or https://'}

        async with self.session.client('lambda') as l:
            response = await l.invoke(
                FunctionName='fetch_webpage',
                Payload=json.dumps({'url': url}).encode(),
            )
            data = await response['Payload'].read()
        data = json.loads(data.decode())
        html = data['data']

        soup = BeautifulSoup(html, "lxml")
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        newlines = []
        for line in text.splitlines():
            newline = ' '.join(i for i in line.split())
            if newline:
                newlines.append(newline)
        text = '\n'.join(newlines)
        return {"result": {"title": data['title'], "content": text}}

async def main():
    b = Browsing()
    print(await b.open_url('https://www.ustc.edu.cn/'))

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
