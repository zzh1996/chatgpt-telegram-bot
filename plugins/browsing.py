import asyncio
import aioboto3
import json
from bs4 import BeautifulSoup
import os
import sys
import tiktoken

class Browsing:
    functions = [{
        "name": "open_url",
        "description": "Open a given URL and fetch the entire page as text, with hyperlinks converted into Markdown format",
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
        if 'data' not in data:
            return {'error': data}
        html = data['data']

        soup = BeautifulSoup(html, "lxml")
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        for a in soup.find_all("a", href=True):
            text = a.get_text().strip()
            if not text:
                continue
            href = a["href"]
            if len(href) < 256:
                a.string = f"[{text}]({href})"

        # for img in soup.find_all("img", src=True):
        #     src = img["src"]
        #     if not src.startswith("data:") and len(src) < 256:
        #         img.string = f"![{img.get('alt', 'img')}]({src})"

        text = soup.get_text()
        newlines = []
        for line in text.splitlines():
            newline = ' '.join(i for i in line.split())
            if newline:
                newlines.append(newline)
        text = '\n'.join(newlines)

        if len(tiktoken.encoding_for_model('gpt-4').encode(text)) >= 16384:
            soup = BeautifulSoup(html, "lxml")
            for script in soup(["script", "style", "noscript"]):
                script.decompose()

            text = soup.get_text()
            newlines = []
            for line in text.splitlines():
                newline = ' '.join(i for i in line.split())
                if newline:
                    newlines.append(newline)
            text = '\n'.join(newlines)

        return {"title": data['title'], "content": text}

async def main():
    b = Browsing()
    url = 'https://www.ustc.edu.cn/'
    if len(sys.argv) > 1:
        url = sys.argv[1]
    print(await b.open_url(url))

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
