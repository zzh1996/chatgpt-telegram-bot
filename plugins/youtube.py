import asyncio
import json
import yt_dlp
import tempfile
import sys

class Youtube:
    functions = [{
        "name": "get_youtube_transcript",
        "description": "Get the full transcript of a YouTube video. You should use this instead of open_url for YouTube video URLs",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of YouTube video",
                },
            },
            "required": ["url"],
        }
    }]

    sub_preferences_en = ['en', 'en-US', 'en-GB', 'en-AU', 'en-CA', 'en-IN', 'en-IE']
    sub_preferences_zh = ['zh', 'zh-CN', 'zh-Hans', 'zh-Hant', 'zh-TW', 'zh-HK', 'zh-SG']
    autosub_preferences = ['en']

    def _get_youtube_transcript(self, url):
        if not yt_dlp.extractor.youtube.YoutubeIE.suitable(url):
            return {'error': 'URL is not a YouTube Video'}

        output = {}

        with yt_dlp.YoutubeDL() as ydl:
            info = ydl.extract_info(url, download=False, process=False)

            if 'title' in info:
                output['title'] = info['title']
            if 'uploader' in info:
                output['uploader'] = info['uploader']
            if 'description' in info:
                output['description'] = info['description']

            if 'title' in info and len([c for c in info['title'] if ord(c) in range(0x3400, 0xA000)]) >= 5:
                sub_preferences = self.sub_preferences_zh + self.sub_preferences_en
            else:
                sub_preferences = self.sub_preferences_en + self.sub_preferences_zh

            subtitle = None
            for lang in sub_preferences:
                if lang in info['subtitles']:
                    subtitle = 'sub', lang
                    break
            if subtitle is None:
                for lang in info['subtitles']:
                    if lang != 'live_chat':
                        subtitle = 'sub', lang
                        break
            if subtitle is None:
                for lang in self.autosub_preferences:
                    if lang in info['automatic_captions']:
                        subtitle = 'autosub', lang
                        break

            if subtitle is None:
                raise ValueError('No subtitle found')

        with tempfile.TemporaryDirectory() as tmpdir:
            options = {
                'outtmpl': f'{tmpdir}/output.%(ext)s',
                'skip_download': True,
                'subtitleslangs': [subtitle[1]],
                'subtitlesformat': 'json3',
            }
            if subtitle[0] == 'sub':
                options['writesubtitles'] = True
            elif subtitle[0] == 'autosub':
                options['writeautomaticsub'] = True

            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])

            with open(f'{tmpdir}/output.{subtitle[1]}.json3') as f:
                json3 = json.load(f)
                subtitle_lines = []
                for event in json3['events']:
                    if 'segs' in event:
                        line = ''.join([seg['utf8'] for seg in event['segs']]).strip()
                        if line:
                            subtitle_lines.append(line)
                subtitle_text = '\n'.join(subtitle_lines)

        output['transcript'] = subtitle_text

        return output

    async def get_youtube_transcript(self, url):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._get_youtube_transcript(url))

async def main():
    y = Youtube()
    url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    if len(sys.argv) > 1:
        url = sys.argv[1]
    print(json.dumps(await y.get_youtube_transcript(url), ensure_ascii=False))

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
