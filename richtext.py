from telethon import types

class RichText:
    def __init__(self, s=''):
        if isinstance(s, str):
            self.children = [{'type': 'text', 'content': s}]
        elif isinstance(s, RichText):
            self.children = s.children
        elif isinstance(s, list) and s and all(isinstance(c, dict) for c in s) and all('type' in c and 'content' in c for c in s):
            self.children = s
        else:
            raise ValueError()

    @classmethod
    def Raw(cls, s):
        return RichText([{'type': 'text', 'content': s}])

    @classmethod
    def Bold(cls, s):
        return RichText([{'type': 'bold', 'content': RichText(s)}])

    @classmethod
    def Code(cls, s):
        return RichText([{'type': 'code', 'content': s}])

    @classmethod
    def Pre(cls, s, language=''):
        return RichText([{'type': 'pre', 'content': s, 'language': language}])

    @classmethod
    def Href(cls, s, url):
        return RichText([{'type': 'href', 'content': RichText(s), 'url': url}])

    def __len__(self):
        return sum(len(c['content']) for c in self.children)

    def __str__(self):
        return f'RichText({self.children})'

    def __repr__(self):
        return f'RichText({self.children})'

    def __add__(self, value):
        if isinstance(value, RichText):
            if len(self) == 0:
                return value
            if len(value) == 0:
                return self
            self_last = self.children[-1].copy()
            value_first = value.children[0].copy()
            self_last_content = self_last['content']
            value_first_content = value_first['content']
            del self_last['content']
            del value_first['content']
            if self_last == value_first:
                self_last['content'] = self_last_content + value_first_content
                return RichText(self.children[:-1] + [self_last] + value.children[1:])
            return RichText(self.children + value.children)
        else:
            return self + RichText(value)

    def __radd__(self, value):
        return RichText(value) + self

    def __eq__(self, value):
        if not isinstance(value, RichText):
            if isinstance(value, str):
                return self == RichText(value)
            return False
        return self.children == value.children

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise NotImplementedError()
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise NotImplementedError()
        if start >= stop:
            return RichText()
        offset = 0
        new_children = []
        for c in self.children:
            l = len(c['content'])
            c_start = offset
            c_stop = c_start + l
            i_start = max(c_start, start)
            i_stop = min(c_stop, stop)
            if i_start < i_stop:
                new_c = c.copy()
                new_c['content'] = c['content'][i_start - offset : i_stop - offset]
                new_children.append(new_c)
            offset += l
        return RichText(new_children)

    # This function returns rich text that includes the raw Markdown content, with formatting tokens.
    # The function processes only a subset of Markdown and does NOT adhere to its respective specification.
    # The challenge in implementing a version that complies with the Markdown standard lies in the fact that common Python Markdown parser libraries do not provide character offset information for AST nodes in the source code.
    @classmethod
    def from_markdown(cls, markdown):
        lines = markdown.splitlines(keepends=True)
        in_pre = False
        pre_lang = None
        fence_len = None
        fence_prefix_spaces = None
        result = RichText()
        code = ''
        for line in lines:
            if in_pre:
                if line.strip() == '`' * fence_len:
                    if code and not code.isspace():
                        if pre_lang is None:
                            pre_lang = ''
                        result += RichText.Pre(code, pre_lang)
                    else:
                        result += code
                    code = ''
                    result += line
                    in_pre = False
                    pre_lang = None
                else:
                    if len(line) - len(line.lstrip(' ')) >= fence_prefix_spaces:
                        code += line[fence_prefix_spaces:]
                    else:
                        code += line.lstrip(' ')
            else:
                if line.strip().startswith('```'):
                    fence_len = len(line.strip()) - len(line.strip().lstrip('`'))
                    fence_prefix_spaces = len(line) - len(line.lstrip(' '))
                    pre_lang = line.strip().lstrip('`').strip()
                    if '`' not in pre_lang:
                        if not pre_lang:
                            pre_lang = None
                        result += line
                        in_pre = True
                    else:
                        result += process_line(line)
                else:
                    result += process_line(line)
        if in_pre:
            result += code
        return result

    def to_telegram(self, offset=0):
        def utf16len(s):
            return len(s.encode('utf-16-le')) // 2

        def strip_entity(s):
            lstripped = s[:len(s) - len(s.lstrip())]
            return utf16len(lstripped), utf16len(s.strip())

        entities = []
        text = ''
        for c in self.children:
            if c['type'] == 'text':
                text += c['content']
                offset += utf16len(c['content'])
            elif c['type'] == 'bold':
                t, e = c['content'].to_telegram(offset)
                text += t
                entities.extend(e)
                start, length = strip_entity(t)
                if length:
                    entities.append(types.MessageEntityBold(offset + start, length))
                offset += utf16len(t)
            elif c['type'] == 'code':
                text += c['content']
                start, length = strip_entity(c['content'])
                if length:
                    entities.append(types.MessageEntityCode(offset + start, length))
                offset += utf16len(c['content'])
            elif c['type'] == 'pre':
                text += c['content']
                start, length = strip_entity(c['content'])
                if length:
                    entities.append(types.MessageEntityPre(offset + start, length, c['language']))
                offset += utf16len(c['content'])
            elif c['type'] == 'href':
                t, e = c['content'].to_telegram(offset)
                text += t
                entities.extend(e)
                start, length = strip_entity(t)
                if length:
                    entities.append(types.MessageEntityTextUrl(offset + start, length, c['url']))
                offset += utf16len(t)
        return text, entities

def process_line(line):
    is_title = False
    prefix = line.strip().split(' ', 1)[0]
    if len(prefix) in range(1, 7) and all(c == '#' for c in prefix):
        is_title = True
    in_code = False
    in_bold = False
    in_escape = False
    buffer = ''
    result = RichText()
    for c in line:
        if in_escape:
            in_escape = False
            buffer += c
        elif in_code:
            if c == '`':
                result += RichText.Code(buffer)
                buffer = c
                in_code = False
            else:
                buffer += c
        elif c == '\\':
            in_escape = True
            buffer += c
        elif c == '`':
            buffer += c
            if in_bold:
                result += RichText.Bold(buffer)
                buffer = ''
            else:
                result += buffer
                buffer = ''
            in_code = True
        elif c == '*' and buffer and buffer[-1] == '*':
            if in_bold:
                buffer += c
                result += RichText.Bold(buffer)
                buffer = ''
                in_bold = False
            else:
                result += buffer[:-1]
                buffer = '**'
                in_bold = True
        else:
            buffer += c
    result += buffer
    if is_title:
        return RichText.Bold(result)
    else:
        return result
