#!/usr/bin/env python3
"""shuo import: shuo's import server, in the cloud.

What it does: brings a YouTube or Bilibili video (or just its sound) into a shuo lesson, with the video's Chinese
captions, or a timed transcript written here from its sound (SenseVoice, on this server's processor). It is the import
part of the shuo helper on my PC (host:shuo) as a web service, so shuo imports on any device, the phone too: shuo (the
page) reaches it as a claude.ai connector named "shuo import", with the same tools as the helper.

How it's reached: MCP over HTTP ("Streamable HTTP", answers as JSON) at  POST /mcp/<SHUO_KEY>.  SHUO_KEY is a long
secret set where it runs (on Hugging Face: the Space's Settings > Variables and secrets > New secret); without it
nothing is answered. GET / says whether it's up (and nothing else).

Run:   SHUO_KEY=... python shuo_import.py           (port 7860, or $PORT)
       python shuo_import.py --selftest             (checks every part, prints a report)
       python shuo_import.py --ytw                  (internal: one yt-dlp task, a request on stdin)
"""
import base64
import glob
import hmac
import json
import os
import queue as _queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

VERSION = '1.0.0'
HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get('PORT') or 7860)
DATA = os.environ.get('SHUO_DATA') or '/tmp/shuo-import'
CACHE = DATA
JOBS_DIR = os.path.join(DATA, 'jobs')
MODELS = os.environ.get('SHUO_MODELS') or os.path.join(HERE, 'models')
BIN = os.path.join(HERE, 'bin')
CONFIG = os.path.join(DATA, 'config.json')
KEEP_HOURS = 24                  # downloaded files are deleted after this long (the page has them by then)
CACHE_CAP = 10 * 1024 ** 3       # and the oldest go first when the jobs take more than this
MAX_CHUNK = 690 * 1024           # job_read: at most this many bytes per call (as base64, under 1 MB per answer)
MAX_SECONDS = 3 * 3600           # a video longer than this isn't imported here
MAX_BODY = 256 * 1024            # the largest request taken (tool arguments are small)
NO_WINDOW = 0
PROTOCOLS = ('2025-11-25', '2025-06-18', '2025-03-26', '2024-11-05')
_log_lock = threading.Lock()


def log(*parts):
    """To the server's own output (Hugging Face shows it under Logs). Never a key, never a whole answer."""
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(str(p) for p in parts)
    with _log_lock:
        try:
            sys.stderr.write(line[:2000] + '\n')
            sys.stderr.flush()
        except Exception:
            pass


class Ctx:
    """What the tools share: the jobs, the queue, the speech models."""

    def __init__(self):
        self.jobs = {}
        self.lock = threading.RLock()
        self.log = log
        self.version = VERSION
        self.store = {}


CTX = Ctx()


class ToolError(Exception):
    def __init__(self, msg, code='failed'):
        super().__init__(msg)
        self.code = code


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------- tools (the same as the PC helper's import tools)

def _obj(props, required=()):
    return {'type': 'object', 'properties': props, 'required': list(required)}


_JOB = {'type': 'string', 'description': 'The job id from yt_start.'}
TOOLS = [
    {'name': 'helper_info', 'title': 'Import server status',
     'description': 'Is the import server ready? Versions, the speech model, ffmpeg and the JavaScript runtime YouTube needs.',
     'inputSchema': _obj({}), 'annotations': {'readOnlyHint': True, 'openWorldHint': False}},
    {'name': 'yt_probe', 'title': 'Look up a video',
     'description': 'Look up one YouTube or Bilibili video: title, channel, length, a small picture, sizes per quality, its description and its Chinese captions.',
     'inputSchema': _obj({'url': {'type': 'string', 'description': 'A YouTube or Bilibili link (or a YouTube video id, or a BV number).'},
                          'thumb': {'type': 'boolean', 'description': 'Include the small picture (default true).'}}, ['url']),
     'annotations': {'readOnlyHint': True, 'openWorldHint': True}},
    {'name': 'yt_start', 'title': 'Bring a video into shuo',
     'description': 'Download a video, or only its sound, with its Chinese captions, or write a timed transcript from its sound. Returns a job; follow it with job_status, then read the file with job_read.',
     'inputSchema': _obj({
         'url': {'type': 'string'},
         'kind': {'type': 'string', 'enum': ['video', 'audio']},
         'quality': {'type': 'integer', 'description': 'Highest video height: 360, 480, 720 or 1080.'},
         'captions': {'type': ['string', 'null'], 'description': 'The caption track to use (a lang from yt_probe), or null.'},
         'transcript': {'type': 'string', 'enum': ['captions', 'generate', 'none']},
         'simplified': {'type': 'boolean', 'description': 'Turn Traditional characters into Simplified (always, for shuo).'},
         'music': {'type': 'boolean', 'description': 'A song: the best sound (otherwise a smaller file, enough for speech).'},
     }, ['url', 'kind']),
     'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'idempotentHint': False, 'openWorldHint': True}},
    {'name': 'job_status', 'title': 'Job progress',
     'description': 'Where a job is: state, plain-English stage, progress 0-1; when done, the media file details and the timed lines. With wait, the answer waits (at most 20 s) until the job moves on from seen.',
     'inputSchema': _obj({'job': _JOB, 'wait': {'type': 'number', 'description': 'Seconds to wait for a change (0-20).'},
                          'seen': {'type': 'object', 'description': 'What was shown last: state, stage, progress.'},
                          'step': {'type': 'number', 'description': 'How much further (0-1) counts as moving on (default 0.03).'}}, ['job']),
     'annotations': {'readOnlyHint': True, 'openWorldHint': False}},
    {'name': 'job_read', 'title': 'Read a downloaded file',
     'description': 'A piece of a finished job\'s file as base64 (part "media" or "thumb"), from offset, at most 690 KB.',
     'inputSchema': _obj({'job': _JOB, 'part': {'type': 'string', 'enum': ['media', 'thumb']},
                          'offset': {'type': 'integer'}, 'length': {'type': 'integer'}}, ['job']),
     'annotations': {'readOnlyHint': True, 'openWorldHint': False}},
    {'name': 'job_cancel', 'title': 'Stop a job',
     'description': 'Stop a download or transcript that is still running.',
     'inputSchema': _obj({'job': _JOB}, ['job']),
     'annotations': {'readOnlyHint': False, 'destructiveHint': False, 'idempotentHint': True, 'openWorldHint': False}},
]


def _once(payload):
    """A big answer as one JSON text (not text and structured content both, which would double a piece of the file)."""
    if isinstance(payload, dict) and len(str(payload.get('data') or '')) + len(json.dumps(payload.get('result') or '')) > 20000:
        return json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    return payload


def call(name, args):
    if not isinstance(args, dict):
        args = {}
    if name == 'helper_info':
        return helper_info()
    if name == 'yt_probe':
        return probe(str(args.get('url') or ''), args.get('thumb', True) is not False)
    if name == 'yt_start':
        return start_job(args)
    if name == 'job_status':
        return _once(job_status(str(args.get('job') or ''), args.get('wait') or 0, args.get('seen'), args.get('step') or 0.03))
    if name == 'job_read':
        return _once(job_read(str(args.get('job') or ''), args.get('part') or 'media', args.get('offset') or 0, args.get('length')))
    if name == 'job_cancel':
        return job_cancel(str(args.get('job') or ''))
    raise ToolError('The import server has no tool called %s.' % name, 'bad_tool')


# ---------------------------------------------------------------- the tools it uses

def read_config():
    return {}


def _run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return (p.stdout or b'').decode('utf-8', 'replace') + (p.stderr or b'').decode('utf-8', 'replace')
    except (OSError, subprocess.SubprocessError):
        return ''


def _cached(key, fn):
    st = CTX.store
    if key not in st:
        st[key] = fn()
    return st[key]


def ffmpeg_path():
    def find():
        for c in (os.path.join(BIN, 'ffmpeg'), shutil.which('ffmpeg')):
            if c and os.path.isfile(c):
                return c
        try:
            import imageio_ffmpeg
            p = imageio_ffmpeg.get_ffmpeg_exe()
            if p and os.path.isfile(p):
                return p
        except Exception:
            pass
        return None
    return _cached('ffmpeg', find)


def _version_tuple(s):
    return tuple(int(x) for x in re.findall(r'\d+', s or '')[:3])


def js_runtimes():
    """What yt-dlp may use to answer YouTube's JavaScript challenges: deno (preferred), or node 22+."""
    def find():
        out = {}
        for p in (os.path.join(BIN, 'deno'), shutil.which('deno'), os.path.expanduser('~/.deno/bin/deno')):
            if p and os.path.isfile(p):
                m = re.search(r'deno (\d+\.\d+\.\d+)', _run([p, '--version']))
                if m and _version_tuple(m.group(1)) >= (2, 0, 0):
                    out['deno'] = {'path': p, 'version': m.group(1)}
                    break
        p = shutil.which('node')
        if p:
            m = re.search(r'v(\d+\.\d+\.\d+)', _run([p, '--version']))
            if m and _version_tuple(m.group(1)) >= (22, 0, 0):
                out['node'] = {'path': p, 'version': m.group(1)}
        return out
    return _cached('js', find)


def _installed(dist):
    try:
        from importlib import metadata
        return metadata.version(dist)
    except Exception:
        return None


# yt-dlp follows YouTube's changes; an old one is the usual reason a download is refused. Once a day (and after a
# download every way refused, at most once an hour) the server asks pip for the newest yt-dlp, in the background; each
# task runs yt-dlp in a new process, so the next one uses it.
UPDATE_FILE = os.path.join(DATA, 'update.json')
UPDATE_EVERY = 24 * 3600


def _read_update():
    try:
        with open(UPDATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def update_ytdlp(force=False, min_gap=UPDATE_EVERY):
    st = CTX.store
    with CTX.lock:
        if st.get('updating'):
            return False
        if not force and time.time() - _read_update().get('checked', 0) < min_gap:
            return False
        st['updating'] = True
    try:
        before = _installed('yt-dlp')
        pip = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '-q']
        if not os.access(os.path.dirname(os.__file__), os.W_OK):
            pip.append('--user')
        ok, err = True, ''
        for cmd in (pip + ['-U', '--pre', 'yt-dlp[default,curl-cffi]'],):
            try:
                p = subprocess.run(cmd, capture_output=True, timeout=900)
                if p.returncode != 0:
                    ok, err = False, (p.stderr or b'').decode('utf-8', 'replace')[-600:]
            except (OSError, subprocess.SubprocessError) as e:
                ok, err = False, repr(e)
        after = _installed('yt-dlp')
        rec = {'checked': time.time(), 'before': before, 'after': after, 'ok': ok}
        if not ok:
            rec['error'] = err
        try:
            os.makedirs(DATA, exist_ok=True)
            with open(UPDATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(rec, f)
        except OSError:
            pass
        log('yt-dlp update:', before, '->', after, 'ok' if ok else 'failed: ' + err.strip()[-300:])
        return ok
    finally:
        st['updating'] = False


# ---------------------------------------------------------------- yt-dlp in a process of its own
# Every look-up and download runs this file again with --ytw: it always uses the newest yt-dlp installed, and nothing
# yt-dlp does (an error, a hang) can stop the server. The server reads its answers line by line.

class YtwError(Exception):
    pass


def ytw(req, job=None, on_line=None, quiet_for=300, total=None):
    """Run one yt-dlp task (see ytw_main). on_line(obj) hears each line as it comes; the answer is the 'result'.
    Stopped when the job is cancelled, when nothing comes for quiet_for seconds, or after total seconds."""
    t0 = time.time()
    while CTX.store.get('updating') and time.time() - t0 < 240:
        time.sleep(0.5)           # yt-dlp is being updated: wait for the new one
    os.makedirs(DATA, exist_ok=True)
    ck = _cookie_copy()
    if ck:   # (each task its own copy: yt-dlp writes its cookies back when it ends)
        req = dict(req, opts=dict(req.get('opts') or {}, cookiefile=ck))
    try:
        return _ytw_run(req, job, on_line, quiet_for, total, t0)
    finally:
        if ck:
            try:
                os.remove(ck)
            except OSError:
                pass


# Optional: SHUO_COOKIES, a cookies.txt (Netscape format) of a spare YouTube or Bilibili account, for when the site
# keeps asking this server to prove it isn't a bot (or Bilibili's subtitles, which need an account). Kept only in the
# secret and in a private copy for each task, deleted when the task ends; never in an answer or the log.
COOKIE_FILES = ('/etc/secrets/cookies.txt', '/etc/secrets/SHUO_COOKIES')   # (Render's Secret Files; HF: the SHUO_COOKIES secret)


def _cookies():
    """The cookies.txt (Netscape format), from the SHUO_COOKIES secret or a secret file (SHUO_COOKIES_FILE, or Render's
    /etc/secrets/cookies.txt). A web form may have turned its tabs into spaces: each line of 7 fields gets its tabs back."""
    t = (os.environ.get('SHUO_COOKIES') or '').strip()
    if not t:
        for f in ((os.environ.get('SHUO_COOKIES_FILE') or '').strip(),) + COOKIE_FILES:
            try:
                if f and os.path.isfile(f) and os.path.getsize(f) < 1024 * 1024:
                    t = open(f, encoding='utf-8', errors='replace').read().strip()
                    if t:
                        break
            except OSError:
                continue
    if not t:
        return ''
    out, n = [], 0
    for ln in t.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        ln = ln.strip()
        if not ln or (ln.startswith('#') and not ln.startswith('#HttpOnly_')):
            continue
        parts = ln.split('\t') if '\t' in ln else ln.split()
        if len(parts) == 6:   # (a cookie with an empty value)
            parts.append('')
        if len(parts) != 7 or not re.match(r'^\d+$', parts[4].strip() or '0'):
            continue
        out.append('\t'.join(x.strip() for x in parts))
        n += 1
    return '# Netscape HTTP Cookie File\n' + '\n'.join(out) if n else ''


def _logins():
    """Which sites the cookies log in to (only yes or no: never what they are)"""
    t = _cookies()
    return {'bilibili': bool(re.search(r'bilibili\.com\t[^\n]*\tSESSDATA\t\S', t)),
            'youtube': bool(re.search(r'youtube\.com\t[^\n]*\t(?:__Secure-[13]PSID|SID|LOGIN_INFO)\t\S', t))}


def _cookie_copy():
    t = _cookies()
    if not t:
        return ''
    d = os.path.join(DATA, '.cookies')
    os.makedirs(d, mode=0o700, exist_ok=True)
    path = os.path.join(d, uuid.uuid4().hex + '.txt')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(t + '\n')
    return path


def _ytw_run(req, job, on_line, quiet_for, total, t0):
    cmd = [sys.executable, '-X', 'utf8', '-u', os.path.abspath(__file__), '--ytw']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=DATA)
    if job is not None:
        job['proc'] = p
    errs = []
    t_err = threading.Thread(target=lambda: errs.append(p.stderr.read()), daemon=True)
    t_err.start()
    state = {'last': time.time(), 'why': None}

    def watchdog():
        while p.poll() is None:
            now = time.time()
            if job is not None and job.get('cancel'):
                state['why'] = 'cancel'
            elif now - state['last'] > quiet_for:
                state['why'] = 'quiet'
            elif total and now - t0 > total:
                state['why'] = 'total'
            if state['why']:
                try:
                    p.kill()
                except Exception:
                    pass
                return
            time.sleep(0.5)
    threading.Thread(target=watchdog, daemon=True).start()
    result, error = None, None
    try:
        try:
            p.stdin.write(json.dumps(req).encode('utf-8'))
            p.stdin.close()
        except OSError:
            pass
        for raw in p.stdout:
            state['last'] = time.time()
            try:
                obj = json.loads(raw.decode('utf-8', 'replace'))
            except ValueError:
                continue
            if 'warn' in obj:
                w = str(obj['warn'])
                if 'impersonat' not in w:
                    log('yt-dlp:', w[:400])
                if job is not None:
                    job.setdefault('warnings', []).append(w[:300])
            elif 'last' in obj:
                if job is not None:
                    job['last'] = obj['last']
            elif 'result' in obj:
                result = obj['result']
            elif 'error' in obj:
                error = str(obj['error'])
            elif on_line:
                on_line(obj)
        p.wait()
    finally:
        if job is not None:
            job.pop('proc', None)
    t_err.join(3)
    if state['why'] == 'cancel' or (job is not None and job.get('cancel')):
        raise Cancelled()
    if state['why'] in ('quiet', 'total'):
        raise YtwError('timed out: the site took too long to answer')
    if error is not None:
        raise YtwError(error)
    if result is None:
        tail = (errs[0] if errs else b'').decode('utf-8', 'replace').strip()[-500:]
        log('ytw ended without an answer:', p.returncode, tail)
        raise YtwError('The download tool stopped unexpectedly (%s). %s' % (p.returncode, tail[-200:]))
    return result


def base_opts(extra=None):
    o = {
        'noplaylist': True,
        'socket_timeout': 30, 'retries': 8, 'fragment_retries': 15, 'extractor_retries': 3,
        'file_access_retries': 5, 'concurrent_fragment_downloads': 4, 'http_chunk_size': 10 * 1024 * 1024,
    }
    rt = js_runtimes()
    if rt:
        o['js_runtimes'] = {k: {'path': v['path']} for k, v in rt.items()}
    ff = ffmpeg_path()
    if ff:
        o['ffmpeg_location'] = ff
    if extra:
        o.update(extra)
    return o


# ---------------------------------------------------------------- links: YouTube's and Bilibili's only
# Only a link to one YouTube or Bilibili video is taken (at most 300 characters), so nothing sent here can make the
# server fetch anything else. A share text ("【title】 https://b23.tv/…") gives the link in it; a b23.tv short link is
# followed here (one request to b23.tv) and must lead to a Bilibili video.

_YT_ID = re.compile(r'^[A-Za-z0-9_-]{11}$')
_BV = re.compile(r'^(BV[0-9A-Za-z]{10})$')


def site_of(url):
    return 'Bilibili' if re.match(r'^https?://(?:[a-z0-9-]+\.)?(?:bilibili\.com|b23\.tv)/', url or '', re.I) else 'YouTube'


def _b23(u):
    """A b23.tv short link: where it leads (a Bilibili video's link), or None."""
    try:
        req = urllib.request.Request(u, method='GET', headers={'User-Agent': 'Mozilla/5.0 (shuo import)'})
        with urllib.request.urlopen(req, timeout=12) as r:
            return r.geturl()
    except Exception:
        return None


def norm_url(url, follow=True):
    u = (url or '').strip()
    if not u:
        raise ToolError('Paste a YouTube or Bilibili link first.', 'bad_url')
    if len(u) > 300:
        raise ToolError('That link is too long. Paste the link of one video.', 'bad_url')
    u = re.sub(r'^(?!https?://)[htps]{0,5}:?//', 'https://', u, flags=re.I)   # (the start of the link cut off: "ttps://…")
    m = re.search(r'https?://[^\s"<>【】]+', u)
    if m and not re.match(r'^https?://', u, re.I):
        u = m.group(0)             # a share text: the link in it
    if _YT_ID.match(u):
        return 'https://www.youtube.com/watch?v=' + u
    if _BV.match(u):
        return 'https://www.bilibili.com/video/' + u + '/'
    if re.match(r'^av\d{1,12}$', u, re.I):
        return 'https://www.bilibili.com/video/' + u.lower() + '/'
    if not re.match(r'^https?://', u, re.I):
        u = 'https://' + u
    m = re.match(r'^https?://(?:www\.|m\.|music\.)?(?:youtube\.com|youtube-nocookie\.com|youtu\.be)(/.*)?$', u, re.I)
    if m:
        path = m.group(1) or '/'
        vid = None
        mm = re.search(r'[?&]v=([A-Za-z0-9_-]{11})', path)
        if mm:
            vid = mm.group(1)
        else:
            mm = re.match(r'^/(?:shorts/|live/|embed/|v/|e/)?([A-Za-z0-9_-]{11})(?:[/?#&]|$)', path)
            if mm and not path.startswith(('/playlist', '/channel', '/c/', '/@', '/user', '/results', '/feed')):
                vid = mm.group(1)
        if not vid:
            if 'list=' in path:
                raise ToolError('That is a playlist link. Paste the link of one video.', 'bad_url')
            raise ToolError('That doesn\'t look like a link to one YouTube video.', 'bad_url')
        return 'https://www.youtube.com/watch?v=' + vid
    if re.match(r'^https?://b23\.tv/[0-9A-Za-z]{4,16}/?(?:\?.*)?$', u, re.I):
        if not follow:
            return u
        to = _b23(u)
        if not to or 'b23.tv' in to:
            raise ToolError('That Bilibili short link couldn\'t be followed. Paste the video\'s full link.', 'bad_url')
        return norm_url(to, follow=False)
    m = re.match(r'^https?://(?:www\.|m\.)?bilibili\.com(/[^#]*)?$', u, re.I)
    if m:
        path = m.group(1) or '/'
        path = re.sub(r'^/[a-z]{2}(?:-[a-z]{2,4})?(?=/video/)', '', path, flags=re.I)   # (a language in it: /zh-Hans/video/…)
        mv = re.match(r'^/video/(BV[0-9A-Za-z]{10}|av\d{1,12})/?(\?.*)?$', path, re.I)
        if not mv:
            if re.match(r'^/(?:bangumi|cheese|medialist|list|festival)/', path) or 'space.bilibili' in u:
                raise ToolError('That is a show, a course or a list. Paste the link of one ordinary video.', 'bad_url')
            raise ToolError('That doesn\'t look like a link to one Bilibili video.', 'bad_url')
        vid = mv.group(1)
        if vid.lower().startswith('av'):
            vid = vid.lower()
        pm = re.search(r'[?&]p=(\d{1,4})\b', mv.group(2) or '')
        return 'https://www.bilibili.com/video/%s/' % vid + ('?p=%s' % pm.group(1) if pm and pm.group(1) != '1' else '')
    raise ToolError('Paste a YouTube or Bilibili link.', 'bad_url')


# ---------------------------------------------------------------- captions offered

_ZH = re.compile(r'^(zh|cmn|yue|wuu|hak|nan|lzh)([-_][A-Za-z0-9-]+)?$', re.I)


def _caption_rank(lang, auto):
    l = lang.lower()
    if auto:
        return 50
    if l in ('zh-hans', 'zh-cn', 'zh-sg', 'zh', 'cmn', 'cmn-hans', 'zh-hans-cn'):
        return 0
    if l in ('zh-hant', 'zh-tw', 'zh-hk', 'zh-mo', 'cmn-hant', 'zh-hant-tw', 'zh-hant-hk'):
        return 10
    if l.startswith(('zh', 'cmn')):
        return 5
    return 20   # yue, wuu, ...


def caption_tracks(info, site='YouTube'):
    out = []
    for lang, fmts in (info.get('subtitles') or {}).items():
        low = lang.lower()
        if low in ('live_chat', 'danmaku'):
            continue                      # viewers' chat and Bilibili's danmaku are never the lesson's text
        auto = False
        base = lang
        if low.startswith('ai-'):         # Bilibili's AI subtitles
            auto, base = True, lang[3:]
        if not _ZH.match(base):
            continue
        name = next((f.get('name') for f in (fmts or []) if f.get('name')), '') or lang
        out.append({'lang': lang, 'name': name, 'auto': auto, 'rank': _caption_rank(base, auto)})
    if site == 'YouTube':
        for lang, fmts in (info.get('automatic_captions') or {}).items():
            # only YouTube's own speech recognition in the video's language ("-orig"), never machine translations
            if not lang.endswith('-orig') or not _ZH.match(lang[:-5]):
                continue
            name = next((f.get('name') for f in (fmts or []) if f.get('name')), '') or lang
            out.append({'lang': lang, 'name': name, 'auto': True, 'rank': _caption_rank(lang, True)})
    out.sort(key=lambda t: (t['rank'], t['lang']))
    for t in out:
        del t['rank']
    return out


# ---------------------------------------------------------------- sizes (as download() picks: H.264 picture and AAC sound first)

def _fmt_size(f, duration):
    s = f.get('filesize') or f.get('filesize_approx')
    if not s and f.get('tbr') and duration:
        s = f['tbr'] * 1000 / 8 * duration
    return int(s or 0)


def _is_h264(f):
    return str(f.get('vcodec') or '').lower().startswith(('avc1', 'h264', 'avc'))


def _is_aac(f):
    return str(f.get('acodec') or '').lower().startswith(('mp4a', 'aac'))


def _speech_audio(audio):
    """The sound a download for speech takes (abr:100): the most kbps up to 100, else the fewest above."""
    under = [f for f in audio if (f.get('abr') or f.get('tbr') or 0) and (f.get('abr') or f.get('tbr')) <= 100]
    if under:
        return max(under, key=lambda f: f.get('abr') or f.get('tbr') or 0)
    return min(audio, key=lambda f: f.get('abr') or f.get('tbr') or 1e9) if audio else None


def size_guesses(info):
    """About how big each download will be, the way download() picks: at each height the highest the video has up to
    it, an H.264 picture where there is one (a phone plays it), with AAC sound for speech (about 100 kbps; never
    YouTube's DRC copy). 'audio' is the sound alone, 'music' the best sound (This is music)."""
    fmts = info.get('formats') or []
    dur = info.get('duration') or 0
    audio = [f for f in fmts if f.get('vcodec') in (None, 'none') and f.get('acodec') not in (None, 'none')
             and 'drc' not in str(f.get('format_id', ''))]
    aac = [f for f in audio if _is_aac(f)] or audio
    sp = _speech_audio(aac)
    a_size = _fmt_size(sp, dur) if sp else 0
    best_a = max(aac, key=lambda f: f.get('abr') or f.get('tbr') or 0) if aac else None
    out = {'audio': a_size, 'music': _fmt_size(best_a, dur) if best_a else 0}
    vids = [f for f in fmts if f.get('vcodec') not in (None, 'none') and f.get('height')]
    heights = sorted({f['height'] for f in vids})
    for h in (360, 480, 720, 1080):
        cand = [f for f in vids if f['height'] <= h]
        if not cand:
            continue
        pick = [f for f in cand if _is_h264(f)] or cand
        top = max(f['height'] for f in pick)
        at = [f for f in pick if f['height'] == top and _fmt_size(f, dur) > 0] or [f for f in pick if f['height'] == top]
        best = min(at, key=lambda f: _fmt_size(f, dur) or 1e15)
        muxed = best.get('acodec') not in (None, 'none')
        out[str(h)] = _fmt_size(best, dur) + (0 if muxed else a_size)
    return out, heights


def friendly(msg, site='YouTube'):
    m = str(msg or '')
    low = m.lower()
    S = site
    pairs = [
        ('confirm your age', 'This video is age-restricted, so %s won\'t let it be downloaded without signing in.' % S),
        ('inappropriate for some users', 'This video is age-restricted, so %s won\'t let it be downloaded without signing in.' % S),
        ('not a bot', '%s is asking the import server to prove it isn\'t a bot, so it won\'t download right now. Try again later, or another video.' % S),
        ('sign in to confirm', '%s is asking the import server to prove it isn\'t a bot, so it won\'t download right now. Try again later, or another video.' % S),
        ('private video', 'This video is private.'),
        ('members-only', 'This video is for channel members only.'),
        ('join this channel', 'This video is for channel members only.'),
        ('premium member', 'This video is only for %s\'s paying members.' % S),
        ('premieres in', 'This video hasn\'t premiered yet.'),
        ('is live', 'This is a live stream. Import it after it ends.'),
        ('live event will begin', 'This live stream hasn\'t started yet.'),
        ('not available in your country', 'This video isn\'t shown in the country the import server is in.'),
        ('geo-restrict', 'This video isn\'t shown in the country the import server is in (or it was deleted).'),
        ('geo restrict', 'This video isn\'t shown in the country the import server is in (or it was deleted).'),
        ('copyright', 'This video was taken down.'),
        ('video unavailable', '%s says this video is unavailable.' % S),
        ('removed by the uploader', 'This video was removed by the person who uploaded it.'),
        ('http error 412', '%s is turning requests away for a while. Try again in a few minutes.' % S),
        ('http error 429', '%s is turning requests away for a while (too many). Try again in a few minutes.' % S),
        ('too many requests', '%s is turning requests away for a while (too many). Try again in a few minutes.' % S),
        ('unable to download webpage', '%s couldn\'t be reached from the import server. Try again in a minute.' % S),
        ('name or service not known', '%s couldn\'t be reached from the import server. Try again in a minute.' % S),
        ('timed out', '%s took too long to answer. Try again.' % S),
        ('http error 403', '%s refused the download. Try again, or choose a lower quality.' % S),
        ('requested format is not available', 'That quality isn\'t available for this video. Choose another one.'),
        ('no space left', 'The import server\'s disk is full. Try again later.'),
        ('ffmpeg', 'Joining the video and its sound failed (ffmpeg). Try Audio only, or a lower quality.'),
        ('js challenge', 'YouTube\'s download check couldn\'t be answered here. Try again later.'),
        ('n challenge', 'YouTube\'s download check couldn\'t be answered here. Try again later.'),
        ('unsupported url', 'That link isn\'t a video the import server knows how to download.'),
    ]
    for k, v in pairs:
        if k in low:
            return v
    m = re.sub(r'^ERROR:\s*', '', m)
    m = re.sub(r'\[[^\]]+\]\s*[A-Za-z0-9_-]{6,14}:\s*', '', m)
    return 'The download failed: ' + m[:300]


def _small_thumb(uri):
    """The picture as a small JPEG (384 px wide) for the page: a data: URI under 100 KB, or None."""
    if not uri or not isinstance(uri, str) or not uri.startswith('data:image/'):
        return None
    ff = ffmpeg_path()
    try:
        head, b64 = uri.split(',', 1)
        raw = base64.b64decode(b64)
    except Exception:
        return None
    if ff:
        d = os.path.join(DATA, 'thumbs')
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, uuid.uuid4().hex + '.img')
        dst = src[:-4] + '.jpg'
        try:
            with open(src, 'wb') as f:
                f.write(raw)
            _run([ff, '-nostdin', '-hide_banner', '-v', 'error', '-y', '-i', src, '-vf', 'scale=384:-2', '-frames:v', '1', '-q:v', '5', dst], timeout=20)
            if os.path.isfile(dst) and 400 < os.path.getsize(dst) < 100000:
                with open(dst, 'rb') as f:
                    return 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode('ascii')
        finally:
            for p in (src, dst):
                try:
                    os.remove(p)
                except OSError:
                    pass
    return uri if len(uri) < 110000 else None


def _desc(info):
    d = info.get('description')
    return d[:2000] if isinstance(d, str) else ''


def probe(url, with_thumb=True):
    url = norm_url(url)
    site = site_of(url)
    try:
        # Bilibili's subtitles are looked up only when yt-dlp is asked for them (writesubtitles: with skip_download and
        # no download, nothing is written); YouTube's come with the video's info anyway
        o = {'skip_download': True}
        if site == 'Bilibili':
            o.update({'writesubtitles': True, 'subtitleslangs': ['all', '-danmaku']})
        res = ytw({'op': 'info', 'url': url, 'opts': base_opts(o), 'thumb': with_thumb}, total=120)
    except Cancelled:
        raise ToolError('Stopped.', 'cancelled')
    except YtwError as e:
        log('probe failed:', site, str(e)[:300])
        raise ToolError(friendly(e, site), 'yt')
    info = res.get('info') or {}
    if info.get('_type') == 'playlist':
        raise ToolError('That is a playlist. Paste the link of one video.', 'bad_url')
    live = info.get('live_status')
    if live in ('is_live', 'is_upcoming'):
        raise ToolError('This is a live stream. Import it after it ends.' if live == 'is_live' else 'This video hasn\'t started yet.', 'live')
    if (info.get('duration') or 0) > MAX_SECONDS:
        raise ToolError('It\'s %s long: the import server takes videos up to 3 hours.' % _hms(info.get('duration')), 'too_long')
    sizes, heights = size_guesses(info)
    return {
        'url': url, 'site': site, 'id': info.get('id'), 'title': info.get('title') or '',
        'channel': info.get('channel') or info.get('uploader') or '', 'duration': info.get('duration') or 0,
        'language': info.get('language') or '', 'thumb': _small_thumb(res.get('thumb')) if with_thumb else None,
        'captions': caption_tracks(info, site), 'heights': heights, 'sizes': sizes,
        'webpage_url': info.get('webpage_url') or url, 'description': _desc(info),
        # Bilibili shows its (AI) subtitles only to someone logged in: without a Bilibili login here, say so
        'need_login': bool(site == 'Bilibili' and not caption_tracks(info, site) and not _logins()['bilibili']),
    }


# ---------------------------------------------------------------- jobs (one runs at a time; files kept a day)

def _job_dir(jid):
    if not re.match(r'^[a-z0-9-]{6,40}$', jid or ''):
        raise ToolError('That isn\'t a job id.', 'bad_job')
    return os.path.join(JOBS_DIR, jid)


PUBLIC = ('id', 'url', 'kind', 'quality', 'captions', 'transcript', 'simplified', 'state', 'stage', 'progress',
          'created', 'updated', 'error', 'title', 'duration', 'music', 'tries', 'site')
SAVED = PUBLIC + ('user',)


def _save(job, result=None):
    d = job['dir']
    try:
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, 'job.json.part')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({k: job.get(k) for k in SAVED}, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(d, 'job.json'))
        if result is not None:
            tmp = os.path.join(d, 'result.json.part')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
            os.replace(tmp, os.path.join(d, 'result.json'))
    except OSError:
        log('save failed', traceback.format_exc())


def _set(job, state=None, stage=None, progress=None):
    if state:
        job['state'] = state
    if stage is not None:
        job['stage'] = stage
    if progress is not None:
        job['progress'] = max(0.0, min(1.0, float(progress)))
    job['updated'] = time.time()
    if state or job['updated'] - job.get('_saved', 0) > 2:
        job['_saved'] = job['updated']
        _save(job)


def _gone():
    return ToolError('That job isn\'t on the import server any more (downloads are kept for a day). Import the video again.', 'no_job')


def _load_job(jid):
    """A job of the one asking (anyone else's is as if it weren't there)."""
    j = _load_any(jid)
    if (j.get('user') or 'owner') != who():
        raise _gone()
    return j


def _load_any(jid):
    with CTX.lock:
        j = CTX.jobs.get(jid)
        if j:
            return j
        d = _job_dir(jid)
        p = os.path.join(d, 'job.json')
        if not os.path.exists(p):
            raise _gone()
        try:
            with open(p, 'r', encoding='utf-8') as f:
                j = json.load(f)
        except (OSError, ValueError):
            raise ToolError('That job\'s record is being written. Ask again in a moment.', 'busy')
        j['dir'] = d
        if j.get('state') not in ('done', 'error', 'cancelled'):
            j['state'] = 'error'
            j['error'] = 'The import server was restarted while this was running. Import the video again.'
        CTX.jobs[jid] = j
        return j


_RUNNING = ('queued', 'downloading', 'merging', 'captions', 'transcribing')


NO_TRANSCRIPTS = ('This video has no Chinese captions, and this import server can\'t write transcripts (it has no speech model). '
                  'Pick a video with Chinese captions.')


def start_job(args):
    url = norm_url(str(args.get('url') or ''))
    kind = args.get('kind')
    if kind not in ('video', 'audio'):
        raise ToolError('Choose Video or Audio only.', 'bad_args')
    q = args.get('quality') or 720
    try:
        q = int(q)
    except (TypeError, ValueError):
        q = 720
    q = min((360, 480, 720, 1080), key=lambda h: abs(h - q))
    tr = args.get('transcript') or ('captions' if args.get('captions') else 'none')
    if tr not in ('captions', 'generate', 'none'):
        tr = 'none'
    cap = args.get('captions') or None
    if cap is not None and (not isinstance(cap, str) or not re.match(r'^[A-Za-z0-9_-]{1,40}$', cap)):
        cap = None
    if tr == 'captions' and not cap:
        tr = 'none'
    if tr == 'generate':   # (a server without the speech model, as on a small free host: said before anything downloads)
        mp = model_paths()
        if not (mp['model'] and mp['tokens'] and mp['vad']):
            raise ToolError(NO_TRANSCRIPTS, 'no_model')
    jid = time.strftime('%y%m%d-%H%M%S-') + uuid.uuid4().hex[:16]
    job = {'id': jid, 'url': url, 'site': site_of(url), 'kind': kind, 'quality': q, 'captions': cap, 'transcript': tr,
           'simplified': True, 'state': 'queued', 'stage': 'Waiting to start',
           'progress': 0.0, 'created': time.time(), 'updated': time.time(), 'error': None, 'cancel': False,
           'dir': os.path.join(JOBS_DIR, jid), 'title': '', 'duration': 0, 'music': bool(args.get('music')),
           'first_try': 0, 'user': who()}
    with CTX.lock:
        mine = sum(1 for j in CTX.jobs.values() if j.get('state') in _RUNNING and (j.get('user') or 'owner') == who())
        busy = sum(1 for j in CTX.jobs.values() if j.get('state') in _RUNNING)
        if mine >= 3:
            raise ToolError('Three of your imports are already waiting on the import server. Wait for one to finish.', 'busy')
        if busy >= 12:
            raise ToolError('The import server is busy with other imports. Try again in a few minutes.', 'busy')
        daily = who_daily()
        if daily is not None and _usage() >= daily:
            raise ToolError('That\'s %d imports today, the most for one day here. Try again tomorrow.' % daily, 'limit')
        if daily is not None:
            _usage(1)
        CTX.jobs[jid] = job
    _save(job)
    CTX.store['queue'].put(jid)
    threading.Thread(target=cleanup_cache, daemon=True).start()
    log('job', jid, 'for', who(), job['site'], kind, q, tr)
    return {'job': jid, 'state': 'queued'}


_ENDED = ('done', 'error', 'cancelled')


def job_status(jid, wait=0, seen=None, step=0.03):
    """With wait (seconds, at most 20) the answer waits until the job moves on: it ends, changes stage, or goes step
    further than seen ({state, stage, progress}, what the page showed last): a call every few seconds, not every second."""
    j = _load_job(jid)
    try:
        wait = max(0.0, min(20.0, float(wait or 0)))
    except (TypeError, ValueError):
        wait = 0.0
    if wait and j.get('state') not in _ENDED:
        seen = seen if isinstance(seen, dict) else {}
        st0 = seen.get('state', j.get('state'))
        stage0 = seen.get('stage', j.get('stage'))
        try:
            p0 = float(seen.get('progress', j.get('progress') or 0))
        except (TypeError, ValueError):
            p0 = float(j.get('progress') or 0)
        try:
            step = max(0.01, min(0.5, float(step or 0.03)))
        except (TypeError, ValueError):
            step = 0.03
        t_end = time.time() + wait
        while time.time() < t_end:
            if j.get('state') in _ENDED or j.get('state') != st0 or j.get('stage') != stage0 or (j.get('progress') or 0) - p0 >= step:
                break
            time.sleep(0.2)
    out = {k: j.get(k) for k in PUBLIC}
    out['job'] = jid
    if wait:
        out['waits'] = True
    if j.get('state') == 'queued':
        q = [x for x in CTX.jobs.values() if x.get('state') == 'queued' and x.get('created', 0) < j.get('created', 0)]
        out['ahead'] = len(q)
    if j.get('state') == 'done':
        try:
            with open(os.path.join(j['dir'], 'result.json'), 'r', encoding='utf-8') as f:
                out['result'] = json.load(f)
        except (OSError, ValueError):
            out['state'] = 'error'
            out['error'] = 'The finished files are missing. Import the video again.'
    return out


def job_cancel(jid):
    j = _load_job(jid)
    if j.get('state') in _ENDED:
        return {'job': jid, 'state': j.get('state')}
    j['cancel'] = True
    if j.get('state') == 'queued':
        _set(j, 'cancelled', 'Stopped')
    p = j.get('proc')
    if p:
        try:
            p.kill()
        except Exception:
            pass
    return {'job': jid, 'state': 'cancelling'}


def job_read(jid, part, offset, length):
    j = _load_job(jid)
    if j.get('state') != 'done':
        raise ToolError('That job isn\'t finished yet.', 'not_ready')
    if part not in ('media', 'thumb'):
        raise ToolError('part is media or thumb.', 'bad_args')
    with open(os.path.join(j['dir'], 'result.json'), 'r', encoding='utf-8') as f:
        res = json.load(f)
    meta = res.get('media') if part == 'media' else res.get('thumbFile')
    if not meta or not meta.get('file'):
        raise ToolError('That job has no %s file.' % part, 'no_file')
    path = os.path.join(j['dir'], os.path.basename(meta['file']))
    if not os.path.isfile(path):
        raise ToolError('That job isn\'t on the import server any more (downloads are kept for a day). Import the video again.', 'no_job')
    size = os.path.getsize(path)
    try:
        offset = max(0, int(offset or 0))
        length = int(length or MAX_CHUNK)
    except (TypeError, ValueError):
        raise ToolError('offset and length must be numbers.', 'bad_args')
    length = max(1, min(length, MAX_CHUNK))
    with open(path, 'rb') as f:
        f.seek(offset)
        data = f.read(length)
    return {'job': jid, 'part': part, 'offset': offset, 'length': len(data), 'size': size,
            'eof': offset + len(data) >= size, 'mime': meta.get('mime'), 'data': base64.b64encode(data).decode('ascii')}


def _worker_loop(q):
    while True:
        jid = q.get()
        try:
            run_job(jid)
        except Exception:
            log('job crashed', jid, traceback.format_exc())


def run_job(jid):
    job = CTX.jobs.get(jid)
    if not job or job.get('cancel') or job.get('state') == 'cancelled':
        if job:
            _set(job, 'cancelled', 'Stopped')
        return
    t0 = time.time()
    try:
        media, info = download(job)
        if job['cancel']:
            raise Cancelled()
        lines, source, warn = [], 'none', None
        if job['transcript'] == 'captions':
            _set(job, 'captions', 'Reading the captions', 0.97)
            lines = read_captions(job)
            source = 'captions'
            if not lines:
                warn = 'The captions had no Chinese lines.'
                source = 'none'
        elif job['transcript'] == 'generate':
            lines = transcribe(job, media['path'])
            source = 'generated'
            if not lines:
                warn = 'No speech was heard in it.'
                source = 'none'
        thumb = _save_thumb(job, info)
        result = {
            'title': job.get('title') or '', 'channel': info.get('channel') or info.get('uploader') or '',
            'id': info.get('id'), 'url': job['url'], 'site': job.get('site') or site_of(job['url']),
            'duration': info.get('duration') or job.get('duration') or 0,
            'kind': job['kind'], 'media': {'file': os.path.basename(media['path']), 'size': media['size'], 'mime': media['mime'],
                                           'name': media['name'], 'width': info.get('width'), 'height': info.get('height')},
            'thumbFile': thumb, 'source': source, 'lang': job.get('captions') if source == 'captions' else ('zh' if source == 'generated' else None),
            'lines': lines, 'warning': warn, 'seconds': round(time.time() - t0, 1), 'description': _desc(info),
        }
        if job.get('songs'):
            result['songs'] = job['songs']
        _save(job, result)
        _set(job, 'done', 'Ready', 1)
        log('job done', jid, job['kind'], media['size'], 'bytes', len(lines), 'lines', source, round(time.time() - t0, 1), 's')
    except Cancelled:
        _set(job, 'cancelled', 'Stopped')
        _rm_media(job)
    except ToolError as e:
        job['error'] = str(e)
        _set(job, 'error', 'Failed')
        log('job failed', jid, str(e)[:300])
    except Exception as e:
        log('job failed', jid, traceback.format_exc())
        job['error'] = friendly(e, job.get('site') or 'YouTube')
        _set(job, 'error', 'Failed')
    finally:
        job.pop('proc', None)


def _rm_media(job):
    for p in glob.glob(os.path.join(job['dir'], 'media*')):
        try:
            os.remove(p)
        except OSError:
            pass


# A refused download (HTTP 403) often works a moment later, or as another of YouTube's apps: tried again the same way
# with fresh links, then as YouTube's TV app and embedded player, then as its Vision Pro app with streamed formats.
TRIES = (
    {'why': None},
    {'why': 'again', 'wait': 2},
    {'why': 'another way', 'wait': 1, 'clients': ['tv', 'web_embedded']},
    {'why': 'another way', 'wait': 1, 'clients': ['visionos'], 'hls': True},
)
TRIES_BILI = (
    {'why': None},
    {'why': 'again', 'wait': 2},
)
_RETRY = ('http error 403', 'forbidden', 'unable to download video data', 'requested format is not available',
          'fragment', 'giving up after', 'did not get any data blocks', 'http error 5', 'unable to download',
          'got error', 'downloaded file is empty', 'no video formats found')


def _sort_for(job, hls=False):
    """Which file: an H.264 picture with AAC sound first (every phone plays them), at the highest height up to the one
    picked, the smallest file there. Speech needs no more than about 100 kbps of sound; music gets the best."""
    sound = 'abr' if job.get('music') else 'abr:100'
    out = []
    if hls:
        out.append('proto:m3u8_native')
    if job['kind'] == 'video':
        out += ['vcodec:h264', 'res:%d' % job['quality'], 'acodec:aac', sound, '+size']
    else:
        out += ['acodec:aac', sound, '+size']
    return out


def download(job):
    tries = TRIES_BILI if job.get('site') == 'Bilibili' else TRIES
    site = job.get('site') or 'YouTube'
    last_err = None
    for k in range(len(tries)):
        t = tries[k]
        if job['cancel']:
            raise Cancelled()
        if k:
            _set(job, 'downloading', '%s refused the download: trying %s (%d of %d)' % (site, t['why'], k + 1, len(tries)), 0.02)
            time.sleep(t.get('wait', 0))
        job['tries'] = k + 1
        try:
            return _download_once(job, t)
        except Cancelled:
            raise
        except ToolError as e:
            if e.code != 'yt_retry':
                raise
            last_err = e
            _rm_media(job)
            log('download try', k + 1, 'of', len(tries), 'failed for', job['id'], str(e)[:200])
    threading.Thread(target=lambda: update_ytdlp(min_gap=3600), daemon=True).start()
    raw = getattr(last_err, 'raw', '') if last_err else 'download failed'
    if '403' in raw or 'forbidden' in raw.lower():
        raise ToolError('%s refused the download, %d ways. The import server is getting a newer downloader now: '
                        'try again in a few minutes.' % (site, len(tries)), 'yt')
    raise ToolError(friendly(raw, site), 'yt')


def _download_once(job, attempt):
    d = job['dir']
    os.makedirs(d, exist_ok=True)
    site = job.get('site') or 'YouTube'
    # the sound without YouTube's "DRC" (loudness-squashed) copies: the speech model hears those worse
    if job['kind'] == 'video':
        fmt = 'bv*+ba[format_id!*=drc]/bv*+ba/b'
    else:
        fmt = 'ba[format_id!*=drc]/ba/b'
    sort = _sort_for(job, attempt.get('hls'))
    sizes = {}

    def on_line(obj):
        if 'info' in obj:
            info = obj['info'] or {}
            if info.get('live_status') in ('is_live', 'is_upcoming'):
                return
            dur = info.get('duration') or 0
            if dur > MAX_SECONDS:
                job['too_long'] = dur
            job['title'] = info.get('title') or ''
            job['duration'] = dur
            req = info.get('requested_formats') or [info]
            job['expected'] = [_fmt_size(f, dur) for f in req]
            job['formats'] = [f.get('format_id') for f in req]
            _set(job, 'downloading', 'Downloading the %s' % ('video' if job['kind'] == 'video' else 'sound'), 0.02)
        elif 'progress' in obj:
            pr = obj['progress']
            fn = pr.get('file') or ''
            if pr.get('status') == 'finished':
                got = pr.get('total') or pr.get('got') or 0
                sizes[fn] = (got, got)
            else:
                sizes[fn] = (pr.get('got') or 0, pr.get('total') or 0)
            exp = job.get('expected') or []
            got = sum(a for a, b in sizes.values())
            tot = max(sum(exp), sum(b for a, b in sizes.values()))
            frac = got / tot if tot else 0
            span = 0.9 if job['transcript'] != 'generate' else 0.45
            _set(job, None, 'Downloading the %s' % ('video' if job['kind'] == 'video' else 'sound'), 0.02 + span * min(1.0, frac))
        elif 'pp' in obj and obj['pp'] in ('Merger', 'FFmpegMerger'):
            _set(job, 'merging', 'Joining the video and its sound')

    opts = {
        'format': fmt, 'format_sort': sort, 'outtmpl': {'default': os.path.join(d, 'media.%(ext)s'), 'subtitle': os.path.join(d, 'subs.%(ext)s')},
        'overwrites': True, 'continuedl': False, 'writethumbnail': False, 'writeinfojson': False, 'keepvideo': False,
    }
    if attempt.get('clients'):
        opts['extractor_args'] = {'youtube': {'player_client': list(attempt['clients'])}}
    if job['kind'] == 'video':
        opts['merge_output_format'] = 'mp4'
    if job['transcript'] == 'captions' and job.get('captions'):
        lang = job['captions']
        auto = site == 'YouTube' and lang.endswith('-orig')
        opts.update({'subtitleslangs': [lang], 'subtitlesformat': 'json3/srv3/vtt/srt/best',
                     'writesubtitles': not auto, 'writeautomaticsub': auto})
    _set(job, 'downloading', 'Looking the video up', 0.01)
    try:
        res = ytw({'op': 'download', 'url': job['url'], 'opts': base_opts(opts), 'thumb': True, 'max_seconds': MAX_SECONDS}, job, on_line)
    except Cancelled:
        raise
    except YtwError as e:
        if job['cancel']:
            raise Cancelled()
        msg = str(e)
        if job.get('too_long'):
            raise ToolError('It\'s %s long: the import server takes videos up to 3 hours.' % _hms(job['too_long']), 'too_long')
        log('download failed', job['id'], msg[:400], (job.get('warnings') or [])[-2:])
        if any(k in msg.lower() for k in _RETRY):
            err = ToolError(friendly(msg, site), 'yt_retry')
            err.raw = msg
            raise err
        raise ToolError(friendly(msg, site), 'yt')
    if res.get('live'):
        raise ToolError('This is a live stream. Import it after it ends.', 'live')
    if res.get('too_long') or job.get('too_long'):
        raise ToolError('It\'s %s long: the import server takes videos up to 3 hours.' % _hms(res.get('too_long') or job.get('too_long')), 'too_long')
    info = res.get('info') or {}
    job['thumb_uri'] = res.get('thumb')
    files = [p for p in glob.glob(os.path.join(d, 'media.*')) if not p.endswith(('.part', '.ytdl', '.temp'))
             and not re.search(r'\.f\d+\.', os.path.basename(p)) and not re.search(r'\.(json3|srv3|vtt|srt)$', p)]
    if not files:
        err = ToolError('The download finished but no file was saved. Try again.', 'yt_retry')
        err.raw = 'downloaded file is empty'
        raise err
    path = max(files, key=os.path.getsize)
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if job['kind'] == 'video':
        mime = {'mp4': 'video/mp4', 'webm': 'video/webm', 'mkv': 'video/x-matroska', 'mov': 'video/quicktime'}.get(ext, 'video/mp4')
    else:
        mime = {'m4a': 'audio/mp4', 'mp4': 'audio/mp4', 'webm': 'audio/webm', 'opus': 'audio/ogg', 'mp3': 'audio/mpeg', 'ogg': 'audio/ogg'}.get(ext, 'audio/mp4')
    safe = re.sub(r'[\\/:*?"<>|\r\n\t]+', ' ', job.get('title') or 'video').strip()[:80] or 'video'
    log('downloaded', job['id'], 'try', job.get('tries'), 'formats', '+'.join(str(x) for x in job.get('formats') or []), os.path.getsize(path), 'bytes', ext)
    return {'path': path, 'size': os.path.getsize(path), 'mime': mime, 'name': safe + '.' + ext}, info


def _hms(sec):
    sec = int(sec or 0)
    h, m = sec // 3600, (sec % 3600) // 60
    return ('%d h %d min' % (h, m)) if h else ('%d min' % m)


def _save_thumb(job, info):
    try:
        uri = _small_thumb(job.pop('thumb_uri', None))
        if not uri:
            return None
        head, b64 = uri.split(',', 1)
        p = os.path.join(job['dir'], 'thumb.jpg')
        with open(p, 'wb') as f:
            f.write(base64.b64decode(b64))
        return {'file': os.path.basename(p), 'mime': 'image/jpeg', 'size': os.path.getsize(p)}
    except Exception:
        return None


def cleanup_cache():
    try:
        now = time.time()
        dirs = []
        for d in glob.glob(os.path.join(JOBS_DIR, '*')):
            if not os.path.isdir(d):
                continue
            jid = os.path.basename(d)
            if jid in CTX.jobs and CTX.jobs[jid].get('state') not in _ENDED:
                continue
            mt = os.path.getmtime(d)
            size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(d, '*')) if os.path.isfile(p))
            dirs.append((mt, d, size))
        total = sum(s for _, _, s in dirs)
        for mt, d, size in sorted(dirs):
            if now - mt > KEEP_HOURS * 3600 or total > CACHE_CAP:
                shutil.rmtree(d, ignore_errors=True)
                total -= size
                CTX.jobs.pop(os.path.basename(d), None)
    except Exception:
        log('cleanup failed', traceback.format_exc())


# ---------------------------------------------------------------- captions

_HAN = re.compile(r'[\u3400-\u9fff\uf900-\ufaff\U00020000-\U0002ffff]')
_TAGS = re.compile(r'<[^>]+>')
_SOUND = re.compile(r'^[\s\[\(（【]*(音乐|音樂|掌声|掌聲|笑声|笑聲|笑|鼓掌|music|applause|laughter|laughs|笑声\s*)[\]\)）】\s]*$', re.I)


def _opencc():
    def make():
        try:
            from opencc import OpenCC
            return OpenCC('t2s')
        except Exception:
            return None
    return _cached('opencc', make)


def to_simplified(text):
    cc = _opencc()
    if not cc or not _HAN.search(text or ''):
        return text
    try:
        return cc.convert(text)
    except Exception:
        return text


# pinyin lines (learners' captions often have one under the Chinese) are not English: they're dropped, since shuo shows
# its own pinyin. Tone marks with a macron or caron never occur in English; toneless pinyin is told by its syllable shape.
_PY_MARKS = re.compile('[\u0101\u01ce\u0113\u011b\u012b\u01d0\u014d\u01d2\u016b\u01d4\u01d6\u01d8\u01da\u01dc]')
_PY_ANY = re.compile('[\u0101\u00e1\u01ce\u00e0\u0113\u00e9\u011b\u00e8\u012b\u00ed\u01d0\u00ec\u014d\u00f3\u01d2\u00f2'
                     '\u016b\u00fa\u01d4\u00f9\u01d6\u01d8\u01da\u01dc\u00fc]')
_PY_SYL = re.compile(r"^(?:(?:[bpmfdtnlgkhjqxrzcsyw]|zh|ch|sh)?(?:i|u|v|\u00fc)?"
                     r"(?:a|o|e|ai|ei|ao|ou|an|en|ang|eng|ong|er|i|u|v|\u00fc|in|ing|un|ui|iu|ie|ue|ia|ian|iang|iao|iong|ua|uai|uan|uang|uo)r?)+$")


def _plain(w):
    import unicodedata
    w = w.replace('\u00fc', 'v').replace('\u01d6', 'v').replace('\u01d8', 'v').replace('\u01da', 'v').replace('\u01dc', 'v')
    return ''.join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn').lower()


def _is_pinyin(line):
    words = re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*", line)
    if not words:
        return False
    marked = sum(1 for w in words if _PY_ANY.search(w))
    shaped = sum(1 for w in words if all(_PY_SYL.match(p) for p in _plain(w).split("'")))
    if _PY_MARKS.search(line) and marked * 3 >= len(words) and shaped * 10 >= len(words) * 7:
        return True
    return shaped == len(words) and marked * 2 >= len(words)


def _split_zh_en(text):
    zh, en = [], []
    for ln in re.split(r'\n+', text):
        ln = re.sub(r'\s+', ' ', ln).strip()
        if not ln or _SOUND.match(ln):
            continue
        if _HAN.search(ln):
            zh.append(ln)
        elif not _is_pinyin(ln):
            en.append(ln)
    return ' '.join(zh).strip(), ' '.join(en).strip()


def parse_json3(data, auto=False):
    events = data.get('events') or []
    if auto:
        # speech recognition captions: words with their own times, regrouped into lines here
        toks = []
        for e in events:
            segs = e.get('segs')
            if not segs or e.get('aAppend'):
                continue
            t0 = (e.get('tStartMs') or 0) / 1000.0
            for s in segs:
                w = (s.get('utf8') or '').replace('\n', ' ')
                if not w.strip():
                    continue
                toks.append((w.strip() if _HAN.search(w) else w, t0 + (s.get('tOffsetMs') or 0) / 1000.0))
        if not toks:
            return []
        end = max((((e.get('tStartMs') or 0) + (e.get('dDurationMs') or 0)) / 1000.0) for e in events)
        return lines_from_tokens([{'start': toks[0][1], 'end': end, 'toks': toks}], punctuated=False)
    out = []
    for e in events:
        segs = e.get('segs')
        if not segs:
            continue
        t0 = (e.get('tStartMs') or 0) / 1000.0
        dur = (e.get('dDurationMs') or 0) / 1000.0
        txt = _TAGS.sub('', ''.join(s.get('utf8') or '' for s in segs))
        zh, en = _split_zh_en(txt)
        if not zh and not en:
            continue
        out.append({'s': round(t0, 3), 'e': round(t0 + max(dur, 0.2), 3), 'text': zh, 'en': en})
    return _tidy(out)


def _vtt_time(s):
    m = re.match(r'(?:(\d+):)?(\d+):(\d+)[.,](\d+)', s.strip())
    if not m:
        return None
    h, mi, se, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, '0')[:3]) / 1000.0


def parse_vtt(text):
    out = []
    for block in re.split(r'\n\s*\n', text.replace('\r\n', '\n').replace('\r', '\n')):
        lines = block.split('\n')
        at = next((i for i, l in enumerate(lines) if '-->' in l), -1)
        if at < 0:
            continue
        a, b = lines[at].split('-->', 1)
        s, e = _vtt_time(a), _vtt_time(b)
        if s is None or e is None:
            continue
        txt = _TAGS.sub('', '\n'.join(lines[at + 1:]))
        zh, en = _split_zh_en(txt)
        if zh or en:
            out.append({'s': round(s, 3), 'e': round(max(e, s + 0.2), 3), 'text': zh, 'en': en})
    # rolling captions repeat the line before: keep each line once
    ded = []
    for c in out:
        if ded and c['text'] == ded[-1]['text'] and c['s'] <= ded[-1]['e'] + 0.05:
            ded[-1]['e'] = max(ded[-1]['e'], c['e'])
            continue
        ded.append(c)
    return _tidy(ded)


def _tidy(cues):
    cues = sorted((c for c in cues if c.get('text') or c.get('en')), key=lambda c: c['s'])
    for i in range(len(cues) - 1):
        if cues[i]['e'] > cues[i + 1]['s'] and cues[i + 1]['s'] > cues[i]['s']:
            cues[i]['e'] = cues[i + 1]['s']
    return cues


def read_captions(job):
    d = job['dir']
    lang = job.get('captions') or ''
    files = sorted(glob.glob(os.path.join(d, 'subs*')))
    if not files:
        raise ToolError('%s didn\'t send the captions. Try again, or choose to make a transcript instead.' % (job.get('site') or 'YouTube'), 'no_captions')
    pref = {'.json3': 0, '.srv3': 1, '.vtt': 2}
    files.sort(key=lambda p: pref.get(os.path.splitext(p)[1], 9))
    p = files[0]
    with open(p, 'rb') as f:
        raw = f.read().decode('utf-8', 'replace')
    ext = os.path.splitext(p)[1]
    if ext == '.json3':
        cues = parse_json3(json.loads(raw), auto=lang.endswith('-orig'))
    elif ext == '.srv3':
        cues = parse_srv3(raw)
    else:
        cues = parse_vtt(raw)
    if job.get('simplified'):
        for c in cues:
            t = c.get('text') or ''
            s2 = to_simplified(t)
            if s2 != t:
                c['text'] = s2
                if len(s2) != len(t):
                    c.pop('ts', None)
    return [c for c in cues if _HAN.search(c.get('text') or '')]


def parse_srv3(raw):
    out = []
    for m in re.finditer(r'<p\s+([^>]*)>(.*?)</p>', raw, re.S):
        attrs, body = m.group(1), m.group(2)
        t = re.search(r'\bt="(\d+)"', attrs)
        d = re.search(r'\bd="(\d+)"', attrs)
        if not t:
            continue
        s = int(t.group(1)) / 1000.0
        e = s + (int(d.group(1)) / 1000.0 if d else 2.0)
        txt = _TAGS.sub('', body).replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&#39;', "'").replace('&quot;', '"')
        zh, en = _split_zh_en(txt)
        if zh or en:
            out.append({'s': round(s, 3), 'e': round(e, 3), 'text': zh, 'en': en})
    return _tidy(out)


# ---------------------------------------------------------------- lines from timed words

_END = set('。！？!?…')
_SOFT = set('，,、；;：:')
SOFT_AT = 16     # at a comma, a line this long (in characters) ends
HARD_AT = 34     # with no punctuation, a line ends at this length
GAP_CUT = 2.5    # v22: this long between two words (music, laughter, a long pause) ends a line, even mid-sentence
TAIL = 0.4       # v22: a line ends this long after its last word starts, not when its punctuation comes
_SPECIAL = re.compile(r'^<\|.*\|>$')
_NOT_SAID = re.compile(r'^[\s\W_]*$')   # punctuation (and spaces) only: nothing said


def _hlen(s):
    return len(_HAN.findall(s))


def _best_cut(cur):
    """Where to cut a line that got too long: before the word after the longest pause (a comma before it counts as
    half a second more), keeping at least 8 characters before the cut. All of it when there's no such place."""
    best, best_k, prev_t, soft = -1.0, len(cur), None, False
    for k, (tx, t) in enumerate(cur):
        if _NOT_SAID.match(tx):
            soft = soft or any(c in _SOFT or c in _END for c in tx)
            continue
        if prev_t is not None and k > 0 and _hlen(''.join(x for x, _ in cur[:k])) >= 8:
            score = (t - prev_t) + (0.5 if soft else 0)
            if score >= best:
                best, best_k = score, k
        prev_t, soft = t, False
    return best_k


def lines_from_tokens(segs, punctuated=True):
    """segs: [{start, end, toks: [(text, t)]}] with t in seconds. One line per sentence; a long sentence is cut at a
    comma, a very long one anywhere, and any line at a pause (GAP_CUT). Each line keeps its characters' times (ts) for
    the word underline. v22: a line lasts while it is being said: from just before its first word to just after its
    last one (TAIL). The speech model puts punctuation at the end of a piece of sound, often seconds later (music,
    laughter), so punctuation never sets a time: the line stops at its last word, and nothing is shown until the next."""
    lines = []

    def flush(cur, seg_end, next_t):
        if not cur:
            return
        text = ''.join(t for t, _ in cur).strip()
        said = [(tx, t) for tx, t in cur if not _NOT_SAID.match(tx)]
        if not text or not said:
            return
        ts, last_t = [], said[0][1]
        for tx, t in cur:
            if _NOT_SAID.match(tx):
                t = last_t          # a comma or a full stop: at its word, not seconds later
            else:
                last_t = t
            ts.extend([round(t, 2)] * len(tx))
        lead = len(''.join(t for t, _ in cur)) - len(''.join(t for t, _ in cur).lstrip())
        ts = ts[lead:lead + len(text)]
        s, last = said[0][1], said[-1][1]
        # the last word is heard a moment after it starts: longer for a word of several characters (YouTube's own
        # captions) or an English word
        w = said[-1][0].strip()
        e = last + max(TAIL, 0.25 * _hlen(w) + 0.1, 0.55 if not _HAN.search(w) and len(w) > 2 else 0)
        if next_t is not None:
            e = min(e, next_t)
        e = min(e, seg_end)
        lines.append({'s': round(max(0.0, s - 0.12), 2), 'e': round(max(e, s + 0.35), 2), 'text': text, 'ts': ts})

    for seg in segs:
        toks = [(t, tt) for t, tt in seg['toks'] if t and not _SPECIAL.match(t)]
        cur = []
        prev_word = None
        for i, (tx, t) in enumerate(toks):
            said = not _NOT_SAID.match(tx)
            # a pause before this word: the line before it ends at its own last word
            if said and cur and prev_word is not None and t - prev_word > GAP_CUT:
                flush(cur, seg['end'], t - 0.05)
                cur = []
            if not cur and tx.strip() and not said:
                if lines and lines[-1]['text'] and lines[-1]['text'][-1] not in _END:
                    lines[-1]['text'] += tx.strip()
                    lines[-1]['ts'].extend([lines[-1]['ts'][-1] if lines[-1]['ts'] else round(t, 2)] * len(tx.strip()))
                continue
            cur.append((tx, t))
            if said:
                prev_word = t
            n = _hlen(''.join(x for x, _ in cur))
            last = i == len(toks) - 1
            if last:
                continue
            cut = tx in _END or (tx in _SOFT and n >= SOFT_AT)
            if not punctuated:
                cut = cut or (toks[i + 1][1] - t > 0.9 and n >= 4)
            if cut:
                nxt = next((tt for x, tt in toks[i + 1:] if not _NOT_SAID.match(x)), None)
                flush(cur, seg['end'], (nxt - 0.05) if nxt is not None else None)
                cur = []
            elif n >= (HARD_AT if punctuated else 22):
                # too long with no full stop: cut where the speaker paused longest (after a comma if there is one)
                k = _best_cut(cur)
                head, cur = cur[:k], cur[k:]
                nxt = next((tt for x, tt in cur if not _NOT_SAID.match(x)), None)
                if nxt is None:
                    nxt = next((tt for x, tt in toks[i + 1:] if not _NOT_SAID.match(x)), None)
                flush(head, seg['end'], (nxt - 0.05) if nxt is not None else None)
        flush(cur, seg['end'], None)
    # a tiny line just after another (a word or two after a short pause, no sentence end before it) joins it
    out = []
    for ln in lines:
        if out and _hlen(ln['text']) <= 2 and out[-1]['text'] and out[-1]['text'][-1] not in _END and ln['s'] - out[-1]['e'] < 0.6 \
                and _hlen(out[-1]['text']) < HARD_AT:
            out[-1]['text'] += ln['text']
            out[-1]['ts'] += ln['ts']
            out[-1]['e'] = ln['e']
            continue
        out.append(ln)
    for i in range(len(out) - 1):
        if out[i]['e'] > out[i + 1]['s']:
            out[i]['e'] = max(out[i]['s'] + 0.2, out[i + 1]['s'])
    # one lone character with two seconds of nothing around it is mostly the speech model hearing words in music or noise
    keep = []
    for i, ln in enumerate(out):
        before = ln['s'] - out[i - 1]['e'] if i else 99
        after = out[i + 1]['s'] - ln['e'] if i + 1 < len(out) else 99
        if _hlen(ln['text']) <= 1 and not re.search(r'[A-Za-z]{2}', ln['text']) and before > 2 and after > 2:
            continue
        keep.append(ln)
    return keep


# v22: singing (a TV episode's theme song, a song playing in a vlog) heard as lines. Sung syllables are held: the
# characters of a sung line come about 0.3 s apart or more and under 3 a second, while talk runs at 4 to 6 a second
# with short gaps (a slow, sad line can look sung, but not five of them in a row).
SONG_GAP = 0.27      # the middle gap between two characters of a sung line, at least (seconds)
SONG_CPS = 2.8       # characters a second in a sung line, at most
SONG_JOIN = 30.0     # sung lines this close (seconds) belong to one song
SONG_MIN = 5         # a song has at least this many sung lines ...
SONG_SHARE = 0.6     # ... which are at least this share of the lines in it ...
SONG_SPAN = 20.0     # ... over at least this long (seconds)


def _sung(ln):
    ts = ln.get('ts') or []
    gaps = sorted(b - a for a, b in zip(ts, ts[1:]) if b - a > 0.001)
    han = _hlen(ln['text'])
    if len(gaps) < 2 or han < 3:
        return False
    mid = gaps[len(gaps) // 2] if len(gaps) % 2 else (gaps[len(gaps) // 2 - 1] + gaps[len(gaps) // 2]) / 2
    dur = max(0.1, ln['e'] - ln['s'])
    return mid >= SONG_GAP and han / dur <= SONG_CPS


def song_parts(lines):
    """The stretches of a transcript that are a song: [(first line, last line)] by index."""
    sung = [i for i, ln in enumerate(lines) if _sung(ln)]
    parts, run = [], []
    for i in sung:
        if run and lines[i]['s'] - lines[run[-1]]['e'] > SONG_JOIN:
            parts.append(run)
            run = []
        run.append(i)
    if run:
        parts.append(run)
    out = []
    for run in parts:
        a, b = run[0], run[-1]
        if len(run) >= SONG_MIN and len(run) >= SONG_SHARE * (b - a + 1) and lines[b]['e'] - lines[a]['s'] >= SONG_SPAN:
            out.append((a, b))
    return out


def drop_songs(lines):
    """A transcript without its songs, and the songs' times [[s, e]]. When songs are most of it (a music video imported
    without "This is music"), all of it stays: the words sung are what there is."""
    parts = song_parts(lines)
    if not parts:
        return lines, []
    gone = sum(b - a + 1 for a, b in parts)
    if gone * 2 > len(lines):
        return lines, []
    skip = set()
    for a, b in parts:
        skip.update(range(a, b + 1))
    return [ln for i, ln in enumerate(lines) if i not in skip], [[lines[a]['s'], lines[b]['e']] for a, b in parts]


# ---------------------------------------------------------------- the transcript (SenseVoice + VAD, on this server)

def model_paths():
    cfg = read_config()
    sv = cfg.get('sense_voice') or os.path.join(MODELS, 'sense-voice')
    model = None
    for name in ('model.int8.onnx', 'model.onnx'):
        p = os.path.join(sv, name)
        if os.path.isfile(p):
            model = p
            break
    tokens = os.path.join(sv, 'tokens.txt')
    vad = cfg.get('vad') or os.path.join(MODELS, 'silero_vad.onnx')
    return {'dir': sv, 'model': model, 'tokens': tokens if os.path.isfile(tokens) else None,
            'vad': vad if os.path.isfile(vad) else None}


def threads():
    n = os.cpu_count() or 4
    return max(2, min(8, n // 2))


def recognizer():
    def make():
        import sherpa_onnx
        mp = model_paths()
        if not mp['model'] or not mp['tokens']:
            raise ToolError('The speech model is missing on the import server, so a transcript can\'t be written.', 'no_model')
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=mp['model'], tokens=mp['tokens'], num_threads=threads(), use_itn=True, language='auto', debug=False)
    with CTX.lock:
        if 'asr' not in CTX.store:
            CTX.store['asr'] = make()
        return CTX.store['asr']


def recognizer_zh():
    def make():
        import sherpa_onnx
        mp = model_paths()
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=mp['model'], tokens=mp['tokens'], num_threads=threads(), use_itn=True, language='zh', debug=False)
    with CTX.lock:
        if 'asr_zh' not in CTX.store:
            CTX.store['asr_zh'] = make()
        return CTX.store['asr_zh']


def decode_audio(path, job=None):
    """The sound as 16 kHz mono int16 (numpy), through ffmpeg."""
    import numpy as np
    ff = ffmpeg_path()
    if not ff:
        raise ToolError('ffmpeg is missing on the import server, so the sound can\'t be read.', 'no_ffmpeg')
    cmd = [ff, '-nostdin', '-hide_banner', '-v', 'error', '-i', path, '-vn', '-sn', '-dn', '-ac', '1', '-ar', '16000', '-f', 's16le', '-']
    p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
    if job is not None:
        job['proc'] = p
    # v22: ffmpeg's messages are read on their own (a full message pipe would stop ffmpeg while this waits for the
    # sound), and ffmpeg is stopped if no sound comes for two minutes
    errs = []
    t_err = threading.Thread(target=lambda: errs.append(p.stderr.read()), daemon=True)
    t_err.start()
    state = {'last': time.time(), 'dead': False}

    def watchdog():
        while p.poll() is None:
            if time.time() - state['last'] > 120:
                state['dead'] = True
                try:
                    p.kill()
                except Exception:
                    pass
                return
            time.sleep(2)
    threading.Thread(target=watchdog, daemon=True).start()
    chunks = []
    while True:
        b = p.stdout.read(1 << 20)
        if not b:
            break
        state['last'] = time.time()
        chunks.append(b)
        if job is not None and job.get('cancel'):
            p.kill()
            raise Cancelled()
    p.wait()
    t_err.join(5)
    err = (errs[0] if errs else b'').decode('utf-8', 'replace')
    if job is not None:
        job.pop('proc', None)
    if state['dead']:
        log('ffmpeg stopped answering while reading the sound of', path, err[-300:])
        raise ToolError('Reading the sound took too long (ffmpeg stopped answering). Try again.', 'no_audio')
    raw = b''.join(chunks)
    if len(raw) < 3200:
        raise ToolError('The sound couldn\'t be read from the file. ' + err[:200], 'no_audio')
    if len(raw) % 2:
        raw = raw[:-1]
    return np.frombuffer(raw, dtype=np.int16)


def speech_segments(pcm, job=None, on_progress=None):
    """Where people speak: [(start_sample, end_sample)], from the Silero VAD."""
    import numpy as np
    import sherpa_onnx
    mp = model_paths()
    if not mp['vad']:
        raise ToolError('The speech detector is missing on the import server, so a transcript can\'t be written.', 'no_model')
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = mp['vad']
    cfg.silero_vad.threshold = 0.4
    cfg.silero_vad.min_silence_duration = 0.35
    cfg.silero_vad.min_speech_duration = 0.2
    try:
        cfg.silero_vad.max_speech_duration = 18
    except Exception:
        pass
    cfg.sample_rate = 16000
    cfg.num_threads = 1
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
    win = cfg.silero_vad.window_size or 512
    f32 = pcm.astype(np.float32) / 32768.0
    segs = []
    n = len(f32)
    step_report = 16000 * 30

    def take():
        while not vad.empty():
            seg = vad.front
            segs.append((int(seg.start), int(seg.start) + len(seg.samples)))
            vad.pop()
    for i in range(0, n - win + 1, win):
        vad.accept_waveform(f32[i:i + win])
        take()
        if i % step_report < win:
            if job is not None and job.get('cancel'):
                raise Cancelled()
            if on_progress:
                on_progress(i / n)
    vad.flush()
    take()
    return segs


def _merge_segments(segs, total, sr=16000):
    """Pad each piece of speech a little, join pieces with tiny gaps, and cut very long ones at their quietest point."""
    pad_l, pad_r = int(0.15 * sr), int(0.25 * sr)
    out = []
    for a, b in segs:
        a, b = max(0, a - pad_l), min(total, b + pad_r)
        if out and a - out[-1][1] < int(0.25 * sr) and (b - out[-1][0]) < 25 * sr:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _decode_batch(rec, pcm, batch):
    import numpy as np
    streams = []
    for a, b in batch:
        s = rec.create_stream()
        s.accept_waveform(16000, (pcm[a:b].astype(np.float32) / 32768.0))
        streams.append(s)
    rec.decode_streams(streams)
    return [s.result for s in streams]


def transcribe(job, path):
    """A timed transcript of the file's speech: [{s, e, text, ts}]."""
    t_all = time.time()
    base = 0.47
    _set(job, 'transcribing', 'Reading the sound', base)
    pcm = decode_audio(path, job)
    total = len(pcm)
    dur = total / 16000.0
    _set(job, None, 'Finding where people speak', base + 0.03)
    segs = speech_segments(pcm, job, lambda f: _set(job, None, 'Finding where people speak', base + 0.03 + 0.07 * f))
    segs = _merge_segments(segs, total)
    if not segs:
        return []
    rec = recognizer()
    out_segs = []
    done_samples = 0
    speech_samples = sum(b - a for a, b in segs) or 1
    i = 0
    B = 6
    while i < len(segs):
        if job.get('cancel'):
            raise Cancelled()
        batch = segs[i:i + B]
        results = _decode_batch(rec, pcm, batch)
        for (a, b), r in zip(batch, results):
            lang = (getattr(r, 'lang', '') or '').strip('<|>')
            toks = list(getattr(r, 'tokens', []) or [])
            tss = list(getattr(r, 'timestamps', []) or [])
            if lang in ('ja', 'ko') and toks:
                # a short Chinese phrase taken for Japanese or Korean: hear it again as Chinese
                r2 = _decode_batch(recognizer_zh(), pcm, [(a, b)])[0]
                toks, tss = list(r2.tokens or []), list(r2.timestamps or [])
            if len(tss) != len(toks):
                tss = [0.0] * len(toks) if not tss else (tss + [tss[-1]] * len(toks))[:len(toks)]
            items = []
            for tx, t in zip(toks, tss):
                tx = tx.replace('\u2581', ' ')
                if not tx or _SPECIAL.match(tx.strip()):
                    continue
                items.append((tx, a / 16000.0 + float(t)))
            if items:
                # English words come as pieces starting with a space; the first one has none
                items[0] = (items[0][0].lstrip(), items[0][1])
                out_segs.append({'start': a / 16000.0, 'end': b / 16000.0, 'toks': items})
            done_samples += b - a
        i += B
        _set(job, None, 'Writing the transcript', base + 0.1 + (0.98 - base - 0.1) * done_samples / speech_samples)
    lines = lines_from_tokens(out_segs, punctuated=True)
    if job.get('simplified'):
        for ln in lines:
            t2 = to_simplified(ln['text'])
            if t2 != ln['text']:
                if len(t2) != len(ln['text']):
                    ln.pop('ts', None)
                ln['text'] = t2
    for ln in lines:
        ln['text'] = re.sub(r'\s+', ' ', ln['text']).strip()
    lines = [ln for ln in lines if ln['text']]
    # v22: songs in a video that isn't music (a theme song) aren't talk: no lines while they play
    if not job.get('music'):
        lines, songs = drop_songs(lines)
        if songs:
            job['songs'] = songs
            log('songs left out of', job.get('id'), songs)
    log('transcribed', job.get('id'), round(dur, 1), 's of sound in', round(time.time() - t_all, 1), 's:', len(lines), 'lines')
    return lines



# ---------------------------------------------------------------- status and checks

def helper_info():
    out = {'ok': True, 'version': VERSION, 'server': 'cloud', 'sites': ['YouTube', 'Bilibili'], 'problems': [], 'logins': _logins()}
    out['ytdlp'] = _installed('yt-dlp')
    if not out['ytdlp']:
        out['ok'] = False
        out['problems'].append('yt-dlp is missing.')
    up = _read_update()
    out['update'] = {'checked': up.get('checked'), 'installed': _installed('yt-dlp'), 'ok': up.get('ok'),
                     'running': bool(CTX.store.get('updating'))}
    out['ffmpeg'] = bool(ffmpeg_path())
    if not out['ffmpeg']:
        out['ok'] = False
        out['problems'].append('ffmpeg is missing.')
    rt = js_runtimes()
    out['js'] = {k: v.get('version') for k, v in rt.items()}
    if not rt:
        out['problems'].append('No JavaScript runtime (deno or node 22+) for YouTube\'s download check.')
    out['pot'] = bool(CTX.store.get('pot'))
    mp = model_paths()
    out['transcripts'] = bool(mp['model'] and mp['tokens'] and mp['vad'])
    try:
        import sherpa_onnx
        out['sherpa_onnx'] = getattr(sherpa_onnx, '__version__', '?')
    except Exception:
        out['transcripts'] = False
    if not out['transcripts']:
        out['problems'].append('Transcripts can\'t be made: the speech model is missing.')
    out['threads'] = threads()
    out['busy'] = [j['id'] for j in CTX.jobs.values() if j.get('state') in _RUNNING]
    out['max_seconds'] = MAX_SECONDS
    out['you'] = {'owner': who() == 'owner', 'daily': who_daily(), 'today': _usage()}
    out['busy'] = [j['id'] for j in CTX.jobs.values() if j.get('state') in _RUNNING and (j.get('user') or 'owner') == who()]
    return out


def selftest(quick=True):
    rep = {'ok': True, 'checks': [], 'version': VERSION, 'python': sys.version.split()[0]}

    def check(name, fn, needed=True):
        t0 = time.time()
        try:
            v = fn()
            rep['checks'].append({'name': name, 'ok': True, 'info': v, 'ms': int((time.time() - t0) * 1000)})
        except Exception as e:
            if needed:
                rep['ok'] = False
            rep['checks'].append({'name': name, 'ok': False, 'needed': needed, 'error': repr(e)[:300]})

    def c_ytdlp():
        v = _installed('yt-dlp')
        if not v:
            raise RuntimeError('yt-dlp is not installed')
        res = ytw({'op': 'version'}, total=60)
        return {'version': v, 'loads': res.get('version'), 'ejs': _installed('yt-dlp-ejs'), 'pot': _installed('bgutil-ytdlp-pot-provider')}
    check('yt-dlp', c_ytdlp)

    def c_ff():
        p = ffmpeg_path()
        if not p:
            raise RuntimeError('no ffmpeg')
        return {'path': p, 'version': _run([p, '-version']).split('\n')[0][:120]}
    check('ffmpeg', c_ff)

    def c_js():
        rt = js_runtimes()
        if not rt:
            raise RuntimeError('no deno or node 22+')
        return rt
    check('javascript runtime', c_js, needed=False)

    def c_cc():
        r = to_simplified('這個視頻說的漢語')
        if r == '這個視頻說的漢語':
            raise RuntimeError('OpenCC is missing: Traditional characters would stay as they are')
        return r
    check('opencc', c_cc)

    def c_links():
        got = [norm_url(u, follow=False) for u in ('dQw4w9WgXcQ', 'https://youtu.be/dQw4w9WgXcQ?t=3', 'BV1GJ411x7h7',
                                                    'https://m.bilibili.com/video/BV1GJ411x7h7?p=2&share=1',
                                                    'https://www.bilibili.com/zh-Hans/video/BV1GJ411x7h7/?share_source=copy_web&vd_source=66ee',
                                                    'ttps://www.bilibili.com/video/BV1GJ411x7h7/',
                                                    '【标题】 https://www.bilibili.com/video/av170001')]
        for bad in ('https://example.com/x.mp4', 'file:///etc/passwd', 'https://www.bilibili.com/bangumi/play/ss1'):
            try:
                norm_url(bad, follow=False)
                raise RuntimeError('took ' + bad)
            except ToolError:
                pass
        return got
    check('links', c_links)

    def c_lines():
        toks = [(c, 0.1 * i) for i, c in enumerate('大家好，我是妮卡。我来自中国，今天我们一起去超市买东西吧！')]
        return [l['text'] for l in lines_from_tokens([{'start': 0, 'end': 4.0, 'toks': toks}])]
    check('lines', c_lines)

    def c_sherpa():
        import sherpa_onnx
        mp = model_paths()
        if not (mp['model'] and mp['tokens'] and mp['vad']):
            raise RuntimeError('the speech model or detector is missing: %r' % mp)
        return {'version': getattr(sherpa_onnx, '__version__', '?'), 'models': mp, 'threads': threads()}
    check('speech model', c_sherpa)

    if not quick:
        def c_asr():
            import numpy as np
            rec = recognizer()
            st = rec.create_stream()
            st.accept_waveform(16000, np.zeros(16000, dtype=np.float32))
            rec.decode_stream(st)
            segs = speech_segments(np.zeros(32000, dtype=np.int16))
            return {'model_runs': True, 'text_of_silence': st.result.text, 'vad_segments_in_silence': len(segs)}
        check('speech model runs', c_asr)
    return rep


# ---------------------------------------------------------------- --ytw: one yt-dlp task in a process of its own
# Reads one request (JSON) from stdin; writes one JSON object per line to stdout:
#   {"info": {...}}  {"progress": {...}}  {"pp": "<name>"}  {"warn": "..."}  {"last": "..."}  then {"result": {...}} or {"error": "..."}

def ytw_main():
    out = sys.stdout.buffer

    def emit(obj):
        out.write(json.dumps(obj, ensure_ascii=True, separators=(',', ':')).encode('ascii') + b'\n')
        out.flush()

    class Log:
        def __init__(self):
            self.last = 0.0

        def debug(self, msg):
            if msg.startswith('[debug] '):
                return
            now = time.time()
            if msg.startswith('[') and now - self.last > 0.5:
                self.last = now
                emit({'last': msg[:300]})

        def info(self, msg):
            self.debug(msg)

        def warning(self, msg):
            emit({'warn': msg[:500]})

        def error(self, msg):
            emit({'warn': 'ERROR ' + msg[:800]})

    FMT_KEYS = ('format_id', 'vcodec', 'acodec', 'height', 'width', 'filesize', 'filesize_approx', 'tbr', 'abr', 'vbr',
                'protocol', 'ext', 'format_note', 'fps', 'language')

    def trim(info):
        if not info:
            return {}
        keep = ('id', 'title', 'channel', 'uploader', 'duration', 'language', 'live_status', 'webpage_url', '_type',
                'width', 'height', 'format_id', 'ext', 'filesize', 'filesize_approx', 'tbr', 'vcodec', 'acodec', 'description')
        o = {k: info.get(k) for k in keep if info.get(k) is not None}
        if isinstance(o.get('description'), str):
            o['description'] = o['description'][:2000]
        o['formats'] = [{k: f.get(k) for k in FMT_KEYS if f.get(k) is not None} for f in (info.get('formats') or [])]
        if info.get('requested_formats'):
            o['requested_formats'] = [{k: f.get(k) for k in FMT_KEYS if f.get(k) is not None} for f in info['requested_formats']]
        for k in ('subtitles', 'automatic_captions'):
            o[k] = {lang: [{'name': f.get('name'), 'ext': f.get('ext')} for f in (fmts or [])][:1]
                    for lang, fmts in (info.get(k) or {}).items()}
        o['thumbnails'] = [{'url': t.get('url'), 'width': t.get('width'), 'preference': t.get('preference')}
                           for t in (info.get('thumbnails') or []) if t.get('url')][-12:]
        return o

    def thumb(ydl, info):
        """A small picture of the video as a data: URI (320 to 720 pixels wide, a JPEG if there is one), or None."""
        ths = [t for t in (info.get('thumbnails') or []) if t.get('url') and str(t.get('url')).startswith('https://')]

        def score(t):
            w = t.get('width') or 0
            jpg = t['url'].split('?')[0].endswith('.jpg')
            return (0 if 320 <= w <= 720 else 1, 0 if jpg else 1, abs(w - 480) if w else 999, -(t.get('preference') or 0))
        for t in sorted(ths, key=score)[:4]:
            try:
                with ydl.urlopen(t['url']) as r:
                    data = r.read(1_500_000)
                if len(data) < 400 or len(data) >= 1_500_000:
                    continue
                mime = 'image/webp' if data[8:12] == b'WEBP' else ('image/png' if data[:4] == b'\x89PNG' else 'image/jpeg')
                return 'data:%s;base64,%s' % (mime, base64.b64encode(data).decode('ascii'))
            except Exception:
                continue
        return None

    req = json.loads(sys.stdin.buffer.read().decode('utf-8'))
    import yt_dlp
    opts = dict(req.get('opts') or {})
    opts.update({'quiet': True, 'no_warnings': False, 'noprogress': True, 'logger': Log(),
                 'color': {'stdout': 'never', 'stderr': 'never'}})
    op = req.get('op')
    emit({'version': yt_dlp.version.__version__})
    if op == 'version':
        emit({'result': {'version': yt_dlp.version.__version__}})
        return
    if op == 'info':
        opts['skip_download'] = True
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(req['url'], download=False)
            res = {'info': trim(info)}
            if req.get('thumb') and info.get('_type') != 'playlist':
                res['thumb'] = thumb(ydl, info)
        emit({'result': res})
        return
    if op == 'download':
        state = {'t': 0.0}

        def hook(d):
            st = d.get('status')
            now = time.time()
            if st == 'downloading' and now - state['t'] < 0.25:
                return
            state['t'] = now
            emit({'progress': {'status': st, 'file': d.get('filename') or d.get('tmpfilename') or '',
                               'got': d.get('downloaded_bytes') or 0,
                               'total': d.get('total_bytes') or d.get('total_bytes_estimate') or 0}})

        def pp_hook(d):
            if d.get('status') == 'started':
                emit({'pp': d.get('postprocessor') or ''})
        opts['progress_hooks'] = [hook]
        opts['postprocessor_hooks'] = [pp_hook]
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(req['url'], download=False)
            emit({'info': trim(info)})
            if info.get('live_status') in ('is_live', 'is_upcoming'):
                emit({'result': {'live': info.get('live_status')}})
                return
            mx = req.get('max_seconds') or 0
            if mx and (info.get('duration') or 0) > mx:
                emit({'result': {'too_long': info.get('duration')}})
                return
            info = ydl.process_ie_result(info, download=True)
            res = {'info': trim(info)}
            if req.get('thumb'):
                res['thumb'] = thumb(ydl, info)
        emit({'result': res})
        return
    emit({'error': 'unknown op %r' % op})


# ---------------------------------------------------------------- MCP over HTTP

_SESSIONS = set()


def tool_result(payload, is_error=False):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    res = {'content': [{'type': 'text', 'text': text}], 'isError': bool(is_error)}
    if isinstance(payload, dict):
        res['structuredContent'] = payload
    return res


def call_tool(name, args):
    try:
        return tool_result(call(name, args or {}))
    except ToolError as e:
        return tool_result({'error': str(e), 'code': getattr(e, 'code', 'failed')}, True)
    except Exception as e:
        log('tool', name, 'crashed:', traceback.format_exc())
        return tool_result({'error': 'The import server hit an error: %s' % (e,), 'code': 'crash'}, True)


def handle(msg):
    """One JSON-RPC message: the answer to send, or None (a notification)."""
    if not isinstance(msg, dict):
        return {'jsonrpc': '2.0', 'id': None, 'error': {'code': -32600, 'message': 'Invalid request'}}
    mid = msg.get('id')
    method = msg.get('method')
    params = msg.get('params') or {}
    if method is None:
        return None                       # a response to something we sent: nothing to do
    if not isinstance(params, dict):
        params = {}
    try:
        if method == 'initialize':
            want = params.get('protocolVersion')
            ver = want if want in PROTOCOLS else PROTOCOLS[1]
            info = params.get('clientInfo') or {}
            log('initialize from', str(info.get('name'))[:60], str(info.get('version'))[:30], 'protocol', want, '->', ver)
            result = {
                'protocolVersion': ver,
                'capabilities': {'tools': {'listChanged': False}},
                'serverInfo': {'name': 'shuo-import', 'title': 'shuo import', 'version': VERSION},
                'instructions': 'Brings YouTube and Bilibili videos (or just their sound) into shuo lessons, with their Chinese captions or a transcript written here.',
            }
        elif method == 'ping':
            result = {}
        elif method == 'tools/list':
            result = {'tools': TOOLS}
        elif method == 'tools/call':
            result = call_tool(params.get('name'), params.get('arguments'))
        elif method == 'resources/list':
            result = {'resources': []}
        elif method == 'resources/templates/list':
            result = {'resourceTemplates': []}
        elif method == 'prompts/list':
            result = {'prompts': []}
        elif str(method).startswith('notifications/'):
            return None
        else:
            if mid is None:
                return None
            return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32601, 'message': 'Method not found: %s' % str(method)[:80]}}
    except Exception as e:
        log('request', method, 'failed:', traceback.format_exc())
        if mid is None:
            return None
        return {'jsonrpc': '2.0', 'id': mid, 'error': {'code': -32603, 'message': str(e)[:300]}}
    if mid is None:
        return None
    return {'jsonrpc': '2.0', 'id': mid, 'result': result}


def _key():
    k = (os.environ.get('SHUO_KEY') or '').strip()   # (a space or new line pasted with it doesn't count)
    return k if len(k) >= 16 else ''


def _people():
    """The keys that may use this server: {key: (name, imports a day or None)}. SHUO_KEY is the owner's (no daily
    limit). SHUO_KEYS lets other people in, "name=key" each (commas or new lines between; each key at least 16
    characters), each with SHUO_DAILY imports a day (10 unless set): a tester gets a key of their own, and taking it out
    of SHUO_KEYS shuts only them out."""
    out = {}
    k = _key()
    if k:
        out[k] = ('owner', None)
    try:
        daily = max(1, int(os.environ.get('SHUO_DAILY') or 10))
    except ValueError:
        daily = 10
    for part in re.split(r'[,;\n]+', os.environ.get('SHUO_KEYS') or ''):
        part = part.strip()
        if '=' not in part:
            continue
        name, key = part.split('=', 1)
        name, key = name.strip().lower(), key.strip()
        if re.match(r'^[a-z0-9_-]{1,32}$', name) and name != 'owner' and len(key) >= 16 and key not in out:
            out[key] = (name, daily)
    return out


def _person(key):
    found = None
    for k, v in _people().items():
        if hmac.compare_digest(key.encode('utf-8'), k.encode('utf-8')):
            found = v
    return found


_who = threading.local()


def who():
    return getattr(_who, 'name', 'owner')


def who_daily():
    return getattr(_who, 'daily', None)


USAGE_FILE = os.path.join(DATA, 'usage.json')
_usage_lock = threading.Lock()


def _usage(add=0):
    """Imports started today (UTC) by the one asking; add counts one more."""
    day = time.strftime('%Y-%m-%d', time.gmtime())
    with _usage_lock:
        try:
            with open(USAGE_FILE, 'r', encoding='utf-8') as f:
                u = json.load(f)
        except (OSError, ValueError):
            u = {}
        if u.get('day') != day:
            u = {'day': day, 'people': {}}
        n = int(u['people'].get(who()) or 0) + add
        if add:
            u['people'][who()] = n
            try:
                os.makedirs(DATA, exist_ok=True)
                with open(USAGE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(u, f)
            except OSError:
                pass
        return n


STATUS_PAGE = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>shuo import</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:34em;margin:2em auto;padding:0 16px;color:#222}
b{color:%s}</style><h1>shuo import</h1><p><b>%s</b></p><p>%s</p><p style="color:#777">Version %s · yt-dlp %s · transcripts: %s · logged in to: %s</p>'''


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'shuo-import'
    sys_version = ''

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body=b'', ctype='application/json', headers=None):
        self.send_response(code)
        if body:
            self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body and self.command != 'HEAD':
            self.wfile.write(body)

    def _route(self):
        """'status', 'mcp' (the right key), 'wrong' (an MCP path with a wrong or missing key), or None."""
        path = urlsplit(self.path).path.rstrip('/') or '/'
        if path in ('/', '/health'):
            return 'status'
        if path == '/mcp' or path.startswith('/mcp/'):
            parts = path.split('/')
            key = unquote(parts[2]) if len(parts) == 3 and parts[2] else ''
            if not key and len(parts) == 2:   # or /mcp with the key as a request header (claude.ai's Request headers)
                a = (self.headers.get('Authorization') or '').strip()
                key = a[7:].strip() if a[:7].lower() == 'bearer ' else (self.headers.get('X-Shuo-Key') or '').strip()
            person = _person(key) if key else None
            if person:
                self._who = person
                return 'mcp'
            return 'wrong'
        return None

    def do_GET(self):
        r = self._route()
        if r == 'status':
            key = bool(_key())
            ok = key and bool(_installed('yt-dlp'))
            others = len(_people()) - (1 if key else 0)
            mp = model_paths()
            body = (STATUS_PAGE % ('#1a7f37' if ok else '#b42318',
                                   'It\'s running.' if ok else 'It\'s running, but not ready yet.',
                                   ('Your key is set: add this server to claude.ai as a connector named "shuo import" (its address ends with /mcp/ and your key).'
                                    + (' %d more %s can use it with their own keys.' % (others, 'person' if others == 1 else 'people') if others else ''))
                                   if key else 'The key isn\'t set yet: in the Space\'s Settings, under Variables and secrets, add a secret named SHUO_KEY (at least 16 characters), then restart the Space.',
                                   VERSION, _installed('yt-dlp') or 'missing', 'yes' if (mp['model'] and mp['vad']) else 'no',
                                   ', '.join(k for k, v in _logins().items() if v) or ('set' if _cookies() else 'none'))).encode('utf-8')
            self._send(200, body, 'text/html; charset=utf-8')
        elif r == 'mcp':
            self._send(405, b'', headers={'Allow': 'POST'})
        else:
            self._send(404, b'{"error":"not found"}')

    def do_HEAD(self):
        self.do_GET()

    def do_DELETE(self):
        r = self._route()
        if r == 'mcp':
            sid = self.headers.get('Mcp-Session-Id')
            if sid:
                _SESSIONS.discard(sid)
            self._send(204)
        else:
            self._send(404, b'{"error":"not found"}')

    def do_OPTIONS(self):
        self._send(204, b'', headers={'Allow': 'GET, POST, DELETE, OPTIONS'})

    def do_POST(self):
        r = self._route()
        if r != 'mcp':
            if r == 'wrong' and not _key():
                self._send(503, b'{"error":"SHUO_KEY is not set on the server"}')
            else:
                self._send(404, b'{"error":"not found"}')
            return
        try:
            n = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            n = 0
        if n <= 0 or n > MAX_BODY:
            self._send(413 if n > MAX_BODY else 400, b'{"error":"bad body"}')
            return
        raw = self.rfile.read(n)
        try:
            msg = json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            self._send(400, json.dumps({'jsonrpc': '2.0', 'id': None, 'error': {'code': -32700, 'message': 'Parse error'}}).encode('utf-8'))
            return
        batch = isinstance(msg, list)
        msgs = msg if batch else [msg]
        if not msgs or len(msgs) > 20:
            self._send(400, b'{"error":"bad batch"}')
            return
        _who.name, _who.daily = self._who
        try:
            answers = [a for a in (handle(m) for m in msgs) if a is not None]
        finally:
            _who.name, _who.daily = 'owner', None
        headers = {}
        if any(isinstance(m, dict) and m.get('method') == 'initialize' for m in msgs):
            sid = uuid.uuid4().hex
            _SESSIONS.add(sid)
            headers['Mcp-Session-Id'] = sid
        if not answers:
            self._send(202, b'', headers=headers)
            return
        body = json.dumps(answers if batch else answers[0], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        self._send(200, body, 'application/json', headers)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


def start_pot_provider():
    """YouTube's "proof of origin" tokens, made here (bgutil's provider, when the image has it): YouTube turns away a
    cloud server's downloads less often with them. yt-dlp's plugin finds it at 127.0.0.1:4416."""
    main = os.environ.get('SHUO_POT') or os.path.expanduser('~/bgutil/server/build/main.js')
    node = shutil.which('node')
    if not (node and os.path.isfile(main)):
        log('no PO token provider here (fine: YouTube may turn more downloads away)')
        return
    def run():
        while True:
            try:
                p = subprocess.Popen([node, main], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                CTX.store['pot'] = True
                log('PO token provider started')
                p.wait()
                CTX.store['pot'] = False
                log('PO token provider stopped (%s): starting it again in 10 s' % p.returncode)
            except Exception:
                log('PO token provider failed to start', traceback.format_exc()[-300:])
            time.sleep(10)
    threading.Thread(target=run, daemon=True, name='pot').start()


def serve():
    os.makedirs(JOBS_DIR, exist_ok=True)
    CTX.store['queue'] = _queue.Queue()
    threading.Thread(target=_worker_loop, args=(CTX.store['queue'],), daemon=True, name='jobs').start()
    threading.Thread(target=cleanup_cache, daemon=True).start()
    threading.Thread(target=lambda: (time.sleep(30), update_ytdlp()), daemon=True, name='update').start()

    def daily():
        while True:
            time.sleep(3600)
            update_ytdlp()
            cleanup_cache()
    threading.Thread(target=daily, daemon=True, name='daily').start()
    start_pot_provider()
    log('shuo import', VERSION, 'on port', PORT, '| key', 'set' if _key() else 'NOT SET', '| yt-dlp', _installed('yt-dlp'),
        '| ffmpeg', bool(ffmpeg_path()), '| js', list(js_runtimes()), '| transcripts', bool(model_paths()['model']))
    Server(('0.0.0.0', PORT), Handler).serve_forever()


if __name__ == '__main__':
    if '--ytw' in sys.argv:
        try:
            ytw_main()
        except KeyboardInterrupt:
            sys.stdout.buffer.write(b'{"error":"Cancelled"}\n')
        except BaseException as e:
            sys.stdout.buffer.write((json.dumps({'error': str(e) or repr(e)}) + '\n').encode('utf-8'))
            sys.stdout.buffer.flush()
            sys.exit(1)
    elif '--selftest' in sys.argv:
        rep = selftest(quick='--full' not in sys.argv)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        sys.exit(0 if rep.get('ok') else 1)
    else:
        serve()
