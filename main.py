import asyncio
import json

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from sse_starlette.sse import EventSourceResponse

load_dotenv()

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/cookie-status")
async def cookie_status():
    from analyzer import cookies_loaded
    return {"loaded": cookies_loaded()}


@app.get("/debug-sponsorship", response_class=PlainTextResponse)
async def debug_sponsorship(video_id: str):
    import requests
    from analyzer import _BROWSER_HEADERS, _get_yt_cookies

    lines = []
    from analyzer import _get_yt_cookies, cookies_loaded
    cookies = _get_yt_cookies()

    if cookies:
        names = [c.name for c in cookies]
        lines += [
            f"cookies.txt loaded: YES  ({len(names)} cookies)",
            f"Session valid:      {cookies_loaded()}",
            "",
        ]
    else:
        lines += ["cookies.txt: NOT FOUND — place it in the project folder", ""]

    try:
        r = requests.get(
            f"https://www.youtube.com/watch?v={video_id}",
            headers=_BROWSER_HEADERS,
            cookies=cookies,
            timeout=10,
        )
        text = r.text
        lines += [
            f"HTTP status:                        {r.status_code}",
            f"Response length:                    {len(text)} chars",
            f"'ytp-paid-content-overlay-link':    {'YES' if 'ytp-paid-content-overlay-link' in text else 'NO'}",
            f"'Includes paid promotion':          {'YES' if 'Includes paid promotion' in text else 'NO'}",
            f"'ytp-paid-content-overlay':         {'YES' if 'ytp-paid-content-overlay' in text else 'NO'}",
            f"'logged_in\":\"1':                   {'\"logged_in\":\"1\"' in text}",
        ]
    except Exception as e:
        lines.append(f"Request failed: {e}")

    return "\n".join(lines)


@app.get("/analyze")
async def analyze(
    url: str,
    months: int = 6,
    max_videos: int = 15,
    max_comments: int = 500,
    translate: bool = False,
):
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def progress_callback(msg: str):
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "message": msg})

    def run_analysis():
        try:
            from analyzer import evaluate_channel
            result = evaluate_channel(
                url,
                months=months,
                max_videos=max_videos,
                max_comments=max_comments,
                translate=translate,
                progress_callback=progress_callback,
            )
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "result", "data": result}
            )
        except Exception as e:
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": str(e)}
            )

    async def event_generator():
        loop.run_in_executor(None, run_analysis)
        while True:
            event = await queue.get()
            yield {"data": json.dumps(event)}
            if event["type"] in ("result", "error"):
                break

    return EventSourceResponse(event_generator())
