import asyncio
import json

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

load_dotenv()

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()



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
