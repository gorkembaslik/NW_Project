# YouTube Partnership Analyzer

Ever wondered whether sponsorships actually hurt a YouTuber's relationship with their audience? This tool answers that. Paste in a channel URL and it pulls their recent videos, separates the sponsored ones from the organic ones, and compares how each group performs in terms of both audience sentiment and engagement.

Built as a partnership evaluation tool for Foreo, but it works for any YouTube channel.

## What it does

- Fetches a channel's recent videos (you control how far back and how many)
- Detects sponsored videos using YouTube's own paid promotion API flag and description disclosures
- Analyzes comment sentiment using a combination of VADER and TextBlob
- Shows you a side-by-side comparison: sponsored vs. organic across sentiment, engagement rate, views, likes, and comments

## How to run it locally

You'll need a [YouTube Data API v3 key](https://console.developers.google.com/).

```bash
git clone https://github.com/gorkembaslik/youtube-partnership-analyzer
cd youtube-partnership-analyzer
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
YOUTUBE_API_KEY=your_key_here
```

Then start the server:
```bash
python -m uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

## Supported URL formats

Any of these work:
- `https://www.youtube.com/@ChannelName`
- `https://www.youtube.com/channel/UCxxxxxxx`
- `https://www.youtube.com/c/ChannelName`
- `https://www.youtube.com/ChannelName`

## Tech stack

FastAPI backend, plain HTML/JS frontend, deployed on Hugging Face Spaces via Docker. Sentiment scoring uses NLTK's VADER (60% weight) and TextBlob (40% weight). Sponsorship detection uses YouTube's `paidProductPlacementDetails` API field, no scraping needed.
