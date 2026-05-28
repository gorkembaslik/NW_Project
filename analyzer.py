import os
import re
import threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import emoji
import isodate
import nltk
import numpy as np
import requests
import translators as ts
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import detect, LangDetectException
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

_lang_lock = threading.Lock()

_BROWSER_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/125.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
}


class EnhancedSentimentAnalyzer:
    def __init__(self):
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.short_phrases = {
            'good', 'nice', 'great', 'awesome', 'wow', 'cool', 'amazing',
            'helpful', 'excellent', 'superb', 'wonderful', 'fantastic',
            'very helpful', 'really good', 'really nice', 'really helpful',
            'very nice', 'very good', 'very impressive', 'really impressive'
        }
        self.stop_words = set(stopwords.words('english'))

    def remove_emojis_and_symbols(self, text):
        text = emoji.replace_emoji(text, replace='')
        text = re.sub(
            r'[\U0001F300-\U0001F5FF\U0001F900-\U0001F9FF'
            r'\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]',
            '', text
        )
        return text

    def normalize_text(self, text):
        text = text.lower()
        text = self.remove_emojis_and_symbols(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return ' '.join(text.split())

    def filter_comment(self, comment, translate=False):
        try:
            normalized_text = self.normalize_text(comment)

            if translate and len(normalized_text.split()) >= 5:
                try:
                    with _lang_lock:
                        language = detect(normalized_text)
                    if language != 'en':
                        try:
                            with _lang_lock:
                                normalized_text = ts.translate_text(
                                    query_text=normalized_text,
                                    from_language=language,
                                    translator='google'
                                )
                        except Exception as e:
                            print(f"Translation failed: {e}")
                            return None
                except LangDetectException:
                    return None

            words = [w for w in normalized_text.split() if w not in self.stop_words]
            filtered_text = ' '.join(words)

            if (len(filtered_text.split()) < 3
                    or filtered_text.strip() in self.short_phrases
                    or len(filtered_text.strip()) < 10):
                return None

            return filtered_text
        except Exception as e:
            print(f"Error filtering comment: {e}")
            return None

    def analyze_sentiment(self, comments, translate=False):
        if not comments:
            return 0, {'positive': 0, 'neutral': 0, 'negative': 0}

        filtered = [self.filter_comment(c, translate=translate) for c in comments]
        filtered = [c for c in filtered if c is not None]

        if not filtered:
            return 0, {'positive': 0, 'neutral': 0, 'negative': 0}

        scores = []
        counts = {'positive': 0, 'neutral': 0, 'negative': 0}

        for comment in filtered:
            vader = self.vader_analyzer.polarity_scores(comment)['compound']
            blob = TextBlob(comment).sentiment.polarity
            combined = 0.6 * vader + 0.4 * blob

            if combined > 0.1:
                counts['positive'] += 1
            elif combined < -0.1:
                counts['negative'] += 1
            else:
                counts['neutral'] += 1

            scores.append(combined)

        normalized = (float(np.median(scores)) + 1) / 2
        return normalized, counts


def get_channel_id(url, youtube=None):
    url = url.strip()

    # 1. Direct /channel/UCxxx — ID is in the URL itself
    m = re.search(r'/channel/(UC[\w-]+)', url)
    if m:
        return m.group(1)

    # 2. @handle format — use the API forHandle parameter
    m = re.search(r'/@([\w.-]+)', url)
    if m and youtube:
        try:
            resp = youtube.channels().list(part='id', forHandle=m.group(1)).execute()
            if resp.get('items'):
                return resp['items'][0]['id']
        except Exception:
            pass

    # 3. Legacy /user/ or /c/ format — use forUsername
    m = re.search(r'/(?:user|c)/([\w.-]+)', url)
    if m and youtube:
        try:
            resp = youtube.channels().list(part='id', forUsername=m.group(1)).execute()
            if resp.get('items'):
                return resp['items'][0]['id']
        except Exception:
            pass

    # 4. Last resort: scrape the page
    try:
        response = requests.get(url, headers=_BROWSER_HEADERS, timeout=15)
        if response.status_code == 200:
            for pattern in (
                r'"externalId":"(UC[\w-]+)"',
                r'"channelId":"(UC[\w-]+)"',
                r'"browseId":"(UC[\w-]+)"',
            ):
                m = re.search(pattern, response.text)
                if m:
                    return m.group(1)
    except Exception:
        pass

    return None


def get_channel_name(youtube, channel_id):
    resp = youtube.channels().list(part='snippet', id=channel_id).execute()
    if resp.get('items'):
        return resp['items'][0]['snippet']['title']
    return None


def get_channel_videos(youtube, channel_id, months, max_videos, emit=None):
    videos = []
    seen_ids = set()
    cutoff = datetime.now() - timedelta(days=months * 30)
    base_url = "https://www.youtube.com/watch?v="

    # Two passes: medium (4-20 min) then long (>20 min).
    # This excludes Shorts at the API level so they never fill up result pages.
    for duration_filter in ('medium', 'long'):
        if len(videos) >= max_videos:
            break

        next_page_token = None
        reached_cutoff = False

        while len(videos) < max_videos and not reached_cutoff:
            request = youtube.search().list(
                part='snippet',
                channelId=channel_id,
                maxResults=50,
                type='video',
                videoDuration=duration_filter,
                order='date',
                pageToken=next_page_token
            ).execute()

            items = request.get('items', [])
            video_ids = [
                i['id']['videoId'] for i in items
                if 'videoId' in i['id'] and i['id']['videoId'] not in seen_ids
            ]

            if not video_ids:
                break

            details_resp = youtube.videos().list(
                part="contentDetails,snippet",
                id=",".join(video_ids)
            ).execute()
            details_by_id = {item['id']: item for item in details_resp.get('items', [])}

            for item in items:
                if len(videos) >= max_videos:
                    break
                if 'videoId' not in item['id']:
                    continue
                video_id = item['id']['videoId']
                if video_id in seen_ids:
                    continue
                details = details_by_id.get(video_id)
                if not details:
                    continue

                publish_date = datetime.strptime(
                    details['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ"
                )
                if publish_date < cutoff:
                    reached_cutoff = True
                    break

                duration = details['contentDetails']['duration']
                try:
                    seconds = isodate.parse_duration(duration).total_seconds()
                except Exception:
                    continue

                if seconds <= 180:
                    continue

                seen_ids.add(video_id)
                videos.append({
                    'video_id': video_id,
                    'video_title': item['snippet']['title'],
                    'video_url': f"{base_url}{video_id}",
                    'description': details['snippet'].get('description', ''),
                })
                if emit:
                    emit(f"Found {len(videos)}/{max_videos}: {item['snippet']['title'][:55]}")

            next_page_token = request.get('nextPageToken')
            if not next_page_token:
                break

    return videos


def batch_get_video_metrics(youtube, video_ids):
    metrics = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        resp = youtube.videos().list(part="statistics", id=",".join(chunk)).execute()
        for item in resp.get('items', []):
            stats = item['statistics']
            metrics[item['id']] = {
                'likes': int(stats.get('likeCount', 0)),
                'views': int(stats.get('viewCount', 0)),
                'comments': int(stats.get('commentCount', 0)),
            }
    return metrics


def get_video_comments(youtube, video_id, max_comments=500):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=min(100, max_comments)
        ).execute()

        comments.extend([
            item['snippet']['topLevelComment']['snippet']['textDisplay']
            for item in request.get('items', [])
        ])

        while 'nextPageToken' in request and len(comments) < max_comments:
            remaining = max_comments - len(comments)
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, remaining),
                pageToken=request['nextPageToken']
            ).execute()
            comments.extend([
                item['snippet']['topLevelComment']['snippet']['textDisplay']
                for item in request.get('items', [])
            ])

        return comments[:max_comments]
    except HttpError as e:
        if e.resp.status == 403 and 'commentsDisabled' in str(e):
            return None
        raise


_SPONSOR_RE = re.compile(
    r'#ad\b|#sponsored\b|#paidpartnership\b|#paidpromotion\b'
    r'|paid\s+promotion|paid\s+partnership|sponsored\s+by'
    r'|in\s+partnership\s+with|this\s+video\s+(is\s+)?sponsored',
    re.IGNORECASE,
)

def _get_yt_cookies():
    """Load YouTube session cookies from cookies.txt (Netscape format)."""
    cookie_path = os.path.join(os.path.dirname(__file__), 'cookies.txt')
    if not os.path.exists(cookie_path):
        return None
    import http.cookiejar
    jar = http.cookiejar.MozillaCookieJar(cookie_path)
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
        return jar
    except Exception as e:
        print(f"Failed to load cookies.txt: {e}")
        return None


def cookies_loaded():
    jar = _get_yt_cookies()
    if not jar:
        return False
    names = {c.name for c in jar}
    return bool(names & {'SID', 'SAPISID', '__Secure-3PSID'})


def check_sponsorship(video_id, description=''):
    # Signal 1: description disclosure — no extra request needed
    if description and _SPONSOR_RE.search(description):
        return True

    # Signal 2: fetch with session cookies so YouTube returns the
    # logged-in page which includes ytp-paid-content-overlay-link
    cookies = _get_yt_cookies()
    if not cookies:
        return False

    try:
        response = requests.get(
            f"https://www.youtube.com/watch?v={video_id}",
            headers=_BROWSER_HEADERS,
            cookies=cookies,
            timeout=10,
        )
        response.raise_for_status()
        if 'Includes paid promotion' in response.text:
            return True
    except Exception:
        pass

    return False


def _process_video(api_key, video, metrics_dict, analyzer, max_comments, translate):
    youtube = build('youtube', 'v3', developerKey=api_key)
    video_id = video['video_id']

    comments = get_video_comments(youtube, video_id, max_comments)
    if comments is None:
        return None

    stats = metrics_dict.get(video_id, {'likes': 0, 'views': 0, 'comments': 0})
    sentiment_score, sentiment_counts = analyzer.analyze_sentiment(comments, translate=translate)
    is_sponsored = check_sponsorship(video_id, description=video.get('description', ''))

    return {
        'video': video,
        'is_sponsored': is_sponsored,
        'likes': stats['likes'],
        'views': stats['views'],
        'comments_count': stats['comments'],
        'sentiment_score': sentiment_score,
        'sentiment_counts': sentiment_counts,
    }


def evaluate_channel(channel_url, months=6, max_videos=15, max_comments=500, translate=False, progress_callback=None):
    def emit(msg):
        print(msg)
        if progress_callback:
            progress_callback(msg)

    api_key = os.getenv('YOUTUBE_API_KEY')
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY not set in environment")

    youtube = build('youtube', 'v3', developerKey=api_key)
    analyzer = EnhancedSentimentAnalyzer()

    emit("Extracting channel ID...")
    channel_id = get_channel_id(channel_url, youtube=youtube)
    if not channel_id:
        raise ValueError("Could not extract channel ID from URL")

    emit("Fetching channel info...")
    channel_name = get_channel_name(youtube, channel_id)

    emit(f"Fetching video list (last {months} months, max {max_videos})...")
    videos = get_channel_videos(youtube, channel_id, months, max_videos, emit=emit)
    emit(f"Found {len(videos)} eligible videos")

    if not videos:
        return _empty_results(channel_name, channel_id)

    emit("Fetching engagement metrics (batched)...")
    video_ids = [v['video_id'] for v in videos]
    metrics_dict = batch_get_video_metrics(youtube, video_ids)

    sponsored_sentiments = []
    unsponsored_sentiments = []
    sp = {"likes": 0, "views": 0, "comments": 0, "count": 0}
    un = {"likes": 0, "views": 0, "comments": 0, "count": 0}
    sp_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    un_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    sp_videos = []
    un_videos = []

    completed = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        future_to_video = {
            pool.submit(_process_video, api_key, video, metrics_dict, analyzer, max_comments, translate): video
            for video in videos
        }
        for future in as_completed(future_to_video):
            video = future_to_video[future]
            completed += 1
            emit(f"[{completed}/{len(videos)}] {video['video_title'][:60]}")

            try:
                result = future.result()
            except Exception as e:
                print(f"Error processing {video['video_id']}: {e}")
                continue

            if result is None:
                continue

            if result['is_sponsored']:
                sponsored_sentiments.append(result['sentiment_score'])
                sp_videos.append(result['video'])
                sp['likes'] += result['likes']
                sp['views'] += result['views']
                sp['comments'] += result['comments_count']
                sp['count'] += 1
                for k in result['sentiment_counts']:
                    sp_counts[k] += result['sentiment_counts'][k]
            else:
                unsponsored_sentiments.append(result['sentiment_score'])
                un_videos.append(result['video'])
                un['likes'] += result['likes']
                un['views'] += result['views']
                un['comments'] += result['comments_count']
                un['count'] += 1
                for k in result['sentiment_counts']:
                    un_counts[k] += result['sentiment_counts'][k]

    avg_sp_sent = float(np.mean(sponsored_sentiments)) if sponsored_sentiments else 0
    avg_un_sent = float(np.mean(unsponsored_sentiments)) if unsponsored_sentiments else 0
    avg_sp_eng = (sp['likes'] + sp['comments']) / (sp['views'] or 1)
    avg_un_eng = (un['likes'] + un['comments']) / (un['views'] or 1)

    return {
        'channel_name': channel_name,
        'channel_id': channel_id,
        'sponsored_sentiment': round(avg_sp_sent, 4),
        'unsponsored_sentiment': round(avg_un_sent, 4),
        'sponsored_engagement': round(avg_sp_eng, 4),
        'unsponsored_engagement': round(avg_un_eng, 4),
        'num_sponsored': len(sp_videos),
        'num_unsponsored': len(un_videos),
        'sponsored_sentiment_counts': sp_counts,
        'unsponsored_sentiment_counts': un_counts,
        'sponsored_likes': sp['likes'],
        'unsponsored_likes': un['likes'],
        'sponsored_views': sp['views'],
        'unsponsored_views': un['views'],
        'sponsored_comments': sp['comments'],
        'unsponsored_comments': un['comments'],
    }


def _empty_results(channel_name, channel_id):
    zero_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    return {
        'channel_name': channel_name,
        'channel_id': channel_id,
        'sponsored_sentiment': 0, 'unsponsored_sentiment': 0,
        'sponsored_engagement': 0, 'unsponsored_engagement': 0,
        'num_sponsored': 0, 'num_unsponsored': 0,
        'sponsored_sentiment_counts': zero_counts,
        'unsponsored_sentiment_counts': zero_counts,
        'sponsored_likes': 0, 'unsponsored_likes': 0,
        'sponsored_views': 0, 'unsponsored_views': 0,
        'sponsored_comments': 0, 'unsponsored_comments': 0,
    }
