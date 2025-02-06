import sys

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QLabel, QLineEdit, QPushButton, QTableWidget, 
                               QTableWidgetItem, QMessageBox, QProgressDialog,
                               QFrame, QHBoxLayout, QHeaderView, QSizePolicy) 

from PySide6.QtCore import Qt, QThread, Signal , QPropertyAnimation, Property

from PySide6.QtGui import QFont, QIcon, QColor, QPainter, QBrush

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import os
import re
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
from datetime import datetime, timedelta

import emoji

from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
import isodate

import translators as ts

class EnhancedSentimentAnalyzer:
    def __init__(self):
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # uninformative phrases
        self.short_phrases = {
            'good', 'nice', 'great', 'awesome', 'wow', 'cool', 'amazing', 
            'helpful', 'excellent', 'superb', 'wonderful', 'fantastic', 
            'very helpful', 'really good', 'really nice', 'really helpful',
            'very nice', 'very good', 'very impressive', 'really impressive'
        }
        
        # Prepare English stopwords
        self.stop_words = set(stopwords.words('english'))

    def remove_emojis_and_symbols(self, text):
      
        text = emoji.replace_emoji(text, replace='')
        
        # Remove transport and map symbols and other special symbols
        text = re.sub(r'[\U0001F300-\U0001F5FF\U0001F900-\U0001F9FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        
        return text
    
    def normalize_text(self, text):

        text = text.lower()        
        text = self.remove_emojis_and_symbols(text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        text = ' '.join(text.split())
        
        return text
    
    def filter_comment(self, comment):
       
        try:
            normalized_text = self.normalize_text(comment)
            
            try:
                language = detect(normalized_text)
                if language != 'en':
                    try:
                        normalized_text = ts.translate_text(query_text=normalized_text, from_language=language, translator='google') #ts.deepl(normalized_text, from_language=language, to_language='en')
                    except Exception as e:
                        print(f"Translation failed: {e}")
                        return None
            except LangDetectException:
                return None
            
            words = normalized_text.split()
            words = [word for word in words if word not in self.stop_words]
            
            filtered_text = ' '.join(words)
            
            if (len(filtered_text.split()) < 3 or 
                filtered_text.strip() in self.short_phrases or 
                len(filtered_text.strip()) < 10):
                return None
            
            return filtered_text
        
        except Exception as e:
            print(f"Error filtering comment: {e}")
            return None
        
    def analyze_sentiment(self, comments):
    
        if not comments:
            return 0, {'positive': 0, 'neutral': 0, 'negative': 0}
        
        filtered_comments = [self.filter_comment(comment) for comment in comments]
        filtered_comments = [comment for comment in filtered_comments if comment is not None]
        
        if not filtered_comments:
            return 0, {'positive': 0, 'neutral': 0, 'negative': 0}
        
        sentiment_scores = []
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for comment in filtered_comments:
        
            vader_scores = self.vader_analyzer.polarity_scores(comment)
            vader_sentiment = vader_scores['compound']
            
            blob_sentiment = TextBlob(comment).sentiment.polarity
            
            # Weighted average of different sentiment methods
            combined_sentiment = (
                0.6 * vader_sentiment +  # VADER for social media text
                0.4 * blob_sentiment     # TextBlob provides additional linguistic analysis
            )
            
            # Categorize the sentiment
            if combined_sentiment > 0.1:
                sentiment_counts['positive'] += 1
            elif combined_sentiment < -0.1:
                sentiment_counts['negative'] += 1
            else:
                sentiment_counts['neutral'] += 1
            
            sentiment_scores.append(combined_sentiment)
        
        if not sentiment_scores:
            return 0, sentiment_counts
        
        median_sentiment = np.median(sentiment_scores)
        
        normalized_sentiment = (median_sentiment + 1) / 2
        
        return normalized_sentiment, sentiment_counts

class EvaluationWorker(QThread):
    finished = Signal(dict)
    progress = Signal(str)
    error = Signal(str)

    def __init__(self, channel_url, api_key='AIzaSyBiSugmzbENy_cAVexEYAM5kYylMcv6ZvA', months=6, max_videos=15):
        super().__init__()
        self.channel_url = channel_url
        self.api_key = api_key
        self.months = months
        self.max_videos = max_videos
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.setup_nltk()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()

    def setup_nltk(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)

    def run(self):
        try:
            channel_id = self.extract_channel_id(self.channel_url)
            self.progress.emit("Evaluating channel...")
            
            results = self.evaluate_channel(channel_id)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def get_channel_videos(self, channel_id):
        self.progress.emit("Fetching channel videos...")
        videos = []
        next_page_token = None
        base_video_url = "https://www.youtube.com/watch?v="
        x_months_ago = datetime.now() - timedelta(days=self.months * 30)

        while len(videos) < self.max_videos:
            request = self.youtube.search().list(
                part='snippet',
                channelId=channel_id,
                maxResults=min(10, self.max_videos - len(videos)),
                type='video',
                order='date',
                pageToken=next_page_token
            ).execute()

            video_ids = [item['id']['videoId'] for item in request.get('items', []) if 'videoId' in item['id']]
            if not video_ids:
                break

            video_details_request = self.youtube.videos().list(
                part="contentDetails, snippet",
                id=",".join(video_ids)
            ).execute()

            for item, video_details in zip(request['items'], video_details_request['items']):
                if 'videoId' in item['id']:
                    video_id = item['id']['videoId']
                    video_title = item['snippet']['title']
                    video_url = f"{base_video_url}{video_id}"

                    publish_date_str = video_details['snippet']['publishedAt']
                    publish_date = datetime.strptime(publish_date_str, "%Y-%m-%dT%H:%M:%SZ")

                    if publish_date < x_months_ago:
                        continue

                    duration = video_details['contentDetails']['duration']
                    seconds = isodate.parse_duration(duration).total_seconds()

                    if seconds <= 180:
                        continue

                    if 'M' not in duration or (duration.startswith('PT') and 'S' in duration and 'M' not in duration):
                        continue

                    videos.append({'video_id': video_id, 'video_title': video_title, 'video_url': video_url})

            next_page_token = request.get('nextPageToken')

            if not next_page_token:
                break

        return videos

    def check_sponsorship_disclaimer(self, video_id):
        url = f"https://www.youtube.com/watch?v={video_id}"
        self.progress.emit(f"Checking sponsorship disclaimer for {video_id}")

        try:
            response = requests.get(url)
            response.raise_for_status()
            return 'paidContentOverlayRenderer' in response.text
        except requests.exceptions.HTTPError as http_err:
            self.error.emit(f"HTTP error occurred: {http_err}")
        except Exception as err:
            self.error.emit(f"An error occurred: {err}")

        return False

    def get_video_comments(self, video_id):
        self.progress.emit(f"Fetching comments for video {video_id}")
        comments = []

        try:
            request = self.youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100
            ).execute()

            comments.extend([item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in request['items']])

            while 'nextPageToken' in request:
                request = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,
                    pageToken=request['nextPageToken']
                ).execute()
                comments.extend([item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in request['items']])

            return comments
        except HttpError as e:
            if e.resp.status == 403 and 'commentsDisabled' in str(e):
                return None
            else:
                self.error.emit(f"Error fetching comments for video {video_id}: {str(e)}")
                raise

    def extract_channel_id(self, url):
        url = ' '.join(url.split())
        response = requests.get(url)
        if response.status_code == 200:
            match = re.search(r'\"externalId\":\"(UC[\w-]+)\"', response.text)
            if match:
                return match.group(1)
        return

    def analyze_sentiment(self, comments):
        return self.sentiment_analyzer.analyze_sentiment(comments)
    
    def get_channel_name(self, channel_id):
        request = self.youtube.channels().list(part='snippet', id=channel_id).execute()
        if request['items']:
            return request['items'][0]['snippet']['title']
        return None

    def get_video_engagement_metrics(self, video_id):
        try:
            request = self.youtube.videos().list(part="statistics", id=video_id).execute()
            if 'items' in request and len(request['items']) > 0:
                stats = request['items'][0]['statistics']
                likes = int(stats.get('likeCount', 0))
                views = int(stats.get('viewCount', 0))
                comments_count = int(stats.get('commentCount', 0))
                return likes, views, comments_count
            else:
                return 0, 0, 0
        except Exception as e:
            self.error.emit(f"Error fetching engagement metrics for video {video_id}: {str(e)}")
            return 0, 0, 0

    def evaluate_channel(self, channel_id):
        channel_name = self.get_channel_name(channel_id)
        videos = self.get_channel_videos(channel_id)
        sponsored_sentiments = []
        unsponsored_sentiments = []
        sponsored_engagement_metrics = {"likes": 0, "views": 0, "comments": 0, "count": 0}
        unsponsored_engagement_metrics = {"likes": 0, "views": 0, "comments": 0, "count": 0}
        sponsored_videos = []
        unsponsored_videos = []
        
        # Initialize sentiment counts
        sponsored_sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        unsponsored_sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}

        for video in videos:
            self.progress.emit(f"Analyzing video: {video['video_title']}")
            video_id = video['video_id']
            comments = self.get_video_comments(video_id)
            if comments is None:
                continue
            
            likes, views, comments_count = self.get_video_engagement_metrics(video_id)
            sentiment_score, sentiment_counts = self.analyze_sentiment(comments)

            if self.check_sponsorship_disclaimer(video_id):
                sponsored_sentiments.append(sentiment_score)
                sponsored_videos.append(video)
                sponsored_engagement_metrics['likes'] += likes
                sponsored_engagement_metrics['views'] += views
                sponsored_engagement_metrics['comments'] += comments_count
                sponsored_engagement_metrics['count'] += 1

                for key in sentiment_counts:
                    sponsored_sentiment_counts[key] += sentiment_counts[key]
            else:
                unsponsored_sentiments.append(sentiment_score)
                unsponsored_videos.append(video)
                unsponsored_engagement_metrics['likes'] += likes
                unsponsored_engagement_metrics['views'] += views
                unsponsored_engagement_metrics['comments'] += comments_count
                unsponsored_engagement_metrics['count'] += 1

                for key in sentiment_counts:
                    unsponsored_sentiment_counts[key] += sentiment_counts[key]

        num_sponsored = len(sponsored_videos)
        num_unsponsored = len(unsponsored_videos)

        avg_sponsored_sentiment = np.mean(sponsored_sentiments) if sponsored_sentiments else 0
        avg_unsponsored_sentiment = np.mean(unsponsored_sentiments) if unsponsored_sentiments else 0

        avg_sponsored_engagement_score = (sponsored_engagement_metrics['likes'] + sponsored_engagement_metrics['comments']) / (sponsored_engagement_metrics['views'] or 1)
        avg_unsponsored_engagement_score = (unsponsored_engagement_metrics['likes'] + unsponsored_engagement_metrics['comments']) / (unsponsored_engagement_metrics['views'] or 1)

        results = {
            'channel_name': channel_name,
            'channel_id': channel_id,
            'sponsored_sentiment': avg_sponsored_sentiment,
            'unsponsored_sentiment': avg_unsponsored_sentiment,
            'sponsored_engagement': avg_sponsored_engagement_score,
            'unsponsored_engagement': avg_unsponsored_engagement_score,
            'num_sponsored': num_sponsored,
            'num_unsponsored': num_unsponsored,
            'sponsored_sentiment_counts': sponsored_sentiment_counts,
            'unsponsored_sentiment_counts': unsponsored_sentiment_counts,
            'sponsored_likes': sponsored_engagement_metrics['likes'],
            'unsponsored_likes': unsponsored_engagement_metrics['likes'],
            'sponsored_views': sponsored_engagement_metrics['views'],
            'unsponsored_views': unsponsored_engagement_metrics['views'],
            'sponsored_comments': sponsored_engagement_metrics['comments'],
            'unsponsored_comments': unsponsored_engagement_metrics['comments']
        }

        return results
  

class ModernFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modernFrame")
        self.setStyleSheet("""
            QFrame#modernFrame {
                background-color: white;
                border-radius: 10px;
                border: 1px solid #e0e0e0;
            }
        """)

class YouTubePartnerEstimator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YouTube Partnership Analyzer")
        
        screen_geometry = QApplication.primaryScreen().geometry()
        self.resize(
            int(screen_geometry.width() * 0.7),
            int(screen_geometry.height() * 0.7)
        )
        self.setMinimumSize(640, 480)

        icon_path = self._get_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        self.setStyleSheet(ResponsiveStylesheet.get_dynamic_stylesheet())

        self._setup_central_widget()
        self._create_header()
        self._create_input_section()
        self._create_results_table()

        self.evaluate_button.clicked.connect(self.start_evaluation)
        

    def _get_icon_path(self):
        
        logo_path = os.path.join(getattr(sys, '_MEIPASS', ''), 'Foreo_Logo.png')
        return logo_path if os.path.exists(logo_path) else None

    def _setup_central_widget(self):
        
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        self.setCentralWidget(central_widget)
        self.main_layout = main_layout

    def _create_header(self):
        
        title = QLabel("YouTube Partnership Analyzer")
        title.setFont(ScalableFont.get_responsive_font(20, QFont.Weight.Bold))
        title.setStyleSheet("color: #D47AB3;")
        title.setAlignment(Qt.AlignCenter)

        
        self.main_layout.addWidget(title)
        

    def _create_input_section(self):
        
        input_layout = QHBoxLayout()
        input_layout.setSpacing(10)

        self.url_input = ResponsiveInput("YouTube Channel URL")
        self.months_input = ResponsiveInput("Months to Analyze")
        self.max_videos_input = ResponsiveInput("Max Videos")
        
        self.evaluate_button = ResponsiveButton("Analyze Channel")
        
        input_layout.addWidget(self.url_input)
        input_layout.addWidget(self.months_input)
        input_layout.addWidget(self.max_videos_input)
        input_layout.addWidget(self.evaluate_button)

        self.main_layout.addLayout(input_layout)

    def _create_results_table(self):
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(['Metric', 'Sponsored', 'Organic'])
        
        
        header = self.results_table.horizontalHeader()
        header_font = QFont()
        header_font.setPointSize(13)  # Set font size
        header_font.setBold(True)     # Make font bold
        header.setFont(header_font)
        header.setStyleSheet("color: #445464")
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setDefaultAlignment(Qt.AlignCenter)

        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.main_layout.addWidget(self.results_table)

    def start_evaluation(self):
        if not self._validate_input():
            return

        self._setup_evaluation()
        self._show_progress_dialog()

    def _validate_input(self):
        
        if not self.url_input.text():
            self.show_error_message("Please enter a YouTube channel URL")
            return False

        try:
            months = int(self.months_input.text())
            if months <= 0:
                self.show_error_message("Please enter a positive number of months")
                return False
        except ValueError:
            self.show_error_message("Invalid number of months")
            return False

        try:
            max_videos = int(self.max_videos_input.text())
            if max_videos <= 0:
                self.show_error_message("Please enter a positive number of maximum videos")
                return False
        except ValueError:
            self.show_error_message("Invalid maximum number of videos")
            return False

        return True

    def _setup_evaluation(self):
        
        self.evaluate_button.setEnabled(False)
        months = int(self.months_input.text())
        max_videos = int(self.max_videos_input.text())
        self.worker = EvaluationWorker(self.url_input.text(), months=months, max_videos=max_videos)
        self._connect_worker_signals()
        self.worker.start()

    def _connect_worker_signals(self):
        
        self.worker.finished.connect(self.handle_results)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.handle_error)

    def _show_progress_dialog(self):
        
        self.progress_dialog = QProgressDialog("Evaluating channel...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Analysis in Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setStyleSheet(self._get_progress_dialog_style())
        self.progress_dialog.canceled.connect(self._handle_cancel_operation)
        self.progress_dialog.show()

    def _handle_cancel_operation(self):
        
        self.worker.terminate()  
        self.evaluate_button.setEnabled(True)  
        self.progress_dialog.hide()  
        QMessageBox.information(self, "Operation Canceled", "The evaluation process has been canceled.")

    def update_results_table(self, results):
        
        metrics = [
            ('Sentiment Score', f"{results['sponsored_sentiment']:.3f}", f"{results['unsponsored_sentiment']:.3f}"),
            ('Engagement Rate', f"{results['sponsored_engagement']:.3f}", f"{results['unsponsored_engagement']:.3f}"),
            ('Videos Analyzed', results['num_sponsored'], results['num_unsponsored']),
            ('Total Views', results['sponsored_views'], results['unsponsored_views']),
            ('Total Likes', results['sponsored_likes'], results['unsponsored_likes']),
            ('Total Comments', results['sponsored_comments'], results['unsponsored_comments']),
            ('Total Positive Comments', results['sponsored_sentiment_counts'].get('positive', 0), results['unsponsored_sentiment_counts'].get('positive', 0)),
            ('Total Neutral Comments', results['sponsored_sentiment_counts'].get('neutral', 0), results['unsponsored_sentiment_counts'].get('neutral', 0)),
            ('Total Negative Comments', results['sponsored_sentiment_counts'].get('negative', 0), results['unsponsored_sentiment_counts'].get('negative', 0)),
            ('Views per Video',
            (round)(results['sponsored_views']/results['num_sponsored']) if results['num_sponsored']>0 else 0,
            (round)(results['unsponsored_views']/results['num_unsponsored']) if results['num_unsponsored']>0 else 0),
            ('Likes per Video',
            (round)(results['sponsored_likes']/results['num_sponsored']) if results['num_sponsored']>0 else 0,
            (round)(results['unsponsored_likes']/results['num_unsponsored']) if results['num_unsponsored']>0 else 0),
            ('Comments per Video',
            (round)(results['sponsored_comments']/results['num_sponsored']) if results['num_sponsored']>0 else 0,
            (round)(results['unsponsored_comments']/results['num_unsponsored']) if results['num_unsponsored']>0 else 0),
            ('Positive Comments per Video',
            (round)(results['sponsored_sentiment_counts'].get('positive', 0) / results['num_sponsored']) if results['num_sponsored']>0 else 0,
            (round)(results['unsponsored_sentiment_counts'].get('positive', 0) / results['num_unsponsored']) if results['num_unsponsored']>0 else 0),
            ('Neutral Comments per Video',
            (round)(results['sponsored_sentiment_counts'].get('neutral', 0) / results['num_sponsored']) if results['num_sponsored']>0 else 0,
            (round)(results['unsponsored_sentiment_counts'].get('neutral', 0) / results['num_unsponsored']) if results['num_unsponsored']>0 else 0),
            ('Negative Comments per Video',
            (round)(results['sponsored_sentiment_counts'].get('negative', 0) / results['num_sponsored']) if results['num_sponsored']>0 else 0,
            (round)(results['unsponsored_sentiment_counts'].get('negative', 0) / results['num_unsponsored']) if results['num_unsponsored']>0 else 0)
        ]

        self.results_table.setRowCount(len(metrics))

        for row, (metric, sponsored, organic) in enumerate(metrics):

            self.results_table.setItem(row, 0, QTableWidgetItem(metric))

            sponsored_item = QTableWidgetItem(str(sponsored))
            sponsored_item.setTextAlignment(Qt.AlignCenter)
            if sponsored > organic:  
                sponsored_item.setBackground(QColor(200, 255, 200))  
            self.results_table.setItem(row, 1, sponsored_item)

            organic_item = QTableWidgetItem(str(organic))
            organic_item.setTextAlignment(Qt.AlignCenter)
            if organic > sponsored:
                organic_item.setBackground(QColor(200, 255, 200))  
            self.results_table.setItem(row, 2, organic_item)
    
    def _get_title_style(self):
        return """
            font-size: 24px;
            font-weight: bold;
            color: #ef2092;
        """

    def _get_subtitle_style(self):
        return """
            font-size: 16px;
            font-weight: bold;
            color: #445464;
        """
    
    def _get_progress_dialog_style(self):
        return """
            QProgressDialog {
                background-color: white;
                border-radius: 10px;
                min-width: 400px;
            }
            QLabel {
                color: #333333;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
            }
        """

    def handle_results(self, results):
        self.progress_dialog.hide()
        self.evaluate_button.setEnabled(True)

        self.update_results_table(results)

    def handle_error(self, error_message):
        self.progress_dialog.hide()
        self.evaluate_button.setEnabled(True)
        self.show_error_message(error_message)

    def update_progress(self, message):
        self.progress_dialog.setLabelText(message)

    def show_error_message(self, message):
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(message)
        error_dialog.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-width: 80px;
            }
        """)
        error_dialog.exec()

class FadeButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._opacity = 1.0  
        self.setStyleSheet(
            "background-color: #ff007f; color: white; font-weight: bold; padding: 8px 15px; border-radius: 5px;"
        )

        self.fade_animation = QPropertyAnimation(self, b"opacity")
        self.fade_animation.setDuration(500)  
        self.fade_animation.setStartValue(1.0)
        self.fade_animation.setEndValue(0.5)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setOpacity(self._opacity)
        painter.setBrush(QBrush(Qt.transparent))
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawRect(self.rect())
        super().paintEvent(event)

    def fade_out(self):
        self.fade_animation.setDirection(QPropertyAnimation.Forward)
        self.fade_animation.start()

    def fade_in(self):
        self.fade_animation.setDirection(QPropertyAnimation.Backward)
        self.fade_animation.start()

    def get_opacity(self):
        return self._opacity

    def set_opacity(self, value):
        self._opacity = value
        self.update()

    opacity = Property(float, get_opacity, set_opacity)

class ResponsiveStylesheet:
    @staticmethod
    def get_dynamic_stylesheet():
        return """
        QMainWindow {
            background-color: #f8f9fa;
        }
        QLabel {
            color: #333;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QLineEdit {
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            background-color: white;
            font-size: 14px;
        }
        QLineEdit:focus {
            border-color: #D47AB3;
        }
        QPushButton {
            background-color: #D47AB3;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px 15px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #7A9E7F;
        }
        QPushButton:disabled {
            background-color: #bdc3c7;
        }
        QTableWidget {
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            gridline-color: #f1f3f5;
        }
        QHeaderView::section {
            background-color: #f1f3f5;
            padding: 8px;
            border: none;
            font-weight: bold;
            color: #2c3e50;
        }
        """

class ScalableFont:
    @staticmethod
    def get_responsive_font(base_size=12, weight=QFont.Weight.Normal):
        font = QFont('Segoe UI', base_size)
        font.setWeight(weight)
        return font

class ResponsiveButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFont(ScalableFont.get_responsive_font(12, QFont.Weight.Bold))

class ResponsiveInput(QLineEdit):
    def __init__(self, placeholder, parent=None):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFont(ScalableFont.get_responsive_font())


def main():
    app = QApplication(sys.argv)
    
    window = YouTubePartnerEstimator()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()