import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QLabel, QLineEdit, QPushButton, QTableWidget, 
                           QTableWidgetItem, QMessageBox, QProgressDialog,
                           QFrame, QHBoxLayout, QHeaderView)#, QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QColor#, QPalette, QPixmap
#import qtawesome as qta

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
#import joblib
#import time
import os
import re
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import requests
from datetime import datetime, timedelta

import emoji
#import unicodedata
from nltk.corpus import stopwords
from langdetect import detect, LangDetectException
import isodate

class EnhancedSentimentAnalyzer:
    def __init__(self):
        # Download necessary NLTK resources
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Prepare a set of short, uninformative phrases
        self.short_phrases = {
            'good', 'nice', 'great', 'awesome', 'wow', 'cool', 'amazing', 
            'helpful', 'excellent', 'superb', 'wonderful', 'fantastic', 
            'very helpful', 'really good', 'really nice', 'really helpful',
            'very nice', 'very good', 'very impressive', 'really impressive'
        }
        
        # Prepare English stopwords
        self.stop_words = set(stopwords.words('english'))

    def remove_emojis_and_symbols(self, text):
        """
        Remove emojis, symbols, and pictographs
        Uses the emoji library to comprehensively remove emojis
        """
        # Remove emojis
        text = emoji.replace_emoji(text, replace='')
        
        # Remove transport and map symbols and other special symbols
        text = re.sub(r'[\U0001F300-\U0001F5FF\U0001F900-\U0001F9FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', '', text)
        
        return text
    
    def normalize_text(self, text):

        # Convert to lowercase
        text = text.lower()
        
        # Remove emojis and symbols
        text = self.remove_emojis_and_symbols(text)
        
        # Remove punctuation and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def filter_comment(self, comment):
        """
        Comprehensive comment filtering process
        """
        try:
            # Normalize text
            normalized_text = self.normalize_text(comment)
            
            # Detect language
            try:
                language = detect(normalized_text)
                if language != 'en':
                    return None
            except LangDetectException:
                # If language detection fails, we'll skip the comment
                return None
            
            # Remove stop words
            words = normalized_text.split()
            words = [word for word in words if word not in self.stop_words]
            
            # Check comment length and against short phrases
            filtered_text = ' '.join(words)
            
            # Remove very short comments or known uninformative phrases
            if (len(filtered_text.split()) < 3 or 
                filtered_text.strip() in self.short_phrases or 
                len(filtered_text.strip()) < 10):
                return None
            
            return filtered_text
        
        except Exception as e:
            print(f"Error filtering comment: {e}")
            return None
        
    def analyze_sentiment(self, comments):
        """
        Enhanced sentiment analysis with comprehensive filtering
        """
        if not comments:
            return 0, {'positive': 0, 'neutral': 0, 'negative': 0}
        
        # Filter comments
        filtered_comments = [self.filter_comment(comment) for comment in comments]
        filtered_comments = [comment for comment in filtered_comments if comment is not None]
        
        if not filtered_comments:
            return 0, {'positive': 0, 'neutral': 0, 'negative': 0}
        
        sentiment_scores = []
        sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for comment in filtered_comments:
            # VADER Sentiment Analysis
            vader_scores = self.vader_analyzer.polarity_scores(comment)
            vader_sentiment = vader_scores['compound']
            
            # TextBlob Sentiment Analysis
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
        
        # If no valid comments after filtering
        if not sentiment_scores:
            return 0, sentiment_counts
        
        # median to reduce impact of extreme values
        median_sentiment = np.median(sentiment_scores)
        
        # Normalize sentiment to 0-1 scale
        normalized_sentiment = (median_sentiment + 1) / 2
        
        return normalized_sentiment, sentiment_counts

class EvaluationWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, channel_url, api_key='', months=6, max_videos=15):
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
            channel_name = self.get_channel_name(channel_id)
            self.progress.emit("Evaluating channel...")
            
            # Unpack all returned values
            (avg_sponsored_sentiment, avg_unsponsored_sentiment, 
            avg_sponsored_engagement_score, avg_unsponsored_engagement_score, 
            num_sponsored, num_unsponsored,
            sponsored_sentiment_counts, unsponsored_sentiment_counts) = self.evaluate_channel(channel_id)
            
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
                'unsponsored_sentiment_counts': unsponsored_sentiment_counts
            }
            
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

                    # Get the publish date
                    publish_date_str = video_details['snippet']['publishedAt']
                    publish_date = datetime.strptime(publish_date_str, "%Y-%m-%dT%H:%M:%SZ")

                    # Filter videos based on the date (only include videos from the last 6 months)
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
                # Add counts to sponsored totals
                for key in sentiment_counts:
                    sponsored_sentiment_counts[key] += sentiment_counts[key]
            else:
                unsponsored_sentiments.append(sentiment_score)
                unsponsored_videos.append(video)
                unsponsored_engagement_metrics['likes'] += likes
                unsponsored_engagement_metrics['views'] += views
                unsponsored_engagement_metrics['comments'] += comments_count
                unsponsored_engagement_metrics['count'] += 1
                # Add counts to unsponsored totals
                for key in sentiment_counts:
                    unsponsored_sentiment_counts[key] += sentiment_counts[key]

        num_sponsored = len(sponsored_videos)
        num_unsponsored = len(unsponsored_videos)

        avg_sponsored_sentiment = np.mean(sponsored_sentiments) if sponsored_sentiments else 0
        avg_unsponsored_sentiment = np.mean(unsponsored_sentiments) if unsponsored_sentiments else 0

        avg_sponsored_engagement_score = (sponsored_engagement_metrics['likes'] + sponsored_engagement_metrics['comments']) / (sponsored_engagement_metrics['views'] or 1)
        avg_unsponsored_engagement_score = (unsponsored_engagement_metrics['likes'] + unsponsored_engagement_metrics['comments']) / (unsponsored_engagement_metrics['views'] or 1)

        results = {
            'sponsored_sentiment': avg_sponsored_sentiment,
            'unsponsored_sentiment': avg_unsponsored_sentiment,
            'sponsored_engagement': avg_sponsored_engagement_score,
            'unsponsored_engagement': avg_unsponsored_engagement_score,
            'num_sponsored': num_sponsored,
            'num_unsponsored': num_unsponsored,
            'sponsored_sentiment_counts': sponsored_sentiment_counts,
            'unsponsored_sentiment_counts': unsponsored_sentiment_counts
        }

        return (avg_sponsored_sentiment, avg_unsponsored_sentiment, 
                avg_sponsored_engagement_score, avg_unsponsored_engagement_score, 
                num_sponsored, num_unsponsored,
                sponsored_sentiment_counts, unsponsored_sentiment_counts)
  

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
        self.init_window()
        self.init_styling()
        self.setup_ui()

    def init_window(self):
        """Initialize window properties and icon"""
        self.setWindowTitle("YouTube Partnership Estimator")
        self.setMinimumSize(1000, 700)
        self.set_window_icon()

    def set_window_icon(self):
        """Set window icon with proper path handling"""
        logo_path = os.path.join(getattr(sys, '_MEIPASS', ''), 'ForeoNewLogo.png')
        self.setWindowIcon(QIcon(logo_path))

    def init_styling(self):
        """Initialize application styling using a separate stylesheet"""
        self.setStyleSheet(self._get_stylesheet())

    def setup_ui(self):
        """Initialize and setup the UI components"""
        self.init_central_widget()
        self.setup_header()
        self.setup_input_section()
        self.setup_results_section()
        #self.setup_recommendation_section()
        self.set_layout_stretches()

    def init_central_widget(self):
        """Initialize the central widget and main layout"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10)
        self.main_layout.setContentsMargins(20, 20, 20, 20)

    def setup_header(self):
        """Setup the header section with title and subtitle"""
        header_frame = ModernFrame()
        #header_layout = QVBoxLayout(header_frame)
        header_layout = QHBoxLayout(header_frame)

        title_label = self.create_label("YouTube Partnership Estimator", self._get_title_style())
        subtitle_label = self.create_label("Analyze YouTube channel for potential partnership", self._get_subtitle_style())

        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        self.main_layout.addWidget(header_frame)

    def setup_input_section(self):
        """Setup the input section with URL input, months input, max videos input, and evaluate button"""
        input_frame = ModernFrame()
        input_layout = QHBoxLayout(input_frame)

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube Channel URL")

        self.months_input = QLineEdit()
        self.months_input.setPlaceholderText("Number of months (e.g., 6)")

        self.max_videos_input = QLineEdit()
        self.max_videos_input.setPlaceholderText("Maximum number of videos (e.g., 20)")

        self.evaluate_button = self.create_evaluate_button()

        input_layout.addWidget(self.url_input)
        input_layout.addWidget(self.months_input)
        input_layout.addWidget(self.max_videos_input)
        input_layout.addWidget(self.evaluate_button)
        self.main_layout.addWidget(input_frame)

    '''
    def setup_results_section(self):
        """Setup the results section with a comparison table"""
        results_frame = ModernFrame()
        results_layout = QVBoxLayout(results_frame)

        # Add a title for the results section
        results_title = self.create_label("Comparison of Metrics", self._get_title_style())
        results_layout.addWidget(results_title)

        # Create the results table
        self.results_table = self.create_comparison_table()
        results_layout.addWidget(self.results_table)

        self.main_layout.addWidget(results_frame)
    '''
    def setup_results_section(self):
        """Setup the results section with a three-column layout"""
        results_frame = ModernFrame()
        results_layout = QVBoxLayout(results_frame)

        # Add section title
        title_label = self.create_label("Comparison of Sponsored vs. Organic Metrics", self._get_section_title_style())
        results_layout.addWidget(title_label)

        # Create and configure the table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(3)
        self.results_table.setHorizontalHeaderLabels(['Metric', 'Sponsored Content', 'Organic Content'])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setStyleSheet(self._get_table_style())
        self.results_table.setFrameStyle(QFrame.NoFrame)
        #self.results_table.setAttribute(Qt.WA_TranslucentBackground)

        results_layout.addWidget(self.results_table)
        self.main_layout.addWidget(results_frame)

    def _get_section_title_style(self):
        """Style for the results section title"""
        return """
            font-size: 24px;
            font-weight: bold;
            color: #445464; 
            text-align: center;
            margin-bottom: 20px;
        """

    def _get_table_style(self):
        return """
            QTableWidget {
                background-color: #ffffff;
                border: 3px solid #ef2092;
                border-radius: 12px;
                gridline-color: #ef2092;
            }
            QTableWidget::item {
                padding: 10px;
                text-align: center;
                color: #333333;
            }
            QTableWidget::item:selected {
                background-color: #fde4ef;  /* Subtle pink for selection */
                color: #ef2092;           /* High contrast pink text */
                font-weight: bold;
            }
            QHeaderView {
                background-color: #ef2092;
                border: none;
            }
            QHeaderView::section {
                background-color: #ef2092; /* Pink header background */
                color: #ffffff;
                font-weight: bold;
                border: none;
                padding: 5px; /* Balanced padding */
                margin: 0px; /* Remove gaps */
                border-radius: 0px; /* Prevent separate radii causing gaps */
            }
            QHeaderView::section:first {
                border-top-left-radius: 12px; /* Align with table */
            }
            QHeaderView::section:last {
                border-top-right-radius: 12px; /* Align with table */
            }
        """

    '''
    def setup_recommendation_section(self):
        """Setup the recommendation section"""
        recommendation_frame = ModernFrame()
        recommendation_layout = QVBoxLayout(recommendation_frame)

        self.recommendation_label = self.create_label("", self._get_recommendation_base_style())
        recommendation_layout.addWidget(self.recommendation_label)
        self.main_layout.addWidget(recommendation_frame)
    '''

    def create_evaluate_button(self):
        """Create and configure the evaluate button"""
        button = QPushButton("Evaluate Channel")
        button.setObjectName("evaluateButton")
        button.setFixedWidth(200)
        button.clicked.connect(self.start_evaluation)
        return button
    '''
    def create_results_table(self):
        """Create and configure the results table"""
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(['Metric', 'Value'])
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        return table

    def create_comparison_table(self):
        """Create and configure the comparison table"""
        table = QTableWidget()
        table.setColumnCount(3)  # Metric, Sponsored, Organic
        table.setHorizontalHeaderLabels(['Metric', 'Sponsored Content', 'Organic Content'])

        # Style header for better appearance
        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Metric column
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Sponsored
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Organic

        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(True)
        return table
    '''
    def create_label(self, text, style):
        """Create a label with specified text and style"""
        label = QLabel(text)
        label.setStyleSheet(style)
        label.setAlignment(Qt.AlignCenter)
        return label

    def set_layout_stretches(self):
        """Set the stretch factors for the main layout"""
        stretches = [0, 0, 1, 0]  # Header, Input, Results, Recommendation
        for index, stretch in enumerate(stretches):
            self.main_layout.setStretch(index, stretch)

    def start_evaluation(self):
        """Start the channel evaluation process"""
        if not self._validate_input():
            return

        self._setup_evaluation()
        self._show_progress_dialog()

    def _validate_input(self):
        """Validate the channel URL, months, and max videos input"""
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
        """Setup the evaluation worker and connections"""
        self.evaluate_button.setEnabled(False)
        months = int(self.months_input.text())
        max_videos = int(self.max_videos_input.text())
        self.worker = EvaluationWorker(self.url_input.text(), months=months, max_videos=max_videos)
        self._connect_worker_signals()
        self.worker.start()

    def _connect_worker_signals(self):
        """Connect worker signals to their respective slots"""
        self.worker.finished.connect(self.handle_results)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.handle_error)

    def _show_progress_dialog(self):
        """Show and configure the progress dialog"""
        self.progress_dialog = QProgressDialog("Evaluating channel...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Analysis in Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setStyleSheet(self._get_progress_dialog_style())
        self.progress_dialog.show()

    def update_results_table(self, results):
        """Update the results table with a comparison layout"""
        metrics = [
            ('Sentiment Score', f"{results['sponsored_sentiment']:.3f}", f"{results['unsponsored_sentiment']:.3f}"),
            ('Engagement Rate', f"{results['sponsored_engagement']:.3f}", f"{results['unsponsored_engagement']:.3f}"),
            ('Videos Analyzed', results['num_sponsored'], results['num_unsponsored']),
            ('Positive Comments', results['sponsored_sentiment_counts'].get('positive', 0), results['unsponsored_sentiment_counts'].get('positive', 0)),
            ('Neutral Comments', results['sponsored_sentiment_counts'].get('neutral', 0), results['unsponsored_sentiment_counts'].get('neutral', 0)),
            ('Negative Comments', results['sponsored_sentiment_counts'].get('negative', 0), results['unsponsored_sentiment_counts'].get('negative', 0)),
            ('Total Comments per Video', (int)((results['sponsored_sentiment_counts'].get('positive', 0)+results['sponsored_sentiment_counts'].get('neutral', 0)+results['sponsored_sentiment_counts'].get('negative', 0))/results['num_sponsored']), (int)((results['unsponsored_sentiment_counts'].get('positive', 0)+results['unsponsored_sentiment_counts'].get('negative', 0)+results['unsponsored_sentiment_counts'].get('neutral', 0))/results['num_unsponsored'])),
            ('Positive Comments per Video',
            (int)(results['sponsored_sentiment_counts'].get('positive', 0) / results['num_sponsored']),
            (int)(results['unsponsored_sentiment_counts'].get('positive', 0) / results['num_unsponsored'])),
            ('Neutral Comments per Video',
            (int)(results['sponsored_sentiment_counts'].get('neutral', 0) / results['num_sponsored']),
            (int)(results['unsponsored_sentiment_counts'].get('neutral', 0) / results['num_unsponsored'])),
            ('Negative Comments per Video',
            (int)(results['sponsored_sentiment_counts'].get('negative', 0) / results['num_sponsored']),
            (int)(results['unsponsored_sentiment_counts'].get('negative', 0) / results['num_unsponsored']))
        ]

        self.results_table.setRowCount(len(metrics))

        for row, (metric, sponsored, organic) in enumerate(metrics):
            # Add metric name
            self.results_table.setItem(row, 0, QTableWidgetItem(metric))

            # Add sponsored value with conditional formatting
            sponsored_item = QTableWidgetItem(str(sponsored))
            sponsored_item.setTextAlignment(Qt.AlignCenter)
            if sponsored > organic:  # Highlight higher values
                sponsored_item.setBackground(QColor(200, 255, 200))  # Light green
            self.results_table.setItem(row, 1, sponsored_item)

            # Add organic value with conditional formatting
            organic_item = QTableWidgetItem(str(organic))
            organic_item.setTextAlignment(Qt.AlignCenter)
            if organic > sponsored:
                organic_item.setBackground(QColor(200, 255, 200))  # Light green
            self.results_table.setItem(row, 2, organic_item)
    '''
    def update_recommendation(self, results):
        """Update the recommendation based on analysis results"""
        is_sponsored = results['sponsored_sentiment'] >= results['unsponsored_sentiment'] and results['sponsored_engagement'] >= results['unsponsored_engagement']
        text, style = self._get_recommendation_content(is_sponsored)
        self.recommendation_label.setText(text)
        self.recommendation_label.setStyleSheet(self._get_recommendation_base_style() + style)

    @staticmethod
    def _get_recommendation_content(is_sponsored):
        """Get recommendation text and style based on sponsorship potential"""
        if is_sponsored:
            return ("✨ This channel shows strong potential for sponsored partnerships! ✨",
                   """
                   background-color: #E8F5E9;
                   color: #2E7D32;
                   font-weight: bold;
                   """)
        return ("⚠️ This channel may need improvement before pursuing sponsored partnerships",
                """
                background-color: #FFEBEE;
                color: #C62828;
                font-weight: bold;
                """)
    '''

    def _get_title_style(self):
        """Return the style for the title label"""
        return """
            font-size: 32px;
            font-weight: bold;
            color: #ef2092;
            margin: 20px;
        """

    def _get_subtitle_style(self):
        """Return the style for the subtitle label"""
        return """
            font-size: 20px;
            font-weight: bold;
            color: #445464;
        """
    '''
    def _get_recommendation_base_style(self):
        """Return the base style for the recommendation label"""
        return """
            font-size: 16px;
            padding: 20px;
            border-radius: 6px;
        """
    '''
    def _get_progress_dialog_style(self):
        """Return the style for the progress dialog"""
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

    @staticmethod
    def _get_stylesheet():
        """Return the application stylesheet"""
        return """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QLabel {
                color: #333333;
            }
            QLineEdit {
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 6px;
                background-color: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 2px solid #ef2092;
            }
            QPushButton#evaluateButton {
                background-color: #fa6dbb;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#evaluateButton:hover {
                background-color: #fa6dbb;
            }
            QPushButton#evaluateButton:disabled {
                background-color: #BDBDBD;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                gridline-color: #f5f5f5;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
            QProgressDialog {
                background-color: white;
                border-radius: 6px;
            }
            QProgressDialog QLabel {
                color: #333333;
                font-size: 14px;
            }
        """

    def handle_results(self, results):
        self.progress_dialog.hide()
        self.evaluate_button.setEnabled(True)

        self.update_results_table(results)
        #self.update_recommendation(results)

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
        error_dialog.exec_()

def main():
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = YouTubePartnerEstimator()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()