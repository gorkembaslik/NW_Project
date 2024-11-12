import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QLabel, QLineEdit, QPushButton, QTableWidget, 
                           QTableWidgetItem, QMessageBox, QProgressDialog,
                           QFrame, QHBoxLayout)#, QSizePolicy, QSpacerItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon #, QColor, QPalette, QPixmap
#import qtawesome as qta

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import joblib
#import time
import os
import re
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import requests
from datetime import datetime, timedelta
#from sklearn.feature_extraction.text import TfidfVectorizer 

class EvaluationWorker(QThread):
    finished = pyqtSignal(dict)
    progress = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, channel_url, api_key=''):
        super().__init__()
        self.channel_url = channel_url
        self.api_key = api_key
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.setup_nltk()
        self.load_models()

    def setup_nltk(self):
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.lemmatizer = WordNetLemmatizer()

    def load_models(self):
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        vectorizer_path = os.path.join(base_path, 'models', 'tfidf_vectorizer.pkl')
        model_path = os.path.join(base_path, 'models', 'sentiment_model.pkl')

        try:
            self.vectorizer = joblib.load(vectorizer_path)
            self.model = joblib.load(model_path)
        except Exception as e:
            self.error.emit(f"Error loading models: {str(e)}")

    def run(self):
        try:
            channel_id = self.extract_channel_id(self.channel_url)
            channel_name = self.get_channel_name(channel_id)
            self.progress.emit("Evaluating channel...")
            overall_sponsored_score, overall_unsponsored_score, num_sponsored, num_unsponsored = self.evaluate_channel(channel_id)
            results = {
                'channel_name': channel_name,
                'channel_id': channel_id,
                'sponsored_score': overall_sponsored_score,
                'unsponsored_score': overall_unsponsored_score,
                'num_sponsored': num_sponsored,
                'num_unsponsored': num_unsponsored
            }
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def get_channel_videos(self, channel_id, max_videos=15):
        self.progress.emit("Fetching channel videos...")
        videos = []
        next_page_token = None
        base_video_url = "https://www.youtube.com/watch?v="
        x_months_ago = datetime.now() - timedelta(days=4*30)
        
        while len(videos) < max_videos:
            request = self.youtube.search().list(
                part='snippet',
                channelId=channel_id,
                maxResults=min(10, max_videos - len(videos)),
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
        comment_array = self.vectorizer.transform(comments)
        sentiment_scores = self.model.predict(comment_array)
        sentiment_mapping = {'positive': 1, 'negative': 0}
        numeric_sentiment_scores = np.array([sentiment_mapping.get(score, 0) for score in sentiment_scores], dtype=np.float64)
        return np.mean(numeric_sentiment_scores) if len(numeric_sentiment_scores) > 0 else 0

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

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

        for video in videos:
            self.progress.emit(f"Analyzing video: {video['video_title']}")
            video_id = video['video_id']
            comments = self.get_video_comments(video_id)
            if comments is None:
                continue
            comments = [self.clean_text(comment) for comment in comments]
            likes, views, comments_count = self.get_video_engagement_metrics(video_id)

            if self.check_sponsorship_disclaimer(video_id):
                sponsored_sentiments.append(self.analyze_sentiment(comments))
                sponsored_videos.append(video)
                sponsored_engagement_metrics['likes'] += likes
                sponsored_engagement_metrics['views'] += views
                sponsored_engagement_metrics['comments'] += comments_count
                sponsored_engagement_metrics['count'] += 1
            else:
                unsponsored_sentiments.append(self.analyze_sentiment(comments))
                unsponsored_videos.append(video)
                unsponsored_engagement_metrics['likes'] += likes
                unsponsored_engagement_metrics['views'] += views
                unsponsored_engagement_metrics['comments'] += comments_count
                unsponsored_engagement_metrics['count'] += 1

        num_sponsored = len(sponsored_videos)
        num_unsponsored = len(unsponsored_videos)

        avg_sponsored_sentiment = np.mean(sponsored_sentiments) if sponsored_sentiments else 0
        avg_unsponsored_sentiment = np.mean(unsponsored_sentiments) if unsponsored_sentiments else 0

        avg_sponsored_engagement_score = (sponsored_engagement_metrics['likes'] + sponsored_engagement_metrics['comments']) / (sponsored_engagement_metrics['views'] or 1)
        avg_unsponsored_engagement_score = (unsponsored_engagement_metrics['likes'] + unsponsored_engagement_metrics['comments']) / (unsponsored_engagement_metrics['views'] or 1)

        overall_sponsored_score = (0.5 * avg_sponsored_sentiment) + (0.5 * avg_sponsored_engagement_score)
        overall_unsponsored_score = (0.5 * avg_unsponsored_sentiment) + (0.5 * avg_unsponsored_engagement_score)

        return overall_sponsored_score, overall_unsponsored_score, num_sponsored, num_unsponsored
  

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
        self.setWindowTitle("YouTube Partner Estimator")
        self.setMinimumSize(1000, 700)
        self.set_window_icon()

    def set_window_icon(self):
        """Set window icon with proper path handling"""
        logo_path = os.path.join(getattr(sys, '_MEIPASS', ''), 'Foreo_Logo.png')
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
        self.setup_recommendation_section()
        self.set_layout_stretches()

    def init_central_widget(self):
        """Initialize the central widget and main layout"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(30, 30, 30, 30)

    def setup_header(self):
        """Setup the header section with title and subtitle"""
        header_frame = ModernFrame()
        header_layout = QVBoxLayout(header_frame)
        
        title_label = self.create_label("YouTube Partner Estimator", self._get_title_style())
        subtitle_label = self.create_label("Analyze YouTube channels for partnership potential", self._get_subtitle_style())
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        self.main_layout.addWidget(header_frame)

    def setup_input_section(self):
        """Setup the input section with URL input and evaluate button"""
        input_frame = ModernFrame()
        input_layout = QHBoxLayout(input_frame)
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube Channel URL")
        
        self.evaluate_button = self.create_evaluate_button()
        
        input_layout.addWidget(self.url_input)
        input_layout.addWidget(self.evaluate_button)
        self.main_layout.addWidget(input_frame)

    def setup_results_section(self):
        """Setup the results section with the table"""
        results_frame = ModernFrame()
        results_layout = QVBoxLayout(results_frame)
        
        self.results_table = self.create_results_table()
        results_layout.addWidget(self.results_table)
        self.main_layout.addWidget(results_frame)

    def setup_recommendation_section(self):
        """Setup the recommendation section"""
        recommendation_frame = ModernFrame()
        recommendation_layout = QVBoxLayout(recommendation_frame)
        
        self.recommendation_label = self.create_label("", self._get_recommendation_base_style())
        recommendation_layout.addWidget(self.recommendation_label)
        self.main_layout.addWidget(recommendation_frame)

    def create_evaluate_button(self):
        """Create and configure the evaluate button"""
        button = QPushButton("Evaluate Channel")
        button.setObjectName("evaluateButton")
        button.setFixedWidth(200)
        button.clicked.connect(self.start_evaluation)
        return button

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
        """Validate the channel URL input"""
        if not self.url_input.text():
            self.show_error_message("Please enter a YouTube channel URL")
            return False
        return True

    def _setup_evaluation(self):
        """Setup the evaluation worker and connections"""
        self.evaluate_button.setEnabled(False)
        self.worker = EvaluationWorker(self.url_input.text())
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
        """Update the results table with new data"""
        metrics = [
            ('Channel Name', results['channel_name']),
            ('Channel ID', results['channel_id']),
            ('Sponsored Content Score', f"{results['sponsored_score']:.4f}"),
            ('Organic Content Score', f"{results['unsponsored_score']:.4f}"),
            ('Sponsored Videos Analyzed', str(results['num_sponsored'])),
            ('Organic Videos Analyzed', str(results['num_unsponsored']))
        ]

        self.results_table.setRowCount(len(metrics))
        for i, (metric, value) in enumerate(metrics):
            self.results_table.setItem(i, 0, QTableWidgetItem(metric))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(value)))

    def update_recommendation(self, results):
        """Update the recommendation based on analysis results"""
        is_sponsored = results['sponsored_score'] > results['unsponsored_score']
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

    def _get_title_style(self):
        """Return the style for the title label"""
        return """
            font-size: 32px;
            font-weight: bold;
            color: #1976D2;
            margin: 20px;
        """

    def _get_subtitle_style(self):
        """Return the style for the subtitle label"""
        return """
            font-size: 16px;
            color: #757575;
        """

    def _get_recommendation_base_style(self):
        """Return the base style for the recommendation label"""
        return """
            font-size: 16px;
            padding: 20px;
            border-radius: 6px;
        """

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
                border: 2px solid #2196F3;
            }
            QPushButton#evaluateButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#evaluateButton:hover {
                background-color: #1976D2;
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
        self.update_recommendation(results)

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