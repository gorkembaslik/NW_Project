# YouTube Partnership Analyzer
A desktop application that analyzes YouTube channels to evaluate and compare sponsored vs. non-sponsored content performance.

## Features
- Analyzes channel content over a specified time period
- Compares sponsored and organic content metrics
- Measures engagement rates and sentiment analysis
- Provides detailed metrics per video type:
  - Views, likes, and comments
  - Sentiment scores
  - Engagement rates
  - Comment sentiment distribution

## Requirements
- Python 3.x
- PySide6
- Google API Python client
- NLTK
- TextBlob
- Other dependencies listed in requirements.txt

## Installation
1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Download NLTK data:
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage
1. Run the application:
```bash
python Foreo_Estimator.py
```
2. Enter the YouTube channel URL
3. Specify the analysis period (months)
4. Set maximum videos to analyze
5. Click "Analyze Channel" to start the evaluation

### P.S. For the desktop app
Run:
pyinstaller --onefile -w --icon=faviconForeo.ico --add-data "emoji.json;." --add-data "Foreo_Logo.png;." Foreo_Estimator.py

- Then the executable file will appear in the dist folder

## Output
The analyzer provides a comprehensive comparison table showing:
- Sentiment scores
- Engagement rates
- Video counts
- View, like, and comment statistics
- Comment sentiment distribution
- Per-video averages for all metrics

## Note
Try to avoid using public Wifi when running the program because googleapiclient may not work.