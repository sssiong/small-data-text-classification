from pathlib import Path

PROJECT_PATH = Path(__file__).parent.absolute()

DATA_PATH = PROJECT_PATH / 'data'

DATASOURCES = {
    'goemotions': {
        'filepath': DATA_PATH / 'goemotions' / 'data' / 'full_dataset' / 'goemotions_1.csv',
        'textcol': 'text',
        'labelcols': ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'],
    },
    'just_dance': {
        'filepath': DATA_PATH / 'just_dance' / 'jd-multi-label-dataset.csv',
        'textcol': 'originalText',
        'labelcols': ['Usability', 'UX', 'H-QOL', 'Memorability', 'Learnability', 'Efficiency', 'Errors/Effectiveness', 'Satisfaction', 'Aesthetics and Appeal', 'Affect and Emotion', 'Anticipation', 'Comfort', 'Detailed Usability', 'Enchantment', 'Engagement', 'Enjoyment and Fun', 'Frustration', 'Hedonic', 'Impact', 'Likeability', 'Motivation', 'Overall Usability', 'Pleasure', 'Support', 'Trust', 'User Differences', 'Bodily image and Appearance', 'Concentration', 'Energy', 'Fatigue', 'Learning', 'Memory', 'Negative feelings', 'Pain and Discomfort', 'Personal relationships', 'Positive feelings', 'Self-esteem', 'Sexual activity', 'Sleep and Rest', 'Social support', 'Thinking'],
    },
    'pubmed': {
        'filepath': DATA_PATH / 'pubmed' / 'PubMed Multi Label Text Classification Dataset Processed.csv',
        'textcol': 'abstractText',
        'labelcols': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z'],
    },
    'research_papers': {
        'filepath': DATA_PATH / 'research_papers' / 'train.csv',
        'textcol': 'ABSTRACT',
        'labelcols': ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance'],
    },
}