import os
HOME_DIR = os.path.expanduser('~') + '/'
PROJECT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_DIR, 'data/')
TAGGED_SENTENCES = os.path.join(PROJECT_DIR, 'train/tagged_sentences.txt')
RAW_SENTENCES = os.path.join(PROJECT_DIR, 'train/raw_sentences.txt')
TAGS = os.path.join(PROJECT_DIR, 'train/tags.txt')
TAGS_PR = os.path.join(PROJECT_DIR, 'train/tags_pr.txt')
WORDS_TAG_PR = os.path.join(PROJECT_DIR, 'words_tag_pr.txt')
LOG_DIR = os.path.join(PROJECT_DIR, 'log/')