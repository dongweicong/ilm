from enum import Enum
import os
import random

from .paths import RAW_DATA_DIR

class Dataset(Enum):
  CUSTOM = 0
  ARXIV_CS_ABSTRACTS = 1
  ROC_STORIES = 2
  ROC_STORIES_NO_TITLE = 3
  LYRICS_STANZAS = 4
  YELP_REVIEWS = 5
  CLOTH_REVIEWS = 6

CLOTH_DIR = os.path.join(RAW_DATA_DIR, 'cloth_review')
def cloth_review(split='train', data_dir=None):
	assert split in ['train', 'valid', 'test']

	if data_dir is None:
	 data_dir = CLOTH_DIR

	reviews=[]

	def compress(f):
		review=[]
		for r in f.read().split('\n\n\n'):
			review.append(r)
		return review

	if split == 'train':
		with open(os.path.join(data_dir, 'cloth_review_train.txt'), 'r') as f:
			reviews=compress(f)
	elif split == 'valid':
		with open(os.path.join(data_dir, 'cloth_review_valid.txt'), 'r') as f:
			reviews=compress(f)
	elif split == 'test':
		with open(os.path.join(data_dir, 'cloth_review_test.txt'), 'r') as f:
			reviews=compress(f)
	else:
		assert False

	return reviews

YELP_DIR = os.path.join(RAW_DATA_DIR, 'yelp_review')
def yelp_review(split='train', data_dir=None,attrs=['id', 'star', 'useful', 'text']):
	assert split in ['train', 'valid', 'test']

	if data_dir is None:
	 data_dir = YELP_DIR

	reviews=[]

	def compress(f):
	  review=[]
	  for r in f.read().split('\n\n\n'):
		try:
		  id, star, useful, text = r.split('\n', 3)
		except:
		  print(r)
		  continue
		a = []
		for attr_name in attrs:
		  a.append(eval(attr_name))
		a = '\n'.join(a)
		review.append(a)
	  return review

	if split == 'train':
		with open(os.path.join(data_dir, 'yelp_review_train.txt'), 'r') as f:
			reviews=compress(f)
	elif split == 'valid':
		with open(os.path.join(data_dir, 'yelp_review_valid.txt'), 'r') as f:
			reviews=compress(f)
	elif split == 'test':
		with open(os.path.join(data_dir, 'yelp_review_test.txt'), 'r') as f:
			reviews=compress(f)
	else:
		assert False

	return reviews

def get_dataset(dataset, split, *args, data_dir=None, shuffle=False, limit=None, **kwargs):
  if type(dataset) != Dataset:
	raise ValueError('Must specify a Dataset enum value')

  if dataset == Dataset.CUSTOM:
	d = custom(split, data_dir)
	if data_dir is None:
	  raise ValueError('Data dir must be specified for custom dataset')
  elif dataset == Dataset.ARXIV_CS_ABSTRACTS:
	d = arxiv_cs_abstracts(split, *args, data_dir=data_dir, **kwargs)
  elif dataset == Dataset.ROC_STORIES:
	d = roc_stories(split, *args, data_dir=data_dir, **kwargs)
  elif dataset == Dataset.ROC_STORIES_NO_TITLE:
	d = roc_stories(split, *args, data_dir=data_dir, with_titles=False, **kwargs)
  elif dataset == Dataset.LYRICS_STANZAS:
	assert split in ['train', 'valid', 'test']
	if data_dir is None:
	  data_dir = os.path.join(RAW_DATA_DIR, 'lyrics_stanzas')
	d = custom(split, data_dir=data_dir)
  elif dataset == Dataset.YELP_REVIEWS:
	d = yelp_review(split, *args, data_dir=data_dir, **kwargs)
  else:
	assert False

  if shuffle:
	random.shuffle(d)

  if limit is not None:
	d = d[:limit]

  return d


def custom(split, data_dir):
  fp = os.path.join(data_dir, '{}.txt'.format(split))
  try:
	with open(fp, 'r') as f:
	  entries = [e.strip() for e in f.read().strip().split('\n\n\n')]
  except:
	raise ValueError('Could not load from {}'.format(fp))
  return entries


ABS_DIR = os.path.join(RAW_DATA_DIR, 'arxiv_cs_abstracts')
def arxiv_cs_abstracts(split='train', data_dir=None, attrs=['title', 'authors', 'categories', 'abstract']):
  assert split in ['train', 'valid', 'test']

  if data_dir is None:
	data_dir = ABS_DIR

  with open(os.path.join(data_dir, 'arxiv_cs_abstracts.txt'), 'r') as f:
	raw = f.read().split('\n\n\n')

  abstracts = []
  for r in raw:
	aid, created, updated, categories, title, authors, abstract = r.split('\n', 6)

	a = []
	for attr_name in attrs:
	  a.append(eval(attr_name))
	a = '\n'.join(a)

	if created.startswith('2018'):
	  if split == 'valid':
		abstracts.append(a)
	elif created.startswith('2019'):
	  if split == 'test':
		abstracts.append(a)
	else:
	  if split == 'train':
		abstracts.append(a)

  return abstracts


ROC_STORIES_DIR = os.path.join(RAW_DATA_DIR, 'roc_stories')
def roc_stories(split='train', data_dir=None, with_titles=True, exclude_nonstandard=True):
  assert split in ['train', 'valid', 'test', 'test_hand_title']

  if data_dir is None:
	data_dir = ROC_STORIES_DIR

  if split == 'train':
	with open(os.path.join(data_dir, 'train_title.txt'), 'r') as f:
	  stories = f.read().split('\n\n\n')
	titled = True
  elif split == 'valid':
	with open(os.path.join(data_dir, 'valid.txt'), 'r') as f:
	  stories = f.read().split('\n\n\n')
	titled = False
  elif split == 'test':
	with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
	  stories = f.read().split('\n\n\n')
	titled = False
  elif split == 'test_hand_title':
	with open(os.path.join(data_dir, 'test_hand_title.txt'), 'r') as f:
	  stories = f.read().split('\n\n\n')
	titled = True

  stories = [s.strip() for s in stories if len(s.strip()) > 0]

  if with_titles != titled:
	if with_titles:
	  stories = ['Unknown Title\n{}'.format(s) for s in stories]
	else:
	  stories = [s.splitlines()[-1] for s in stories]

  if exclude_nonstandard:
	from nltk.tokenize import sent_tokenize

	standardized = []
	for s in stories:
	  paragraphs = s.splitlines()
	  if len(paragraphs) != (2 if with_titles else 1):
		continue
	  try:
		if len(sent_tokenize(paragraphs[-1])) != 5:
		  continue
	  except:
		raise Exception('Need to call nltk.download(\'punkt\')')
	  standardized.append(s)
	stories = standardized

  return stories
