import sys, os, nltk, pickle, argparse, gzip, csv, json, torch, numpy as np, torch.nn as nn
from collections import defaultdict

sys.path.append('..')
from utils import Logger, get_logfiles, tokenize, architect_prefix, builder_prefix, type2id, initialize_rngs, write_commit_hashes

class Vocabulary(object):
	"""Simple vocabulary wrapper."""
	def __init__(self, data_path='../data/logs/', vector_filename=None, embed_size=300, use_speaker_tokens=False, use_builder_action_tokens=False, add_words=True, lower=False, threshold=0, all_splits=False, add_builder_utterances=False, builder_utterances_only=False):
		"""
		Args:
			data_path (string): path to CwC official data directory.
			vector_filename (string, optional): path to pretrained embeddings file.
			embed_size (int, optional): size of word embeddings.
			use_speaker_tokens (boolean, optional): use speaker tokens <Architect> </Architect> and <Builder> </Builder> instead of sentence tokens <s> and </s>
			use_builder_action_tokens (boolean, optional): use builder action tokens for pickup/putdown actions, e.g. <builder_pickup_red> and <builder_putdown_red>
			add_words (boolean, optional): whether or not to add OOV words to vocabulary as random vectors. If not, all OOV tokens are treated as unk.
			lower (boolean, optional): whether or not to lowercase all tokens.
			keep_all_embeddings (boolean, optional): whether or not to keep embeddings in pretrained files for all words (even those out-of-domain). Signific