import praw
import urllib
import cv2
import math
import numpy as np
from enum import Enum
import argparse
import re

SPLIT_REGEX = "(\s,_\)\(\.\n)"

# Eliminate unrelated or useless comments.
# Comments should be:
# 1. An insult
# 2. Not too meta
# 3. Funny
# 4. Applicable to more than one specific person

# These are difficult to quantize, so I'm training a random forest classifier to do it for me.

# I need samples of positive and negative classes.
# If I really have to, I'll break out a sentiment analysis algo. It's a pre-processing step, so I don't need to worry too much about runtime.

class CommentProcessor:
	def __init__(self, ):
		self._classifier = None
		self.comments={}

		cv2.

class Comment:
	def __init__(self, image_name, comment):
		self.image_name = image_name
		self.text = comment.body.lower()
		self.comment = comment
		self.calculate_features()

	def calculate_features(self):
		# words, bigrams (?), length (#tokens), upvotes (percentage of image upvotes), children comment number, amount of punctuation
		# sentiment analysis?

		# Use segment-anything for clustering?

		# tokens
		self.tokens = re.split(SPLIT_REGEX, self.text)

		# bigrams
		# Note: have a look at this data to make sure that bigram-buckets have a decent amount of comments in each
		self.bigrams = []
		for i in range(len(self.tokens)-1):
			self.bigrams.append(f"{self.tokens[i]}{self.tokens[i+1]}")

		# length
		self.length = len(self.text)
		
		# upvotes
		self.upvotes = self.comment.score