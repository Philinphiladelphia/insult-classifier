import praw
import urllib
import cv2
import math
import numpy as np
from enum import Enum
import argparse

IMAGE_HEIGHT = 200 # pixels
FACE_MARGIN = 40 # pixels

HAAR_CASCADE_MODEL = 'haarcascade_frontalface_default.xml'
DEEP_LEARNING_WEIGHTS = 'face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
DEEP_LEARNING_ARCHITECTURE = 'face_detector/deploy.prototxt'

CONFIDENCE_THRESHOLD = 0.9

MINIMUM_POST_UPVOTES = 100
MINIMUM_COMMENT_PERCENT = 0.1
MAX_COMMENTS_PER_POST = 10

IMAGE_LIMIT = 100

BUILD_CSVS = False

class ClassifierType(Enum):
	HAAR_CASCADE = 0
	DEEP_LEARNING = 1

class FaceClassifier:
	def __init__(self):
		self.face_count = 0

	def classify_image(self, pixels):
		# perform face detection
		(box, box_found) = self.get_first_box_from_image(pixels)

		if not box_found:
			return
		
		x, y, width, height = box
		x2, y2 = x + width, y + height

		cropped_y_min = max(0, int(y)-FACE_MARGIN)
		cropped_y_max = min(pixels.shape[0], int(y2)+FACE_MARGIN)

		cropped_x_min = max(0, int(x)-FACE_MARGIN)
		cropped_x_max = min(pixels.shape[1], int(x2)+FACE_MARGIN)

		pixels = pixels[cropped_y_min:cropped_y_max, cropped_x_min:cropped_x_max]

		scaling_factor = IMAGE_HEIGHT / pixels.shape[0]
		pixels = cv2.resize(pixels, (math.floor(pixels.shape[1]*scaling_factor), IMAGE_HEIGHT))

		cv2.rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
		cv2.imwrite(f"faces/face{self.face_count}.jpg", pixels)
		self.face_count += 1

	def get_reddit_images_and_text(self):
		reddit = praw.Reddit(
			client_id="UO6tHfQYLdjRv4kulBsouA",
			client_secret="TR7PT5CSdg-kgJ6tc5MMSW1VVpajBg",
			user_agent="kappapridekappaross",
		)

		print(reddit.read_only)

		subreddit = reddit.subreddit("RoastMe")
		image_names = []

		# Iterate through top submissions
		for submission in subreddit.hot(limit=IMAGE_LIMIT):

			if submission.score <= MINIMUM_POST_UPVOTES:
				continue

			# Get the link of the submission
			url = str(submission.url)
			print(submission.url)

			# Check if the link is an image
			if url.endswith("jpg") or url.endswith("jpeg") or url.endswith("png"):
				# Retrieve the image and save it in current folder
				image_data = self.url_to_image(url)
				image_data = self.classify_image(image_data)
			else:
				continue

			comment_num = 0
			for top_level_comment in submission.comments:
				comment_num+=1

				if comment_num > MAX_COMMENTS_PER_POST:
					break

				try:
					if top_level_comment.score <= (MINIMUM_POST_UPVOTES*MINIMUM_COMMENT_PERCENT):
						continue

					print(top_level_comment.body)

					if BUILD_CSVS:
						while True:
							letter = input("y/n/s?")

							if letter == 'y':
								file = open("comment_classification_data\good_insult.csv", "a")  # append mode
								file.write(top_level_comment.body + '\n')
								file.close()
								break
							if letter == 'n':
								f = open("comment_classification_data\\bad_insult.csv", "a")  # append mode
								f.write(top_level_comment.body + '\n')
								f.close()
								break
							if letter == 's':
								break
				
				except:
					pass

		return image_names

	def url_to_image(self, url):
		# download the image, convert it to a NumPy array, and then read
		# it into OpenCV format
		resp = urllib.request.urlopen(url)
		image = np.asarray(bytearray(resp.read()), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		# return the image
		return image

class HaarClassifier(FaceClassifier):
	def __init__(self):
		self.classifier = cv2.CascadeClassifier(HAAR_CASCADE_MODEL)
		super().__init__()

	def get_first_box_from_image(self, pixels):
		bboxes = self.classifier.detectMultiScale(pixels)
		if len(bboxes > 0):
			return (bboxes[0], True)
		else:
			return (None, False)

class DeepLearningClassifier(FaceClassifier):
	def __init__(self):
		self.classifier = cv2.dnn.readNetFromCaffe(DEEP_LEARNING_ARCHITECTURE, DEEP_LEARNING_WEIGHTS)
		super().__init__()

	def get_first_box_from_image(self, pixels):
		(h, w) = pixels.shape[:2]
		# TODO careful of the resize
		blob = cv2.dnn.blobFromImage(cv2.resize(pixels, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

		self.classifier.setInput(blob)
		detections = self.classifier.forward()

		confidence = 0
		most_conf_img_index = 0
		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			current_confidence = detections[0, 0, i, 2]
			if current_confidence > confidence:
				confidence = current_confidence
				most_conf_img_index = i

		if confidence > CONFIDENCE_THRESHOLD:
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, most_conf_img_index, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# convert to x, y, w, h
			box = (startX, startY, (endX - startX), (endY - startY))
			return (box, True)

		else:
			return (None, False)

ap = argparse.ArgumentParser()
ap.add_argument("-ha", "--haar", default=0,
	help="Use Haar model")
ap.add_argument("-d", "--deep_learning", default=0,
	help="Use DL model")
args = vars(ap.parse_args())

model_found = False
if args["haar"]:
	fc = HaarClassifier()
	model_found = True
if args["deep_learning"]:
	fc = DeepLearningClassifier()
	model_found = True
if not model_found:
	print("No face detection model found. Re-run with '-ha 1' or'-d 1' options.")
	exit(0)


fc.get_reddit_images_and_text()