import cv2

class VideoCamera(object):
	def __init__(self):
		self.video = cv2.VideoCapture(0)
	def __del__(self):
		self.video.release()
	def get_frame(self):
		success,image = self.video.read()
		ret, jpeg = cv2.imencode('.jpg',image)
		return jpeg.tobytes()
		#
		#
