from head_detector import HeadDetector
import os
import cv2

if __name__ == '__main__':
	
	# Initialize detector
	head_detector = HeadDetector()
	# prepare a img object
	img_path = os.path.join('/home/tangwang/test/test1/', '0000001.jpg')
	img = cv2.imread(img_path)
	# call function to detect
	detect_list = head_detector.detect_img(img)
	# print detection result
	for detect_item in detect_list:
		print(detect_item)



