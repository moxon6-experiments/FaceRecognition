import cv2

cam = cv2.VideoCapture(0)

while True:
	ret, frame = cam.read()
	if ret:
		cv2.imshow("Window", frame)
	ch = cv2.waitKey(1)
	if ch == 27:
		break
	