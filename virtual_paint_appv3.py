import cv2
import time
import os
import mediapipe as mp
import numpy as np

CurrentTime = 0
PreviousTime = 0

Capture = cv2.VideoCapture(0)
Capture.set(3, 1280)
Capture.set(4, 720)
Capture.set(cv2.CAP_PROP_FPS, 60)

DrawColor = (255, 200, 100)
Board = np.zeros((900, 1280, 3), np.uint8)

# detector = TH.HandDetector(min_detection_confidence=.85)
hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=1)
draw = mp.solutions.drawing_utils

#contants
ml = 150
max_x, max_y = 250+ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
x = 0
y = 0
xp, yp = 0, 0
# prevx, prevy = 0,0

# drawing tools
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

#get tools function
def getTool(x):
	if x < 50 + ml:
		return "line"

	elif x<100 + ml:
		return "rectangle"

	elif x < 150 + ml:
		return"draw"

	elif x<200 + ml:
		return "circle"

	else:
		return "erase"

def index_down(yi, y9):
	if (y9 - yi) > 40:
		return False

	return True
    


while True:
	ret, frame = Capture.read()
	frame = cv2.flip(frame,1)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame[0:125, 0:1280] = BrushBoard
	op = hand_landmark.process(rgb)

	if op.multi_hand_landmarks:
		for i in op.multi_hand_landmarks:
			draw.draw_landmarks(frame, i, hands.HAND_CONNECTIONS)
			x, y = int(i.landmark[8].x*1280), int(i.landmark[8].y*720)

			if x < max_x and y < max_y and x > ml:
				if time_init:
					ctime = time.time()
					time_init = False
				ptime = time.time()

				cv2.circle(frame, (x, y), rad, (0,255,255), 2)
				rad -= 1
				if (ptime - ctime) > 0.8:
					curr_tool = getTool(x)
					print("your current tool set to : ", curr_tool)
					time_init = True
					rad = 40
			else:
				time_init = True
				rad = 40

			if curr_tool == "draw":
				xi, yi = int(i.landmark[12].x*1280), int(i.landmark[12].y*720)
				y9  = int(i.landmark[9].y*720)

				if index_down(yi, y9):
					cv2.line(Board, (xp, yp), (x, y), color=DrawColor,thickness=thick)
					xp, yp = x, y

				else:
					xp = x
					yp = y



			elif curr_tool == "line":
				xi, yi = int(i.landmark[12].x*1280), int(i.landmark[12].y*720)
				y9  = int(i.landmark[9].y*720)

				if index_down(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.line(frame, (xii, yii), (x, y), (50,152,255), thick)

				else:
					if var_inits:
						cv2.line(Board, (xp, yp), (x, y), color=DrawColor,thickness=thick)
						var_inits = False

			elif curr_tool == "rectangle":
				xi, yi = int(i.landmark[12].x*1280), int(i.landmark[12].y*1280)
				y9  = int(i.landmark[9].y*480)

				if index_down(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.rectangle(frame, (xii, yii), (x, y), (0,255,255), thick)

				else:
					if var_inits:
						cv2.rectangle(Board, (xii, yii), (x, y), color=DrawColor, thickness = thick)
                        
                        #cv2.rectanglee(Board, (xii, yii), (x, y), color=DrawColor,thickness=thick)
						var_inits = False

			elif curr_tool == "circle":
				xi, yi = int(i.landmark[12].x*1280), int(i.landmark[12].y*720)
				y9  = int(i.landmark[9].y*720)

				if index_down(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.circle(frame, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick)

				else:
					if var_inits:
						#cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), thick)
						cv2.circle(Board, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), color=DrawColor, thickness=thick)
						var_inits = False

			elif curr_tool == "erase":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_down(yi, y9):
					cv2.circle(frame, (x, y), 30, (0,0,0), -1)
					cv2.circle(Board, (x, y), 30, 255, -1)


	# op = cv2.bitwise_and(frame, Board)
	# frame[:, :, 1] = op[:, :, 1]
	# frame[:, :, 2] = op[:, :, 2]
	BoardGray = cv2.cvtColor(Board, cv2.COLOR_BGR2GRAY)
	_, BoardLines = cv2.threshold(BoardGray, 50, 255, cv2.THRESH_BINARY_INV)
	BoardLines = cv2.cvtColor(BoardLines, cv2.COLOR_GRAY2BGR)
	frame = cv2.bitwise_and(frame, BoardLines)
	frame = cv2.bitwise_or(frame, Board)

	frame[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frame[:max_y, ml:max_x], 0.3, 0)

	cv2.putText(frame, curr_tool, (270+ml,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("VirtualPaint MiniProject", frame)

	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		Capture.release()
		break
 