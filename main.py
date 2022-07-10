import cv2
import mediapipe as mp
 
FRAME_WIDTH  = 640
FRAME_HEIGHT =  480
 
THUMB_TIP = 4
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
 
Z_THRESHOLD_PRESS = -80
 
VK = {
    'Q': { 'x':50,  'y':100, 'w':50, 'h':50 },
    'W': { 'x':100, 'y':100, 'w':50, 'h':50 },
    'E': { 'x':150, 'y':100, 'w':50, 'h':50 },
    'R': { 'x':200, 'y':100, 'w':50, 'h':50 },
    'T': { 'x':250, 'y':100, 'w':50, 'h':50 },
    'Y': { 'x':300, 'y':100, 'w':50, 'h':50 },
    'U': { 'x':350, 'y':100, 'w':50, 'h':50 },
    'I': { 'x':400, 'y':100, 'w':50, 'h':50 },
    'O': { 'x':450, 'y':100, 'w':50, 'h':50 },
    'P': { 'x':500, 'y':100, 'w':50, 'h':50 },
    
    
    'A': { 'x':50, 'y':200, 'w':50, 'h':50 },
    'S': { 'x':100, 'y':200, 'w':50, 'h':50 },
    'D': { 'x':150, 'y':200, 'w':50, 'h':50 },
    'F': { 'x':200, 'y':200, 'w':50, 'h':50 },
    'G': { 'x':250, 'y':200, 'w':50, 'h':50 },
    'H': { 'x':300, 'y':200, 'w':50, 'h':50 },
    'J': { 'x':350, 'y':200, 'w':50, 'h':50 },
    'K': { 'x':400, 'y':200, 'w':50, 'h':50 },
    'L': { 'x':450, 'y':200, 'w':50, 'h':50 },
    
    
    'Z': { 'x':50, 'y':300, 'w':50, 'h':50 },
    'X': { 'x':100, 'y':300, 'w':50, 'h':50 },
    'C': { 'x':150, 'y':300, 'w':50, 'h':50 },
    'V': { 'x':200, 'y':300, 'w':50, 'h':50 },
    'B': { 'x':250, 'y':300, 'w':50, 'h':50 },
    'N': { 'x':300, 'y':300, 'w':50, 'h':50 },
    'M': { 'x':3500, 'y':300, 'w':50, 'h':50 },

}
 
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
 
def draw_keys(img, x, y, z):
    for k in VK:
        if ((VK[k]['x'] < x < VK[k]['x']+VK[k]['w']) and (VK[k]['y'] < y < VK[k]['y']+VK[k]['h']) and (z <= Z_THRESHOLD_PRESS)):
            cv2.rectangle(img, (VK[k]['x'], VK[k]['y']), (VK[k]['x']+VK[k]['w'], VK[k]['y']+VK[k]['h']), (0,0,255), -1) # thickness -1 means filled rectangle
            cv2.putText(img, f"{k}", (VK[k]['x']+30,VK[k]['y']+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (VK[k]['x'], VK[k]['y']), (VK[k]['x']+VK[k]['w'], VK[k]['y']+VK[k]['h']), (0,255,0), 1) # thickness -1 means filled rectangle
            cv2.putText(img, f"{k}", (VK[k]['x']+30,VK[k]['y']+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
 
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
 
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
 
        x = 0
        y = 0
        z = 0
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # DEBUG
                try:
                    index_finger_tip = handLms.landmark[INDEX_FINGER_TIP]
                    x = int(index_finger_tip.x * FRAME_WIDTH)
                    y = int(index_finger_tip.y * FRAME_HEIGHT)
                    z = int(index_finger_tip.z * FRAME_WIDTH)
                    #print(f"x={x} , y={y} , z={z}")
                    if (z <= Z_THRESHOLD_PRESS):
                        color = (0,0,255) # BGR
                    else:
                        color = (0,255,0)
                    cv2.putText(img, f"{x}, {y}, {z}", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                except IndexError:
                    index_finger_tip = None
 
        draw_keys(img, x, y, z)
 
        cv2.imshow("OpenCV Video Capture", img)
 
        if cv2.waitKey(1) & 0xFF == 27: # ESC
            break
 
if __name__ == "__main__":
    main()
 