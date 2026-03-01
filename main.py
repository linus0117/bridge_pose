import cv2
import mediapipe as mp
import numpy as np
import utils  

# 設定 MediaPipe 
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. 開啟攝影機
cap = cv2.VideoCapture(0)

print("程式啟動中...按 'q' 鍵可以結束程式")

while True:
    success, img = cap.read()
    if not success:
        break

    # 取得畫面的寬度和高度 (轉換座標要用)
    h, w, c = img.shape

    # 轉成 RGB 給 MediaPipe 吃
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    # 3. 處理骨架數據 (Week 2 & 3 重點)
    if results.pose_landmarks:
        
        # 取得所有關鍵點的清單
        landmarks = results.pose_landmarks.landmark

        # --- 鎖定關鍵點  ---
        # 測側面，要根據人是朝左還是朝右，選擇 11,23,25 (左側) 或 12,24,26 (右側)
        
        #  抓取左邊原始座標(11,23,25)
        shoulder_lm = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip_lm      = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]     
        knee_lm     = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]    
        ankle_lm    = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
  
        #抓取右邊三個原始座標(12,24,26)
        r_shoulder_lm = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_hip_lm      = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee_lm     = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle_lm    = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        points = None #一個空箱子裝座標
        
        if (shoulder_lm.visibility  > 0.5 and 
                hip_lm.visibility   > 0.5 and 
                knee_lm.visibility  > 0.5 and
                ankle_lm.visibility > 0.5):
            #計算出左邊座標，放入箱子
            p1 = [int(shoulder_lm.x * w),   int(shoulder_lm.y * h)]
            p2 = [int(hip_lm.x * w),        int(hip_lm.y * h)]
            p3 = [int(knee_lm.x * w),       int(knee_lm.y * h)]
            p4 = [int(ankle_lm.x * w),      int(ankle_lm.y * h)]
            
            points = [p1, p2, p3, p4]
            
        elif (r_shoulder_lm.visibility  > 0.5 and 
              r_hip_lm.visibility       > 0.5 and 
              r_knee_lm.visibility      > 0.5 and
              r_ankle_lm.visibility     > 0.5):
            #算出右邊座標放入箱子
            p1 = [int(r_shoulder_lm.x * w), int(r_shoulder_lm.y * h)]
            p2 = [int(r_hip_lm.x * w),      int(r_hip_lm.y * h)]
            p3 = [int(r_knee_lm.x * w),     int(r_knee_lm.y * h)]
            p4 = [int(r_ankle_lm.x * w),    int(r_ankle_lm.y * h)]
            
            points = [p1, p2, p3, p4]
            
        #抓到方向後開始畫圖
        if points:
            shoulder = points[0]
            hip      = points[1]
            knee     = points[2]
            ankle    = points[3]
            #畫點
            cv2.circle(img, (shoulder[0], shoulder[1]),   10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (hip[0], hip[1]),             10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (knee[0], knee[1]),           10, (0, 255,255) , cv2.FILLED)
            cv2.circle(img, (ankle[0],ankle[1]),          10, (0, 255,255) , cv2.FILLED)
            #畫標準線
            cv2.line(img, (shoulder[0], shoulder[1]), (knee[0], knee[1]), (255, 255, 255), 3)
            
            #畫紅線
            cv2.line(img, (shoulder[0], shoulder[1]), (hip[0], hip[1]),(0, 0, 255), 2)
            cv2.line(img, (hip[0], hip[1]),(knee[0], knee[1]),(0, 0, 255), 2)
            cv2.line(img, (knee[0], knee[1]), (ankle[0], ankle[1]), (0, 0, 255), 2)
            
            #算角度還有顯示
            angle = utils.calculate_angle(shoulder, hip, knee)
            cv2.putText(img, str(int(angle)), (hip[0], hip[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    else:
            print("身體完全沒入鏡，暫停計算")
    # --- 4. 顯示畫面 ---
    cv2.imshow("Bridge Exercise - Week 2 & 3", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()