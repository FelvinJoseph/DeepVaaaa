# import cv2
# import dlib
# import datetime
# import time
# from scipy.spatial import distance
# from imutils import face_utils
# from threading import Thread
# from playsound import playsound

# def mouth_aspect_ratio(mouth):
#     A = distance.euclidean(mouth[2], mouth[10])
#     B = distance.euclidean(mouth[4], mouth[8])
#     C = distance.euclidean(mouth[0], mouth[6])
#     mar = (A + B) / (2.0 * C)
#     return mar

# def play_beep():
#     playsound("beep.wav")

# def play_alarm():
#     playsound("alarm.wav")

# def get_flipped_time(now):
#     flipped_hour = (now.hour + 12) % 24
#     return now.replace(hour=flipped_hour, second=0, microsecond=0)

# def alarm_checker(alarm_time, index):
#     print(f"[INFO] Alarm #{index} set for: {alarm_time.strftime('%I:%M %p')}")
#     while True:
#         now = datetime.datetime.now().replace(second=0, microsecond=0)
#         if now == alarm_time:
#             print(f"[ALARM] Alarm #{index} ringing at {now.strftime('%I:%M %p')}!")
#             play_alarm()
#             break
#         time.sleep(1)

# # Constants
# MOUTH_AR_THRESH = 0.7
# MOUTH_AR_CONSEC_FRAMES = 15

# # State
# COUNTER = 0
# yawn_count = 0

# # Setup
# cap = cv2.VideoCapture(0)
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# print("[INFO] Yawn detection started. Press Q to quit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)

#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#         mouth = shape[mStart:mEnd]
#         mar = mouth_aspect_ratio(mouth)

#         if mar > MOUTH_AR_THRESH:
#             COUNTER += 1
#             if COUNTER >= MOUTH_AR_CONSEC_FRAMES:
#                 yawn_count += 1
#                 print(f"[YAWN] Yawn #{yawn_count} detected.")
#                 Thread(target=play_beep).start()

#                 now = datetime.datetime.now()
#                 flipped_time = get_flipped_time(now)
#                 Thread(target=alarm_checker, args=(flipped_time, yawn_count), daemon=True).start()

#                 COUNTER = 0
#         else:
#             COUNTER = 0

#     cv2.imshow(f"Yawn Detection App - Total Yawns: {yawn_count}", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import dlib
# import datetime
# import time
# from scipy.spatial import distance
# from imutils import face_utils
# from threading import Thread
# from playsound import playsound

# # --- FUNCTIONS ---

# def mouth_aspect_ratio(mouth):
#     A = distance.euclidean(mouth[2], mouth[10])  # vertical
#     B = distance.euclidean(mouth[4], mouth[8])   # vertical
#     C = distance.euclidean(mouth[0], mouth[6])   # horizontal
#     mar = (A + B) / (2.0 * C)
#     return mar

# def play_beep():
#     playsound("beep.wav")

# def play_alarm():
#     playsound("alarm.wav")

# def get_flipped_time(now):
#     flipped_hour = (now.hour + 12) % 24
#     return now.replace(hour=flipped_hour, second=0, microsecond=0)

# def alarm_checker(alarm_time, index):
#     print(f"[INFO] Alarm #{index} set for: {alarm_time.strftime('%I:%M %p')}")
#     while True:
#         now = datetime.datetime.now().replace(second=0, microsecond=0)
#         if now == alarm_time:
#             print(f"[ALARM] Alarm #{index} ringing at {now.strftime('%I:%M %p')}!")
#             play_alarm()
#             break
#         time.sleep(1)

# # --- CONSTANTS ---
# MOUTH_AR_THRESH = 0.7
# MOUTH_AR_CONSEC_FRAMES = 15

# COUNTER = 0
# yawn_count = 0

# # --- SETUP ---
# print("[INFO] Loading model and initializing camera...")
# cap = cv2.VideoCapture(0)
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# print("[INFO] Yawn detection started. Press Q to quit.")

# # --- MAIN LOOP ---
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)

#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#         mouth = shape[mStart:mEnd]
#         mar = mouth_aspect_ratio(mouth)

#         if mar > MOUTH_AR_THRESH:
#             COUNTER += 1
#             if COUNTER >= MOUTH_AR_CONSEC_FRAMES:
#                 yawn_count += 1
#                 print(f"[YAWN] Yawn #{yawn_count} detected.")
#                 Thread(target=play_beep).start()

#                 now = datetime.datetime.now()
#                 flipped_time = get_flipped_time(now)
#                 Thread(target=alarm_checker, args=(flipped_time, yawn_count), daemon=True).start()

#                 COUNTER = 0
#         else:
#             COUNTER = 0

#     cv2.imshow(f"Yawn Detection App - Total Yawns: {yawn_count}", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # --- CLEANUP ---
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import datetime
# import time
# import numpy as np
# from playsound import playsound
# from threading import Thread
# import mediapipe as mp

# # --- FUNCTIONS ---

# def mouth_aspect_ratio(landmarks):
#     # These indices are approximate for inner mouth using MediaPipe FaceMesh
#     top_lip = landmarks[13]
#     bottom_lip = landmarks[14]
#     left_lip = landmarks[78]
#     right_lip = landmarks[308]

#     A = np.linalg.norm(top_lip - bottom_lip)
#     C = np.linalg.norm(left_lip - right_lip)

#     mar = A / C
#     return mar

# def play_beep():
#     playsound("beep.wav")

# def play_alarm():
#     playsound("alarm.wav")

# def get_flipped_time(now):
#     flipped_hour = (now.hour + 12) % 24
#     return now.replace(hour=flipped_hour, second=0, microsecond=0)

# def alarm_checker(alarm_time, index):
#     print(f"[INFO] Alarm #{index} set for: {alarm_time.strftime('%I:%M %p')}")
#     while True:
#         now = datetime.datetime.now().replace(second=0, microsecond=0)
#         if now == alarm_time:
#             print(f"[ALARM] Alarm #{index} ringing at {now.strftime('%I:%M %p')}!")
#             play_alarm()
#             break
#         time.sleep(1)

# # --- CONSTANTS ---
# MOUTH_AR_THRESH = 0.4
# MOUTH_AR_CONSEC_FRAMES = 15

# COUNTER = 0
# yawn_count = 0

# # --- SETUP ---
# print("[INFO] Loading MediaPipe model and initializing camera...")
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
# cap = cv2.VideoCapture(0)

# print("[INFO] Yawn detection started. Press Q to quit.")

# # --- MAIN LOOP ---
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     if results.multi_face_landmarks:
#         face_landmarks = results.multi_face_landmarks[0]
#         h, w, _ = frame.shape
#         landmarks = np.array([[p.x * w, p.y * h] for p in face_landmarks.landmark])

#         mar = mouth_aspect_ratio(landmarks)

#         if mar > MOUTH_AR_THRESH:
#             COUNTER += 1
#             if COUNTER >= MOUTH_AR_CONSEC_FRAMES:
#                 yawn_count += 1
#                 print(f"[YAWN] Yawn #{yawn_count} detected.")
#                 Thread(target=play_beep).start()

#                 now = datetime.datetime.now()
#                 flipped_time = get_flipped_time(now)
#                 Thread(target=alarm_checker, args=(flipped_time, yawn_count), daemon=True).start()

#                 COUNTER = 0
#         else:
#             COUNTER = 0

#     cv2.imshow(f"Yawn Detection App - Total Yawns: {yawn_count}", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # --- CLEANUP ---
# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# from playsound import playsound
# import mediapipe as mp
# from threading import Thread

# # Initialize MediaPipe Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # Mouth landmark indices (approximate for MediaPipe)
# MOUTH_INNER_LANDMARKS = [13, 14, 78, 308]  # Top, bottom, left, right points

# # Constants
# MOUTH_AR_THRESH = 0.4  # Adjust this threshold based on testing
# MOUTH_AR_CONSEC_FRAMES = 15  # Number of consecutive frames to trigger yawn
# YAWN_BEEP_FILE = "beep.wav"  # Make sure this file exists in your directory

# def mouth_aspect_ratio(landmarks):
#     """Calculate mouth aspect ratio using MediaPipe landmarks"""
#     # Get the specific mouth points
#     top_lip = landmarks[13]
#     bottom_lip = landmarks[14]
#     left_lip = landmarks[78]
#     right_lip = landmarks[308]
    
#     # Calculate vertical and horizontal distances
#     vertical_dist = np.linalg.norm(top_lip - bottom_lip)
#     horizontal_dist = np.linalg.norm(left_lip - right_lip)
    
#     # Compute mouth aspect ratio
#     mar = vertical_dist / horizontal_dist
#     return mar

# def play_beep():
#     """Play the beep sound in a separate thread"""
#     try:
#         playsound(YAWN_BEEP_FILE)
#     except:
#         print("Could not play beep sound")

# def main():
#     print("[INFO] Yawn detection started. Press 'Q' to quit.")
    
#     cap = cv2.VideoCapture(0)
#     counter = 0
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
            
#         # Convert to RGB and process with MediaPipe
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb)
        
#         if results.multi_face_landmarks:
#             face_landmarks = results.multi_face_landmarks[0]
#             h, w = frame.shape[:2]
            
#             # Convert landmarks to pixel coordinates
#             landmarks = np.array([[p.x * w, p.y * h] for p in face_landmarks.landmark])
            
#             # Calculate mouth aspect ratio
#             mar = mouth_aspect_ratio(landmarks)
            
#             # Check for yawn
#             if mar > MOUTH_AR_THRESH:
#                 counter += 1
#                 if counter >= MOUTH_AR_CONSEC_FRAMES:
#                     print("[YAWN] Yawn detected!")
#                     Thread(target=play_beep).start()
#                     counter = 0
#             else:
#                 counter = 0
                
#             # Draw mouth landmarks (optional visualization)
#             for idx in MOUTH_INNER_LANDMARKS:
#                 x, y = landmarks[idx]
#                 cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        
#         # Display frame
#         cv2.imshow("Yawn Detection", frame)
        
#         # Exit on 'Q' key
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
            
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

from gtts import gTTS
import cv2
import datetime
import time
import numpy as np
from playsound import playsound
from threading import Thread
import mediapipe as mp
import tkinter as tk
from tkinter import scrolledtext
import os
import pandas as pd
import matplotlib.pyplot as plt



# Global variables
yawn_log = []
MOUTH_AR_THRESH = 0.4
MOUTH_AR_CONSEC_FRAMES = 15
alarm_time = None

# Ensure folders exist
os.makedirs("screenshots", exist_ok=True)
os.makedirs("nft_yawns", exist_ok=True)

class YawnDetectorApp:
    def play_malayalam_poem(self):
        poem = "മഴ പെയ്യുന്നു രാപകലൊടുങ്ങി\nചിത്രങ്ങളായ് ഓർമ്മകൾ വരഞ്ഞു\nനിദ്ര വിഴുങ്ങി നിശബ്ദത വീണു\nപുഞ്ചിരിപ്പോലൊരു ഓളം കവിഞ്ഞു"
        try:
            tts = gTTS(text=poem, lang='ml')
            tts.save("poem.mp3")
            playsound("poem.mp3")
            os.remove("poem.mp3")  # delete after playing
        except Exception as e:
            print(f"❌ Failed to play poem: {e}")

    def handle_yawn_audio(self):
        self.play_beep()               # Play the beep sound
        self.play_malayalam_poem()     # Then play the Malayalam poem



    def __init__(self, root):
        self.root = root
        self.root.title("Yawn Detection App")
        self.yawn_count = 0
        self.counter = 0
        self.detection_active = False

        self.setup_gui()

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.cap = None

    def setup_gui(self):
        self.btn_start = tk.Button(self.root, text="Start Detection", command=self.start_detection)
        self.btn_start.pack(pady=5)

        self.btn_stop = tk.Button(self.root, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.btn_stop.pack(pady=5)

        self.lbl_count = tk.Label(self.root, text="Yawns Detected: 0", font=('Arial', 14))
        self.lbl_count.pack(pady=10)

        self.lbl_napometer = tk.Label(self.root, text="Nap-o-Meter: [          ]", font=('Arial', 12))
        self.lbl_napometer.pack(pady=2)


        self.lbl_alarm = tk.Label(self.root, text="Next Alarm: None", font=('Arial', 12))
        self.lbl_alarm.pack(pady=5)

        self.log_text = scrolledtext.ScrolledText(self.root, width=50, height=10)
        self.log_text.pack(pady=10, padx=10)
        self.log_text.insert(tk.END, "Yawn Detection Log:\n")

        self.lbl_video = tk.Label(self.root)
        self.lbl_video.pack()

    def start_detection(self):
        self.detection_active = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def stop_detection(self):
        self.detection_active = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()

    def mouth_aspect_ratio(self, landmarks):
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        left_lip = landmarks[78]
        right_lip = landmarks[308]
        A = np.linalg.norm(top_lip - bottom_lip)
        C = np.linalg.norm(left_lip - right_lip)
        return A / C

    def play_beep(self):
        try:
            playsound("beep.wav")
        except:
            print("Beep sound not found")

    def flip_am_pm_time(self):
        now = datetime.datetime.now()
        flipped_hour = (now.hour + 12) % 24
        flipped_time = now.replace(hour=flipped_hour)
        return flipped_time.strftime("%H:%M")

    def check_alarm(self, target_time_str):
        print(f"Alarm set for: {target_time_str}")
        while self.detection_active:
            now = datetime.datetime.now().strftime("%H:%M")
            if now == target_time_str:
                print("⏰ Alarm ringing!")
                try:
                    playsound("alarm.wav")
                except:
                    print("alarm.wav not found")
                break
            time.sleep(10)

    def generate_nft_image(self, frame, yawn_num):
        # Apply glitch and color inversion
        nft = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        nft = 255 - nft  # invert colors
        rows, cols, _ = nft.shape
        for i in range(0, rows, 20):
            offset = np.random.randint(-5, 5)
            nft[i:i+10] = np.roll(nft[i:i+10], offset, axis=1)

        filename = f"nft_yawns/yawnNFT_{yawn_num:03d}.png"
        cv2.imwrite(filename, cv2.cvtColor(nft, cv2.COLOR_RGB2BGR))
        print(f"🖼️ NFT saved: {filename}")

    def log_yawn(self, frame):
        yawn_time = datetime.datetime.now()
        time_str = yawn_time.strftime("%Y-%m-%d %H:%M:%S")
        yawn_log.append({"time": time_str})
        self.yawn_count += 1

        # Update Nap-o-Meter
        filled_blocks = min(self.yawn_count, 3)  # max 10 blocks
        nap_bar = "[" + "█" * filled_blocks + " " * (3 - filled_blocks) + "]"
        self.lbl_napometer.config(text=f"Nap-o-Meter: {nap_bar}")

        # Trigger message when fully filled
        if filled_blocks == 3:
            self.log_text.insert(tk.END, "⚠️ Critical yawn level! You need a nap ASAP! 🛌\n")


        self.lbl_count.config(text=f"Yawns Detected: {self.yawn_count}")
        self.log_text.insert(tk.END, f"{self.yawn_count}. Yawn at {time_str}\n")
        self.log_text.see(tk.END)

        screenshot_filename = f"screenshots/yawn_{yawn_time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(screenshot_filename, frame)
        print(f"📸 Screenshot saved: {screenshot_filename}")

        self.generate_nft_image(frame, self.yawn_count)

        flipped = self.flip_am_pm_time()
        self.lbl_alarm.config(text=f"Next Alarm: {flipped}")
        Thread(target=self.check_alarm, args=(flipped,)).start()

    def update_frame(self):
        if not self.detection_active:
            return

        ret, frame = self.cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                landmarks = np.array([[p.x * w, p.y * h] for p in face_landmarks.landmark])

                mar = self.mouth_aspect_ratio(landmarks)

                if mar > MOUTH_AR_THRESH:
                    self.counter += 1
                    if self.counter >= MOUTH_AR_CONSEC_FRAMES:
                        Thread(target=self.handle_yawn_audio).start()

                        self.log_yawn(frame)
                        self.counter = 0
                else:
                    self.counter = 0

                for idx in [13, 14, 78, 308]:
                    x, y = landmarks[idx]
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = tk.PhotoImage(data=cv2.imencode('.png', img)[1].tobytes())
            self.lbl_video.config(image=img)
            self.lbl_video.image = img

        self.root.after(10, self.update_frame)

    def save_log(self):
        df = pd.DataFrame(yawn_log)
        filename = f"yawn_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Log saved as {filename}")

if __name__ == "__main__":
    root = tk.Tk()
    app = YawnDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [app.stop_detection(), root.quit()])
    root.mainloop()


