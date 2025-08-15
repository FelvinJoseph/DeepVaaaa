
import os
import sys
import threading
import logging
import cv2
import numpy as np
import streamlit as st
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import datetime
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pandas as pd
import time
from playsound import playsound  # ✅ For local audio playback

# Suppress unnecessary warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

# Directories
os.makedirs("screenshots", exist_ok=True)
os.makedirs("nft_images", exist_ok=True)

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Constants
MOUTH_AR_THRESH = 0.45
MOUTH_AR_CONSEC_FRAMES = 5
BEEP_FILE = "beep.wav"  # ✅ Keep a short wav in same folder

# NFT Image Generator
def create_nft_image(image, yawn_count):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    nft_img = pil_img.filter(ImageFilter.EMBOSS)
    draw = ImageDraw.Draw(nft_img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 10), f"YAWN NFT #{yawn_count}", fill="gold", font=font)
    draw.text((10, 40), "Certified Sleepy Moment", fill="white", font=font)
    draw.text((10, 70), datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), fill="cyan", font=font)
    return cv2.cvtColor(np.array(nft_img), cv2.COLOR_RGB2BGR)

# Video Processor
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.counter = 0
        self.frame_buffer = []
        self.max_buffer = 5

        self.total_yawns = 0
        self.session_yawns = 0
        self.show_alert = False
        self.alert_start_time = None
        self.yawn_log = []

        self.yawned_in_current_open = False

    def mouth_aspect_ratio(self, landmarks):
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        left_lip = landmarks[78]
        right_lip = landmarks[308]
        A = np.linalg.norm(top_lip - bottom_lip)
        C = np.linalg.norm(left_lip - right_lip)
        if C == 0:
            return 0.0
        return A / C

    def _play_beep_thread(self, path=BEEP_FILE):
        """Play a short beep sound locally."""
        try:
            if os.path.exists(path):
                playsound(path)
                return
        except Exception as e:
            logger.error(f"playsound error: {e}")
        # Fallback: Windows beep
        try:
            if sys.platform.startswith("win"):
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except Exception as e:
            logger.error(f"winsound fallback error: {e}")

    def play_beep(self):
        threading.Thread(target=self._play_beep_thread, daemon=True).start()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            image = frame.to_ndarray(format="bgr24")
            h, w = image.shape[:2]

            self.frame_buffer.append(image)
            if len(self.frame_buffer) > self.max_buffer:
                self.frame_buffer.pop(0)

            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                lm_coords = np.array([[lm.x * w, lm.y * h] for lm in landmarks])

                mar = self.mouth_aspect_ratio(lm_coords)
                cv2.putText(image, f"MAR: {mar:.2f}", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if mar > MOUTH_AR_THRESH:
                    self.counter += 1
                    if self.counter >= MOUTH_AR_CONSEC_FRAMES and not self.yawned_in_current_open:
                        self.total_yawns += 1
                        self.session_yawns += 1
                        self.yawned_in_current_open = True

                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"screenshots/yawn_{self.total_yawns}_{timestamp}.jpg"
                        cv2.imwrite(screenshot_path, image)

                        nft_image = create_nft_image(image, self.total_yawns)
                        nft_path = f"nft_images/nft_{self.total_yawns}_{timestamp}.jpg"
                        cv2.imwrite(nft_path, nft_image)

                        self.yawn_log.append({
                            "count": self.total_yawns,
                            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image": screenshot_path,
                            "nft_image": nft_path
                        })

                        self.play_beep()  # ✅ Local sound here

                        if self.session_yawns >= 2:
                            self.show_alert = True
                            self.alert_start_time = time.time()
                else:
                    self.counter = 0
                    self.yawned_in_current_open = False

            if self.show_alert:
                overlay = image.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)

                funny_messages = [
                    "NAP TIME! 🛌💤", "SLEEP ATTACK! 😴",
                    "YAWN OVERLOAD! 🥱", "BED CALLING! 🛏️"
                ]
                msg = funny_messages[self.total_yawns % len(funny_messages)]
                cv2.putText(image, msg, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                if self.alert_start_time and (time.time() - self.alert_start_time > 5):
                    self.show_alert = False
                    self.session_yawns = 0

            meter_height = min(self.session_yawns * 50, 100)
            cv2.rectangle(image, (10, 10), (40, 110), (255, 255, 255), 2)
            if self.session_yawns > 0:
                cv2.rectangle(image, (12, 110 - meter_height), (38, 108), (0, 0, 255), -1)
            cv2.putText(image, f"{self.session_yawns}/2", (15, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return av.VideoFrame.from_ndarray(image, format="bgr24")

        except Exception as e:
            logger.error(f"Processing error: {e}")
            if self.frame_buffer:
                return av.VideoFrame.from_ndarray(self.frame_buffer[-1], format="bgr24")
            return frame

# Streamlit UI
def main():
    st.set_page_config(page_title="Yawn Detector Pro", layout="wide")

    st.title("😴 Yawn Detector Pro")
    st.markdown("""Detects yawns, tracks your sleepiness, and creates NFT memories of your yawns!
    - **Nap-O-Meter** shows current session yawns
    - **Red alert** after 2 yawns with funny messages
    - **NFT generator** saves artistic yawn moments""")

    col1, col2 = st.columns([3, 1])
    with col1:
        ctx = webrtc_streamer(
            key="yawn-detector",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True
        )

    with col2:
        st.subheader("Nap-O-Meter")
        if ctx and ctx.video_processor:
            processor = ctx.video_processor
            st.progress(min(processor.session_yawns * 50, 100))
            st.caption(f"Session yawns: {processor.session_yawns}/2")

            st.subheader("Total Yawns")
            st.metric("Count", processor.total_yawns)

            if st.button("Reset Session Counter"):
                processor.session_yawns = 0
                processor.show_alert = False
                st.rerun()

            st.subheader("Latest Yawn NFT")
            if processor.yawn_log:
                last_yawn = processor.yawn_log[-1]
                st.image(last_yawn["nft_image"], caption=f"Yawn #{last_yawn['count']}")
                if st.button("Save This NFT"):
                    st.success(f"Saved at {last_yawn['nft_image']}")
            else:
                st.info("No yawns detected yet")

    st.subheader("Yawn History Log")
    if ctx and ctx.video_processor and ctx.video_processor.yawn_log:
        log_df = pd.DataFrame(ctx.video_processor.yawn_log)
        st.dataframe(log_df[["count", "time"]], use_container_width=True)

        with st.expander("Detailed Logs"):
            st.write(ctx.video_processor.yawn_log)

if __name__ == "__main__":
    main()



