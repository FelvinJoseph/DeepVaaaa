<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />


# DeepVaaaa 🥱


## Basic Details
### Team Name: Syntax_Error;


### Team Members
- Team Lead: Felvin Joseph A F - RIT Kottayam
- Member 2: Sharoon Kiran P - RIT Kottayam

### Project Description
Ever caught yourself yawning in the middle of a Zoom meeting or in work? 😴 This app is here to catch you in the act! 📸 Using AI magic 🪄 and your webcam 📷, it detects yawns in real-time, plays a warning beep 🔔, and even flashes a red alert 🚨 if you yawn twice. Perfect for sleepy coders, night owls 🦉, and online class survivors! 🎓

### The Problem (that doesn't exist)
People yawn. A lot. 😪 Especially during online meetings, late-night coding sessions, or that one boring lecture we all know about 🥱. But what if… someone stopped you from yawning? 🤯 Nobody asked for it, but here we are — solving a problem that was never a problem in the first place! 🚀

### The Solution (that nobody asked for)
Introducing DeepVaaaa😴 - the Yawn Detector Pro — the Streamlit AI that spies on your face through your webcam 👀
catches your yawns in 4K 😴📸
blasts a sound to shame you 🔊
and if you dare yawn twice… boom! 
🚨 Red Alert Mode with a funny roast message
It even saves screenshots so you can relive your sleepy crimes forever 💤📂.

We can add more features to send the weird screenshots to someone or make alarms in the inverted time of yawning etc..

## Technical Details
### Technologies/Components Used
- Languages used 🐍:Python 3.10.11
- Frameworks used 🌐: Streamlit
- Libraries used :
  streamlit-webrtc – powers the live webcam feed 🎥
  opencv-python – captures your sleepy face in HD 🖼️
  mediapipe – detects the yawn of doom 🤖
  numpy – crunches mouth aspect ratio numbers 📊
  Pillow (PIL) – makes your yawns NFT-worthy 🎨
  playsound – plays the “wake up!” sound effect 🔊
  base64 – (optional) for embedding audio like a cyberpunk hacker 🕵️‍♂️
  pandas – keeps track of all your yawn crimes in a log 📜
- Tools used 🧰:
  VS Code – for writing and debugging without actually yawning 🖋️
  Git & GitHub – so you can brag about your “scientific” project 🐙
  Local machine webcam – the ultimate sleepy detector 📸


### Implementation
For Software:
# Installation
- First, make sure you have Python 3.9+ installed.
- Then, install the required dependencies:
  ```bash
  # Create a virtual environment (optional but recommended)
  python -m venv venv
  venv\Scripts\activate   # On Windows
  source venv/bin/activate   # On macOS/Linux
  ```
- Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
- Add your beep sound
  Place a short .wav file named beep.wav in the same folder as yawn_app.py
- Make sure that the structure is correct
  ```arduino
  yawn-detector-pro/
  ├── beep.wav
  ├── yawn_app.py
  ├── requirements.txt
  ```
# Run
```bash
streamlit run yawn_app.py

#Then open the link provided (usually http://localhost:8501/) in your browser.
```
Steps:
- Allow camera access when prompted in your browser.
- Keep your face visible to the camera.
- Start yawning

Output files 📂
- Screenshots: saved in screenshots/
- NFT Images: saved in nft_images/
- Yawn History: shown in the app log

### Project Documentation
For Software:

# Screenshots (Add at least 3)
![Screenshot1]
<img width="1850" height="960" alt="Screenshot 2025-08-16 125335" src="https://github.com/user-attachments/assets/21a5a011-7ac1-487d-ab12-2dd17e26be5f" />
The app’s home screen where you activate the webcam and start yawn detection.

![Screenshot2]
<img width="1847" height="962" alt="Screenshot 2025-08-16 125445" src="https://github.com/user-attachments/assets/fbd64d19-ead2-4731-ba3b-5418c6b4087a" />
The moment a yawn is detected — webcam feed is captured and a beep alert is triggered.

![Screenshot3]
<img width="1847" height="969" alt="Screenshot 2025-08-16 125504" src="https://github.com/user-attachments/assets/f45c6d73-e5e0-4545-9fe6-515813981ade" />
After the second yawn, the app switches into Red Alert Mode with a funny warning message.

![Screenshot4]
<img width="494" height="770" alt="Screenshot 2025-08-16 125526" src="https://github.com/user-attachments/assets/8dfa8b92-98d3-4257-ae8c-26d3f0bc06f0" />
The total number of yawns detected during the current session.A fun "NFT-style" snapshot of your most recent yawn, decorated with text, timestamp, and styling.

![Screenshot5]
<img width="1843" height="954" alt="Screenshot 2025-08-16 125608" src="https://github.com/user-attachments/assets/9bf15ee6-7c29-43c3-8b22-4218cfbb4edf" />
A quick table summarizing all detected yawns with their count number and timestamp

# Diagrams
Workflow of the Yawn Detector: Webcam → MediaPipe Landmark Detection → Yawn Detection Logic → Alerts (Beep / Screenshot / Red Mode).
- First we need to click the start button and select device
- Then after the webcam permissions are activated, our app becomes active and ready to detect yawns
- Whenever a yawn is detected, the beep sound is played with the gauge filled to half portion
- When the second yawn is detected the screen becomes red with a funny message
- Reset Session Counter can be used to count the no of yawns and display the nft images
- The yawn log wwith detailed information can also be viewed
- When the stop button is clicked it stops detection


### Project Demo
# Video
https://github.com/user-attachments/assets/88b85334-24be-44fa-8363-4de026ca8dad



## Team Contributions👨‍💻
Felvin Joseph A F :
- Built the core yawn detection logic 🥱
- Integrated sound alerts 🔊
- Designed the Nap-O-Meter & Red Alert Mode 🚨
- Created NFT-style yawn snapshots 🎨

Sharoon Kiran P:
- Helped with UI/UX improvements in Streamlit 🖥️
- Set up logs & history tracking 📜
- Assisted in README, docs, and GitHub repo setup 📂

---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--25-25?link=https%3A%2F%2Fwww.tinkerhub.org%2Fevents%2FQ2Q1TQKX6Q%2FUseless%2520Projects)
