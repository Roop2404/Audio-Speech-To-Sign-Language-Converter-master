from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.staticfiles import finders
from django.contrib.auth.decorators import login_required
from django.views.decorators.gzip import gzip_page
import cv2
import mediapipe as mp
import pyttsx3
import threading
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine in a better way
class SpeechEngine:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.current_speech = None
        self.is_running = True
        self.lock = threading.Lock()
        self.start_engine()

    def start_engine(self):
        def run_engine():
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            
            while self.is_running:
                try:
                    # Get the latest text with a short timeout
                    text = self.speech_queue.get(timeout=0.1)
                    
                    # Clear queue of any backed up items
                    while not self.speech_queue.empty():
                        try:
                            self.speech_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    with self.lock:
                        self.current_speech = text
                        engine.say(text)
                        engine.runAndWait()
                        self.current_speech = None
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Speech error: {e}")
                    time.sleep(0.1)

        threading.Thread(target=run_engine, daemon=True).start()

    def speak(self, text):
        # Only queue new speech if it's different from current speech
        with self.lock:
            if text != self.current_speech:
                # Clear queue before adding new text
                while not self.speech_queue.empty():
                    try:
                        self.speech_queue.get_nowait()
                    except queue.Empty:
                        break
                self.speech_queue.put(text)

# Create global speech engine instance
speech_engine = SpeechEngine()

# Improved sign mappings with confidence thresholds
SIGNS = {
    'hello': 'Hello',
    'good': 'Good',
    'bad': 'Bad',
    'thank_you': 'Thank you',
    'yes': 'Yes',
    'no': 'No',
    'please': 'Please',
    'how_are_you': 'How are you'
}

class SignDetector:
    def __init__(self):
        self.sign_history = []
        self.history_size = 2  # Reduced for faster response
        self.last_detection = None
        self.detection_count = 0
        self.last_confident_sign = None
        self.confidence_threshold = 2  # Reduced threshold for faster response

    def calculate_hand_features(self, hand_landmarks):
        """Calculate comprehensive hand features"""
        # Get all landmark positions
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        # Calculate palm center
        palm_center = np.mean(points[[0, 5, 17]], axis=0)
        
        # Calculate finger states
        finger_states = []
        for finger_base, finger_tip in [(5,8), (9,12), (13,16), (17,20)]:
            extended = points[finger_tip][1] < points[finger_base][1]
            finger_states.append(extended)
            
        # Calculate thumb state
        thumb_extended = points[4][0] > points[3][0]
        
        return {
            'palm_center': palm_center,
            'finger_states': finger_states,
            'thumb_extended': thumb_extended,
            'points': points
        }

    def detect_sign(self, hand_landmarks):
        """Improved sign detection with more robust features"""
        features = self.calculate_hand_features(hand_landmarks)
        
        # Get finger states
        fingers = features['finger_states']
        thumb = features['thumb_extended']
        points = features['points']

        # More accurate sign detection rules
        if all(fingers) and thumb:
            return 'hello'
        elif not any(fingers) and thumb:
            return 'good'
        elif not any(fingers) and not thumb:
            return 'bad'
        elif fingers[0] and fingers[1] and not any(fingers[2:]):
            # Check specific angle for thank_you
            angle = np.arctan2(points[8][1] - points[5][1], 
                             points[8][0] - points[5][0])
            if -0.3 < angle < 0.3:
                return 'thank_you'
        elif fingers[0] and not any(fingers[1:]) and thumb:
            return 'yes'
        elif not any(fingers) and not thumb:
            return 'no'
        elif fingers[0] and not any(fingers[1:]) and not thumb:
            return 'please'
        elif all(fingers) and not thumb:
            return 'how_are_you'
        
        return 'unknown'

    def get_confident_sign(self, detected_sign):
        """Improved confidence calculation with faster response"""
        if detected_sign == self.last_detection:
            self.detection_count += 1
        else:
            self.detection_count = 1
            self.last_detection = detected_sign

        # Return confident sign more quickly
        if self.detection_count >= self.confidence_threshold:
            if detected_sign != self.last_confident_sign:
                self.last_confident_sign = detected_sign
                return detected_sign, True  # True indicates it's a new detection
            return detected_sign, False  # False indicates it's a repeated detection
        return 'unknown', False

@gzip_page
@login_required(login_url="login")
def sign_to_speech_view(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    detector = SignDetector()
    prev_sign = None
    last_spoken_time = 0
    speech_cooldown = 0.1  # Very short cooldown

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
                )
                
                # Detect sign
                detected_sign = detector.detect_sign(hand_landmarks)
                confident_sign, is_new = detector.get_confident_sign(detected_sign)
                
                # Handle speech output only for new detections
                current_time = time.time()
                if (confident_sign != 'unknown' and 
                    is_new and 
                    current_time - last_spoken_time >= speech_cooldown):
                    
                    phrase = SIGNS.get(confident_sign, "Unrecognized sign")
                    speech_engine.speak(phrase)
                    last_spoken_time = current_time
                    prev_sign = confident_sign

                # Visual feedback
                status_color = (0, 255, 0) if confident_sign != 'unknown' else (0, 165, 255)
                cv2.putText(frame, f"Sign: {confident_sign}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, status_color, 2)
                cv2.putText(frame, f"Confidence: {detector.detection_count}/{detector.confidence_threshold}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, status_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def home_view(request):
    return render(request, 'home.html')

def about_view(request):
    return render(request, 'about.html')

def contact_view(request):
    return render(request, 'contact.html')

@login_required(login_url="login")
def animation_view(request):
    if request.method == 'POST':
        text = request.POST.get('sen')
        # Tokenizing the sentence
        text.lower()
        words = word_tokenize(text)

        tagged = nltk.pos_tag(words)
        tense = {
            "future": len([word for word in tagged if word[1] == "MD"]),
            "present": len([word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]]),
            "past": len([word for word in tagged if word[1] in ["VBD", "VBN"]]),
            "present_continuous": len([word for word in tagged if word[1] in ["VBG"]])
        }

        # Stopwords that will be removed
        stop_words = set(["mightn't", 're', 'wasn', 'wouldn', 'be', 'has', 'that', 'does', 'shouldn', 'do', "you've", 'off', 'for', "didn't", 'm', 'ain', 'haven', "weren't", 'are', "she's", "wasn't", 'its', "haven't", "wouldn't", 'don', 'weren', 's', "you'd", "don't", 'doesn', "hadn't", 'is', 'was', "that'll", "should've", 'a', 'then', 'the', 'mustn', 'i', 'nor', 'as', "it's", "needn't", 'd', 'am', 'have', 'hasn', 'o', "aren't", "you'll", "couldn't", "you're", "mustn't", 'didn', "doesn't", 'll', 'an', 'hadn', 'whom', 'y', "hasn't", 'itself', 'couldn', 'needn', "shan't", 'isn', 'been', 'such', 'shan', "shouldn't", 'aren', 'being', 'were', 'did', 'ma', 't', 'having', 'mightn', 've', "isn't", "won't"])

        # Removing stopwords and applying lemmatizing NLP process to words
        lr = WordNetLemmatizer()
        filtered_text = []
        for w, p in zip(words, tagged):
            if w not in stop_words:
                if p[1] in ['VBG', 'VBD', 'VBZ', 'VBN', 'NN']:
                    filtered_text.append(lr.lemmatize(w, pos='v'))
                elif p[1] in ['JJ', 'JJR', 'JJS', 'RBR', 'RBS']:
                    filtered_text.append(lr.lemmatize(w, pos='a'))
                else:
                    filtered_text.append(lr.lemmatize(w))

        # Adding the specific word to specify tense
        words = filtered_text
        temp = []
        for w in words:
            if w == 'I':
                temp.append('Me')
            else:
                temp.append(w)
        words = temp
        probable_tense = max(tense, key=tense.get)

        if probable_tense == "past" and tense["past"] >= 1:
            temp = ["Before"]
            temp = temp + words
            words = temp
        elif probable_tense == "future" and tense["future"] >= 1:
            if "Will" not in words:
                temp = ["Will"]
                temp = temp + words
                words = temp
        elif probable_tense == "present":
            if tense["present_continuous"] >= 1:
                temp = ["Now"]
                temp = temp + words
                words = temp

        filtered_text = []
        for w in words:
            path = w + ".mp4"
            f = finders.find(path)
            # Splitting the word if its animation is not present in database
            if not f:
                for c in w:
                    filtered_text.append(c)
            # Otherwise animation of word
            else:
                filtered_text.append(w)
        words = filtered_text

        return render(request, 'animation.html', {'words': words, 'text': text})
    else:
        return render(request, 'animation.html')

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            # Log the user in
            return redirect('animation')
    else:
        form = UserCreationForm()
    return render(request, 'signup.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            # Log in user
            user = form.get_user()
            login(request, user)
            if 'next' in request.POST:
                return redirect(request.POST.get('next'))
            else:
                return redirect('animation')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    return redirect("home")
