from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn import functional as F
import spotipy
from spotipy.oauth2 import SpotifyOAuth

app = Flask(__name__)

# 모델 로드
model_name = "yeeeeeeeeeeeeeeeeeeo/Kobert-parrot"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Spotify 설정 (Render 배포 후 자동 적용)
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="c2442928740348a1839090999b8b4994",
    client_secret="0627291231d046338dcebf8a78e233ad",
    redirect_uri="https://parrot-song.onrender.com/callback",
    scope="playlist-modify-public",
    cache_path="token.json"
))

labels = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']

@app.route("/")
def intro():
    return render_template("intro.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.json.get("text", "").strip()
    if not text:
        return jsonify({"emotion": "중립", "img": "/static/images/중립.png", "songs": ""})

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1).item()
    
    emotion = labels[pred]
    img_path = f"/static/images/{emotion}.png"

    queries = {
        '공포': 'calm healing ambient', '놀람': 'energetic surprise pop',
        '분노': 'angry rock metal aggressive', '슬픔': 'sad emotional ballad acoustic',
        '중립': 'chill lo-fi calm', '행복': 'happy dance pop summer upbeat',
        '혐오': 'intense aggressive hiphop rap'
    }
    try:
        results = sp.search(q=queries[emotion], type='track', limit=3, market='KR')
        songs = [f"<li><a href='{t['external_urls']['spotify']}' target='_blank'>▶ {t['name']} - {t['artists'][0]['name']}</a></li>"
                 for t in results['tracks']['items']]
        song_list = "".join(songs) or "<li>곧 노래가 나와요!</li>"
    except:
        song_list = "<li>스포티파이 연결 중...</li>"

    return jsonify({"emotion": emotion, "img": img_path, "songs": song_list})

@app.route("/callback")
def callback():
    return "Spotify 로그인 완료! 창을 닫고 돌아가세요"

if __name__ == "__main__":
    app.run(debug=True)
