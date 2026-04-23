# 🎵 Project Context: Emotion-Aware Music Recommendation Chatbot

> This document serves as the full contextual reference for AI assistance throughout this project.
> Any AI tool reading this should understand the full scope, stack, methodology, and constraints of this research system.

---

## 📌 Project Identity

| Field                  | Detail                                                                                             |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| **Project Title**      | An NLP-Based Chatbot for Empathetic Emotional Support and Mood-Based Music Playlist Recommendation |
| **Research Type**      | Applied NLP System / Final Year Research Project                                                   |
| **Target Publication** | Journal MIND (Multimedia, AI, Networking, Database) — SINTA 3, ITENAS Bandung                      |
| **Team Size**          | 3 People                                                                                           |
| **Language**           | English (paper), Indonesian (possible UI)                                                          |

---

## 🎯 Project Overview

This project builds a **hybrid chatbot system** that:

1. Accepts a user's **rant/emotional text input** (one paragraph)
2. Detects the user's **dominant emotion** using NLP
3. Returns an **empathetic emotional support response** (text)
4. Recommends a **mood-based music playlist** (list of songs)

The word **"Hybrid"** refers to the combination of:

- **Retrieval-Based** approach for empathetic response generation
- **Rule-Based + Valence-Arousal Mapping** for music recommendation

---

## 👥 Team Work Scope

### Person 1 — NLP & AI Engineer

- Data collection & preprocessing
- Fine-tune emotion detection model (RoBERTa / IndoBERT on GoEmotions)
- Build retrieval-based empathetic response system
- Evaluate model (accuracy, F1-score, BLEU/ROUGE)
- Write NLP methodology section of the paper

### Person 2 — Backend & Integration Engineer

- Build Flask API (`/api/chat`, `/api/playlist` endpoints)
- Integrate NLP model into API pipeline
- Build Mood-to-Music mapping logic (Valence-Arousal model)
- Connect Spotify API for real-time playlist retrieval
- Handle full JSON request/response structure
- API testing via Postman

### Person 3 — Frontend & Evaluation Engineer

- Build chat UI using **Vite React + Tailwind CSS**
- Display support text + playlist cards
- Connect frontend to Flask via **Axios**
- Design and conduct user testing (SUS / PANAS scale)
- Collect and analyze user satisfaction data
- Write evaluation section of the paper + system figures

---

## 🗂️ Dataset Plan

### Dataset 1 — Conversational / Rant Dataset

**Purpose:** Train/source empathetic response generation

- **EmpatheticDialogues** (Facebook AI) — 25k emotional conversations ✅ Primary
- **DailyDialog** — general human dialogue with emotion labels
- **Counseling Conversations Dataset** — therapy-like responses

### Dataset 2 — Emotion / Sentiment Dataset

**Purpose:** Classify dominant emotion from user rant text

- **GoEmotions** (Google) — 58k Reddit comments, 27 emotion categories ✅ Primary
- **EmoBank** — valence, arousal, dominance ratings
- **SemEval 2018 Task 1** — multi-label emotion classification benchmark

### Dataset 3 — Mood-to-Music Mapping Dataset

**Purpose:** Match detected emotion/mood to appropriate songs

- **Spotify API + MoodyLyrics** — lyrics with mood tags ✅ Primary
- **PMEMO Dataset** — music emotion with valence & arousal scores
- **MER Dataset** — music features mapped to emotion quadrants

---

## 🧠 NLP Methods & Models

### Emotion Detection

| Method                               | Role                                        |
| ------------------------------------ | ------------------------------------------- |
| **RoBERTa fine-tuned on GoEmotions** | Primary emotion classifier (recommended)    |
| **IndoBERT**                         | Alternative if input is Indonesian language |
| **BiLSTM + Attention**               | Lightweight fallback option                 |

### Empathetic Response Generation

| Method                              | Role                                                   |
| ----------------------------------- | ------------------------------------------------------ |
| **Retrieval-Based (Sentence-BERT)** | Primary — retrieve best match from EmpatheticDialogues |
| **DialoGPT / BlenderBot**           | Optional generative layer (hybrid extension)           |

### Mood-to-Music Mapping

| Method                       | Role                                   |
| ---------------------------- | -------------------------------------- |
| **Valence-Arousal 2D Model** | Map emotion to mood quadrant           |
| **Rule-Based Mapping Table** | Emotion label → mood category → genre  |
| **Spotify API**              | Real-time song retrieval by mood/genre |

---

## 🔄 Full System Pipeline

```
USER INPUT (Rant Text)
        │
        ▼
[1] TEXT PREPROCESSING
    - Case folding
    - Remove punctuation / stopwords
    - Tokenization
    - Lemmatization
        │
        ▼
[2] EMOTION DETECTION  ← Dataset 2 (GoEmotions + RoBERTa)
    - Text embedding
    - Emotion classification
    - Output: dominant emotion label + confidence score
    - Output: valence & arousal scores
        │
        ├─────────────────────────┐
        ▼                         ▼
[3A] RESPONSE GENERATION     [3B] MOOD-TO-MUSIC MAPPING
     ← Dataset 1                  ← Dataset 3
     (EmpatheticDialogues)        (Spotify API / PMEMO)
     Retrieval-Based              Valence-Arousal → mood category
     → empathetic support text    → retrieve song list
        │                         │
        └──────────┬──────────────┘
                   ▼
[4] RESPONSE PACKAGING
    - Combine support text + playlist
    - Format into JSON
    - Return via Flask API
```

---

## 🌐 Frontend Flow (Vite React)

```
User types rant → Validate input → Build JSON payload
→ Show loading spinner → Axios POST /api/chat
→ Receive response → Render support text bubble
→ Render playlist cards → Re-enable input
```

**Error handling:** Show user-friendly error message if API returns 4xx/5xx

---

## ⚙️ Backend Flow (Flask)

```
Receive POST /api/chat → Validate JSON
→ Extract message → Log to session history
→ Pass to NLP pipeline → Package response
→ Save to chat history → Return HTTP 200 JSON
```

---

## 📦 JSON Data Structures

### Request (Frontend → Backend)

```json
{
  "user_id": "user_001",
  "message": "I'm so tired and stressed, nothing is going right today...",
  "timestamp": "2026-03-11T10:30:00Z"
}
```

### Response (Backend → Frontend)

```json
{
  "user_id": "user_001",
  "timestamp": "2026-03-11T10:30:05Z",
  "emotion": {
    "label": "sadness",
    "confidence": 0.87,
    "secondary_emotion": "anxiety"
  },
  "support_response": {
    "text": "It sounds like you're carrying a lot right now..."
  },
  "playlist": {
    "mood_category": "melancholic",
    "songs": [
      {
        "song_id": "spotify_001",
        "title": "Song Title",
        "artist": "Artist Name",
        "genre": "indie",
        "mood_tag": "melancholic",
        "spotify_url": "https://open.spotify.com/track/...",
        "cover_image": "https://..."
      }
    ],
    "total_songs": 5
  }
}
```

### Emotion Detection (Internal NLP Output)

```json
{
  "raw_text": "I'm so tired and stressed...",
  "preprocessed_text": "tired stressed nothing going right feel empty",
  "emotions": [
    { "label": "sadness", "score": 0.87 },
    { "label": "anxiety", "score": 0.65 }
  ],
  "dominant_emotion": "sadness",
  "valence": -0.75,
  "arousal": -0.4
}
```

### Mood-to-Music Mapping Config

```json
{
  "mood_mapping": [
    {
      "emotion_label": "sadness",
      "valence_range": [-1.0, -0.5],
      "arousal_range": [-1.0, 0.0],
      "mood_category": "melancholic",
      "recommended_genres": ["indie", "folk", "acoustic"],
      "tempo_range": "slow"
    },
    {
      "emotion_label": "anger",
      "valence_range": [-1.0, -0.3],
      "arousal_range": [0.5, 1.0],
      "mood_category": "intense",
      "recommended_genres": ["rock", "metal", "hip-hop"],
      "tempo_range": "fast"
    },
    {
      "emotion_label": "joy",
      "valence_range": [0.5, 1.0],
      "arousal_range": [0.3, 1.0],
      "mood_category": "euphoric",
      "recommended_genres": ["pop", "dance", "funk"],
      "tempo_range": "fast"
    },
    {
      "emotion_label": "anxiety",
      "valence_range": [-0.8, -0.2],
      "arousal_range": [0.3, 0.8],
      "mood_category": "tense",
      "recommended_genres": ["ambient", "lo-fi", "classical"],
      "tempo_range": "slow"
    },
    {
      "emotion_label": "calm",
      "valence_range": [0.2, 0.7],
      "arousal_range": [-1.0, 0.2],
      "mood_category": "peaceful",
      "recommended_genres": ["ambient", "jazz", "classical"],
      "tempo_range": "slow"
    }
  ]
}
```

### Error Response

```json
{
  "status": "error",
  "code": 422,
  "message": "Input text is too short. Please write at least 1 sentence.",
  "timestamp": "2026-03-11T10:30:05Z"
}
```

---

## 🔧 Tech Stack

| Layer           | Technology               | Notes                                  |
| --------------- | ------------------------ | -------------------------------------- |
| Frontend        | Vite React               | SPA, demo/prototype purpose            |
| Styling         | Tailwind CSS             | Utility-first, fast UI                 |
| HTTP Client     | Axios                    | API calls from frontend                |
| Backend         | Flask (Python)           | Beginner-friendly, sufficient for demo |
| NLP Models      | HuggingFace Transformers | Pre-trained, fine-tunable              |
| Emotion Model   | RoBERTa / IndoBERT       | Fine-tuned on GoEmotions               |
| Music API       | Spotify API              | Real-time playlist retrieval           |
| Database        | MongoDB or PostgreSQL    | Song-mood mapping storage              |
| Session Storage | Flask Session / MongoDB  | Chat history per user                  |

---

## 📊 Evaluation Plan

### NLP Model Evaluation (Person 1)

- Accuracy, Precision, Recall, F1-Score
- Confusion matrix across emotion labels
- Benchmark against baseline models

### System Evaluation (Person 3)

- **SUS (System Usability Scale)** — 10-question usability questionnaire
- **PANAS Scale** — Positive and Negative Affect Schedule, measure emotional response
- User satisfaction rating on playlist relevance
- Sample size: minimum 20-30 respondents recommended for MIND journal

---

## 📝 Paper Structure (Journal MIND)

1. **Abstract**
2. **Introduction** — problem, motivation, objectives
3. **Literature Review** — emotion detection, chatbot, music recommendation
4. **Methodology** — datasets, preprocessing, model architecture, system design
5. **Implementation** — system build, tech stack, interface
6. **Results & Evaluation** — model performance, user testing results
7. **Conclusion & Future Work**
8. **References**

---

## ⚠️ Important Constraints & Notes

- Frontend is **demo/prototype only** — not for production deployment
- Backend developer is **beginner level** — keep Flask implementation simple
- The "Hybrid" claim must be technically justified in the paper (retrieval + generative, or retrieval + rule-based)
- Plagiarism must be kept **below 20%**
- Paper must be written in **English**
- All three team members must have clear, equal contribution sections in the paper
- NLP pipeline quality is the **primary research contribution** — frontend aesthetics are secondary

---

## 🔗 Key References to Explore

- GoEmotions paper (Demszky et al., 2020)
- EmpatheticDialogues paper (Rashkin et al., 2019)
- RoBERTa paper (Liu et al., 2019)
- Valence-Arousal emotion model (Russell, 1980)
- PMEMO Dataset paper
- Journal MIND submission guidelines: ITENAS Bandung

---

## 🗂️ Model Output Folder Structure

When entering the backend phase (Phase 5), all trained models and precomputed files should be stored under a single `model_output/` directory at the root of the Flask project.

```
model_output/
├── emotion_detection/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
│
├── empathetic_retrieval/
│   ├── response_embeddings.npy
│   ├── response_texts.pkl
│   └── model_name.txt
│
└── mood_music/
    └── mood_mapping.json
```

### Loading Each Model in Flask

```python
# emotion detection
emotion_model = RobertaForSequenceClassification.from_pretrained("./model_output/emotion_detection")
emotion_tokenizer = RobertaTokenizer.from_pretrained("./model_output/emotion_detection")

# empathetic retrieval
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
response_embeddings = np.load("./model_output/empathetic_retrieval/response_embeddings.npy")
response_texts = pickle.load(open("./model_output/empathetic_retrieval/response_texts.pkl", "rb"))

# mood music
with open("./model_output/mood_music/mood_mapping.json") as f:
    mood_mapping = json.load(f)
```

### ⏱️ Expected Performance Per Component

| Component                   | Estimated Latency | Notes                   |
| --------------------------- | ----------------- | ----------------------- |
| Emotion Detection (RoBERTa) | ~200–400ms        | Heaviest step           |
| SBERT Encode (user input)   | ~50–100ms         | Single sentence         |
| Cosine Similarity (`.npy`)  | ~5–10ms           | In-memory, very fast    |
| Spotify API Call            | ~200–500ms        | Network dependent       |
| **Total Response Time**     | **~500ms–1s**     | Under normal conditions |

_Last updated: April 2026 — Generated as AI context document for project assistance_
