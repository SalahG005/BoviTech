# Bovitech — Multimodal Smart Cattle Monitoring

**Bovitech** is a research-grade, multimodal decision-support stack for intelligent cattle monitoring. It combines **wearable IMU signals**, **barn environment**, **acoustics**, and **herd context** to support behavior understanding, stress awareness, production tracking, and early health signals. The system mixes classical ML, deep learning, and reinforcement learning depending on the modality.

This project was developed as part of an academic program at **[ESPRIT School of Engineering](https://esprit.tn/)** (Tunisia).

---

## Mobile app screenshots

Operator UI built with **Expo / React Native** (`main-bovitech-main/`). Flow: onboarding → herd dashboard → cow detail → vocal AI analysis.

### Welcome & authentication

<p align="center">
  <img src="docs/screenshots/01-welcome.png" alt="BoviTech welcome screen — Surveille, Analyse, Optimise" width="260" />
  &nbsp;&nbsp;
  <img src="docs/screenshots/02-login.png" alt="BoviTech login screen" width="260" />
</p>

<p align="center"><sub>Landing · Sign in</sub></p>

### Home dashboard

<p align="center">
  <img src="docs/screenshots/03-home-overview.png" alt="Home — weather, herd KPIs, alerts, THI and health index" width="260" />
  &nbsp;&nbsp;
  <img src="docs/screenshots/04-home-production.png" alt="Home — milk production chart over 7 days" width="260" />
</p>

<p align="center"><sub>Weather & KPIs · Milk production (7 days)</sub></p>

### Herd management

<p align="center">
  <img src="docs/screenshots/05-herd.png" alt="My Herd — searchable cow list with status filters" width="260" />
  &nbsp;&nbsp;
  <img src="docs/screenshots/06-add-cow.png" alt="Add cow form with photo, breed, sex and auto IDs" width="260" />
</p>

<p align="center"><sub>Herd list & filters · Register a cow</sub></p>

### Cow detail & vocal AI

<p align="center">
  <img src="docs/screenshots/07-cow-detail-summary.png" alt="Cow detail — summary cards and Parle-moi entry" width="220" />
  &nbsp;
  <img src="docs/screenshots/08-cow-detail-activity.png" alt="Cow detail — behavioral activity curve" width="220" />
  &nbsp;
  <img src="docs/screenshots/09-parle-moi-vocal.png" alt="Parle-moi — vocal recording and AI classification (cough, normal, ovulation, hunger)" width="220" />
</p>

<p align="center"><sub>Health summary · Activity curve · <b>Parle-moi</b> vocal classification (API :8008)</sub></p>

---

## Why this repository matters

- **End-to-end scope**: from tabular sensor pipelines and trained artifacts to a **mobile operator UI** and a **lightweight HTTP inference service** usable on a farm PC or edge device.
- **Multimodal by design**: behavior, milk, stress, vocalization, and fused “illness / early-warning” states are wired behind a single, documented API surface.
- **Explainability where it counts**: the illness pathway integrates SHAP-style explanations so predictions are reviewable, not black-box alerts.

---

## Capabilities

| Area | What it does | Typical technique |
|------|----------------|-------------------|
| **Behavior** | Second-level activity classes from collar (and fused) sensor features | Random Forest (multimodal-ready pipeline) |
| **Acoustics** | Bovine vocalization classification from recorded audio | CNN on Mel-spectrogram–like input (Keras `.h5`) |
| **Stress** | Risk classes from THI, core body temperature, lying/rest proxies | PyTorch sequence model (`StressDetectionV3`) |
| **Milk** | Daily yield estimate from behavior + history-style features | XGBoost + sklearn preprocessing pipeline |
| **Health / early warning** | Three-way state: healthy, at-risk, ill — plus temporal health score | PPO (Stable-Baselines3) + SHAP explanations |
| **Simulation / demo** | Tick-based simulation and barn sensor ingest for demos | Built into the inference server |

Behavior classes supported in production mapping include: *Walking, Standing, Feeding (head up / down), Licking, Drinking, Lying* (see `BEHAVIOR_LABELS` in `src/model_http_api.py`).

---


## Model cards

Each figure is the card for one Bovitech model. Numbers on the cards match the unified pipeline below.

### Behavior

<p align="center">
  <img src="docs/readme/behaviour.jpg" alt="Behavior model — Random Forest on collar IMU signals, 7 classes, 82.6 percent accuracy" width="520" />
</p>

A Random Forest reads **124 collar features** (accelerometer, magnetometer, and head) and labels activity **every second** across **7 classes**. Accuracy is **82.6%**, and macro F1 is **0.673**. The share of lying behaviour is forwarded to the milk model.

### Milk yield

<p align="center">
  <img src="docs/readme/milk-yield.jpg" alt="Milk yield model — XGBoost, R squared 0.95, MAPE 2.58 percent" width="520" />
</p>

An XGBoost regressor estimates daily yield in kilograms from **18 features**: behaviour, body temperature, the temperature-humidity index (THI), days in milk, and milk history. **R² is 0.95**, **MAPE is 2.6%** (2.58% on the card), and test RMSE is **1.37 kg**.

### Stress

<p align="center">
  <img src="docs/readme/stress.jpg" alt="Stress model — BiLSTM with cross-attention, Normal, At-risk, and Stressed" width="520" />
</p>

A **BiLSTM with cross-attention** fuses three signals — the THI wave, neck temperature, and lying behaviour — into **Normal**, **At-risk**, or **Stressed**. Accuracy is **86.8%** over 3 classes, macro F1 is **0.665**, and the card marks about a **2-hour early warning**.

### Illness

<p align="center">
  <img src="docs/readme/illness.jpg" alt="Illness model — PPO reinforcement learning with SHAP, Healthy, At-risk, and Ill" width="520" />
</p>

A **PPO** reinforcement-learning agent classifies **Healthy**, **At-risk**, or **Ill** from a **34-dimensional** state built from behaviour, milk yield, stress, core body temperature, and the barn environment. Training uses a class-weighted loss. The card reports **96.9%** accuracy, and **SHAP** keeps each alert explainable.

### Acoustics

<p align="center">
  <img src="docs/readme/audio.jpg" alt="Audio model — CNN on a mel-spectrogram, Cough, Estrus, Food, and Normal" width="520" />
</p>

A **CNN** classifies a **128×128 mel-spectrogram** (librosa) into **Cough**, **Estrus**, **Food**, or **Normal**. The pipeline diagram also places a sklearn baseline next to this CNN, with a 4-class softmax.

### Herd tracking

<p align="center">
  <img src="docs/readme/gps-tracking.jpg" alt="Herd tracking — LSTM on GPS trajectories with a zone alert" width="520" />
</p>

An **LSTM** follows the GPS trajectory, detects route deviation, and raises a real-time zone alert. This tracker is the seventh model on the project cards. It sits beside the six-model pipeline rather than inside one of its boxes.

### Chatbot

<p align="center">
  <img src="docs/readme/chatbot.jpg" alt="Chatbot — Vet, Feed, Meteo, and Vision agents on the mobile app" width="520" />
</p>

The assistant on the phone joins an **LLM + RAG** branch (farmer questions and alert summaries) with a **vision** branch that reads a behaviour image. Vision validation accuracy is **87.17%**.

---


## High-level architecture

```mermaid
flowchart LR
  subgraph client["Operator devices"]
    A["Expo / React Native app\n(main-bovitech-main)"]
  end
  subgraph inference["ML inference service"]
    B["Python HTTP API\nsrc/model_http_api.py :8008"]
  end
  subgraph models["Trained artifacts"]
    C["finale_model/\n.joblib / .pt / .h5 / SB3 zip"]
  end
  subgraph optional["Optional backends"]
    D["PI_Backend\nDjango + DRF"]
    E["bovitech-chatbot-main\nDjango + agents"]
  end
  A -->|REST JSON| B
  B --> models
  A -.->|optional| D
  A -.->|optional| E
```


## Unified pipeline

<p align="center">
  <img src="docs/readme/unified-pipeline.jpg" alt="BoviTech unified AI pipeline, version 3.0, from sensors to the mobile app" width="640" />
</p>

This is the v3.0 architecture: **6 AI models**, one shared backend, a chatbot, and the React Native app. Solid arrows are the primary path. Dashed arrows are secondary cross-feeds.

**1. IoT sensor layer.** Six inputs enter together: collar IMU (accelerometer, magnetometer, head), core body temperature, barn THI from temperature and humidity, audio clips (`.wav`, 4 classes), behaviour labels (7 classes, 1 second), and milk records (daily kg and days in milk).

**2. Core models.** Four models run on those signals.

| Model | Method | What it reports |
|---|---|---|
| Behaviour | Random Forest | 124 features · accuracy 82.6% · macro F1 0.673 |
| Milk yield | XGBoost regressor | 18 features · R² 0.95 · MAPE 2.6% |
| Stress | BiLSTM + attention | 3 classes · accuracy 86.8% · macro F1 0.665 |
| Audio | CNN + sklearn baseline | 4-class softmax · Cough, Estrus, Food, Normal |

Behaviour sends two cross-feeds: the behaviour proportion into milk yield, and lying behaviour into stress.

**3. Composite models.** These do not read raw sensors on their own. They wait for the core models.

- **Estrus detection** is an XGBoost classifier. It needs behaviour, milk, stress, and THI, and it looks for mounting or agitation, a milk deviation, and a time-window signal. The output is binary, and an alert fires only above a threshold.
- **Illness prediction** is a reinforcement-learning agent. It needs behaviour, milk, stress, and core body temperature, packed into a 34-dimensional state. A class-weighted loss separates **Healthy**, **At-risk**, and **Ill**.

**4. Shared backend API.** One REST API stores inference and metadata **per cow** and **per day** for behaviour, milk, stress, estrus, illness, and audio.

**5. Chatbot.** Two branches read that API. Branch A is NLP / LLM: farmer questions, alert summaries, a JSON contract, and streaming text. Branch B is vision: a behaviour image, with validation accuracy **87.17%**.

**6. React Native app (iOS and Android).** The same API fills four screens: a dashboard of health scores, alerts for illness, estrus, and stress, a cow profile with milk and behaviour trends, and a chat screen. The app adds push notifications, offline support, and separate vet and farmer roles.

An online feedback loop returns app activity toward the models, and dashboard data can also be read directly from the API.

---

## Repository layout

| Path | Role |
|------|------|
| `src/model_http_api.py` | **Main inference server** (stdlib `ThreadingHTTPServer`): `/health`, `/predict/*`, `/simulate/tick`, `/barn_sensor` |
| `src/train_model.py`, `src/predict_behavior.py`, `src/pipeline_utils.py` | IMMU → per-second features → train / predict behavior |
| `src/stress_v3_model.py`, `stress_sensor/` | Stress model training / evaluation tooling |
| `src/vocal_preprocess.py` | Audio decoding (incl. ffmpeg), features for vocal CNN |
| `src/build_rl_state.py`, `src/illness/` | Fused state vector + PPO policy + SHAP explainer + health score |
| `src/dashboard_app.py` | Streamlit-style analytics entry point (when used) |
| `finale_model/` | Checkpoints and pipelines expected at runtime (see below) |
| `main-bovitech-main/` | **Bovitech mobile app** (Expo SDK ~54, React Native) |
| `PI_Backend/` | Optional **Django** REST API (accounts, farms, cows) |
| `bovitech-chatbot-main/` | Optional **Django** chatbot backend (e.g. vet / feed assistants) |

---

## Requirements

### Inference + training (Python)

- **Python** ≥ 3.9 (3.10+ recommended)
- Core dependencies are listed in **`requirements.txt`**, including: `pandas`, `numpy`, `scikit-learn`, `joblib`, `torch`, `tensorflow>=2.15`, `librosa`, `soundfile`, `streamlit`, `plotly`, `stable-baselines3`, `gymnasium`.

**System packages (recommended):**

- **ffmpeg** — for robust audio decode in vocal classification (Windows: install ffmpeg and/or set `FFMPEG_PATH` to `ffmpeg.exe`).
- **libsndfile** — usually pulled in via `soundfile` wheels; on minimal Linux images you may need the system library.

**Optional (illness explainability):**

- `shap` — required if you use the illness explainer paths (`src/illness/illness_xai.py`). Install with `pip install shap` if not already present in your environment.

### Mobile app

- **Node.js** 18+
- **npm** or **yarn**
- **Expo CLI** (via `npx expo`)

### Optional Django services

- **`PI_Backend/requirements.txt`**: Django LTS 5.2+, Django REST framework, JWT, CORS.

> **Note on naming:** this monorepo uses a **built-in Python HTTP server** for the ML API (not FastAPI). Optional Django apps cover persistence and chat. If you migrate to FastAPI or Supabase, treat that as a deployment choice — the current code paths are as documented above.

---

## Installation

```bash
git clone https://github.com/SalahG005/BoviTech.git
cd BoviTech

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
# Optional: pip install shap
```

### Model artifacts (`finale_model/`)

The server expects trained files under `finale_model/` (and paths can be overridden with environment variables). Examples referenced in code:

- Behavior: `behavior_rf_multimodal.joblib`
- Milk: `milk_xgb_pred_behavior_daily_milkhist_pipeline.joblib`
- Stress: `StressDetectionV3_trained.pt` (or set `STRESS_V3_CHECKPOINT`)
- Vocal: Keras `.h5` (default discovery under `finale_model/` or set `VOCAL_MODEL_PATH`)
- Illness PPO: SB3-export zip (set `ILLNESS_PPO_PATH` if non-default)

If you clone without large binaries, obtain checkpoints from your team’s artifact store or retrain using the scripts under `src/` and `stress_sensor/`.

---

## Running the stack

### 1) ML inference API (required for on-device predictions)

From repository root:

```bash
python src/model_http_api.py
```

Default bind: **`http://0.0.0.0:8008`**.

Smoke test:

```bash
curl http://localhost:8008/health
```

### 2) Mobile app (`main-bovitech-main/`)

```bash
cd main-bovitech-main
npm install
npx expo start
```

**Point the app at your API** (see `main-bovitech-main/src/config/api.js`):

| Context | Default `API_BASE_URL` |
|--------|-------------------------|
| Android emulator | `http://10.0.2.2:8008` |
| iOS simulator / web | `http://localhost:8008` |
| Physical device | Set `EXPO_PUBLIC_API_BASE_URL` to `http://<your-pc-lan-ip>:8008` |

Optional timeouts (ms) for slow CPU inference:

- `EXPO_PUBLIC_API_HTTP_TIMEOUT_MS`
- `EXPO_PUBLIC_API_VOCAL_TIMEOUT_MS`

Optional chatbot backend URL:

- `EXPO_PUBLIC_CHATBOT_BASE_URL` (defaults to port **8000** for the Django chatbot project)

### 3) Optional: `PI_Backend` (Django)

```bash
cd PI_Backend
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

### 4) Optional: `bovitech-chatbot-main`

Follow that project’s `.env` / `SECRET_KEY` / `GROQ_API_KEY` setup in its `backend` app.

---

## HTTP API surface (inference server)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness and dependency check |
| `POST` | `/predict/behavior` | Behavior class from sensor payload |
| `POST` | `/predict/milk` | Milk yield prediction |
| `POST` | `/predict/stress` | Stress classes (THI, CBT, lying / behavior context) |
| `POST` | `/predict/vocal` | Vocal classification (`audio_wav_base64`) |
| `POST` | `/predict/illness` | PPO + temporal health score + optional SHAP narrative |
| `GET` | `/simulate/tick` | Demo / simulation tick |
| `POST` | `/barn_sensor` | Ingest barn environment samples (temperature / humidity → THI) |

Request/response shapes are defined by `RequestHandler` in `src/model_http_api.py` and mirrored in `main-bovitech-main/src/services/predictionApi.js`.

---

## Environment variables (selected)

| Variable | Purpose |
|----------|---------|
| `STRESS_V3_CHECKPOINT` | Override path to `StressDetectionV3` `.pt` |
| `VOCAL_MODEL_PATH` | Override path to vocal `.h5` |
| `VOCAL_SAMPLE_RATE`, `VOCAL_CLASS_ORDER`, `VOCAL_FFMPEG_TIMEOUT_SEC` | Vocal preprocessing / labeling order / decode timeout |
| `FFMPEG_PATH` | Explicit path to `ffmpeg` on Windows or constrained PATH |
| `ILLNESS_PPO_PATH` | Override path to illness PPO export |

---

## Training and batch workflows

Behavior pipeline (IMMU → per-second dataset → RandomForest):

```bash
python src/build_dataset.py --sensor-root sensor_data/sensor_data --cow C01 --date 0725 --include-mag --output-csv artifacts/datasets/dataset_C01_0725.csv
python src/train_model.py --sensor-root sensor_data/sensor_data --cows C01 C02 --dates 0725 --include-mag --out-dir artifacts/model
python src/predict_behavior.py --immu-file path/to/T01_0725.csv --model-dir artifacts/model --output-csv artifacts/predictions/out.csv
```

> Raw datasets are **not** committed by default (see `.gitignore` for `sensor_data/`, `artifacts/`). Place data locally or wire your own storage.

Illness / RL details: `src/illness/README.md`.

---

## Frontend & backend (summary)

| Layer | Technology |
|-------|------------|
| **Mobile UX** | React Native, **Expo**, React Navigation, charts, maps, i18n |
| **Inference** | Python 3, NumPy/Pandas/sklearn, PyTorch, TensorFlow/Keras, Stable-Baselines3 |
| **Optional services** | Django + DRF (`PI_Backend`, `bovitech-chatbot-main`) |

---

## Roadmap ideas

- Hardening deployment: container images, health metrics, HTTPS reverse proxy.
- Optional migration of `model_http_api.py` to **FastAPI** + OpenAPI for generated clients.
- Centralized persistence (**PostgreSQL** / **Supabase**) for telemetry and audit trails.
- On-device or edge **TFLite** paths for low-latency vocal and behavior models.

---

## License

Specify your license here (e.g. MIT, Apache-2.0, or academic / internal-only). If this is a school project, state restrictions from ESPRIT or your team’s policy.

---

## Citation / academic use

If you reuse this work academically, please cite the **ESPRIT** program and link this repository once public.
