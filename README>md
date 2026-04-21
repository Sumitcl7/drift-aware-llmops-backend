# Drift-Aware LLMOps Backend (FastAPI)

FastAPI backend for a multimodal LLMOps system with:
- text routing
- image/video embedding endpoints
- dashboard summary + chart APIs
- drift evaluation and retrain policy checks
- Supabase logging

## Tech Stack
- Python 3.10+
- FastAPI + Uvicorn
- Supabase (PostgreSQL)
- Embedding modules (image/video)

---

## Project Structure (important folders)

- `api/` → API routes (`llm_router.py`)
- `monitoring/` → dashboard + retrain logic
- `database/` → DB clients / integration
- `embedders/` → embedding modules

---

## 1. Setup (Local)

### Clone
```bash
git clone https://github.com/Sumitcl7/drift-aware-llmops-backend.git
cd drift-aware-llmops-backend
```

### Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Environment variables
Create `.env` from `.env.example`:

```bash
# Windows PowerShell
Copy-Item .env.example .env
# Linux/macOS
cp .env.example .env
```

Fill values in `.env`:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- optional policy vars (`DRIFT_THRESHOLD`, `MIN_SAMPLES`, etc.)

---

## 2. Run Backend

```bash
python -m uvicorn api.llm_router:app --host 127.0.0.1 --port 8000
```

Health test:
```bash
curl http://127.0.0.1:8000/health
```

---

## 3. Key Endpoints

- `GET /health`
- `POST /query`
- `POST /embed-image`
- `POST /embed-video`
- `GET /dashboard/summary`
- `GET /dashboard/charts`
- `GET /retrain/evaluate`
- `POST /retrain/run-now`

---

## 4. Deployment (AWS EC2 - simple path)

1. Launch EC2 (Ubuntu), open ports:
   - 22 (SSH)
   - 8000 (backend) or use reverse proxy with 80/443

2. Install Python + git, clone repo, set up venv, install requirements.

3. Add `.env` on server (never commit secrets).

4. Run with:
```bash
uvicorn api.llm_router:app --host 0.0.0.0 --port 8000
```

5. (Recommended) use `systemd` or `pm2` + Nginx reverse proxy for production.

---

## 5. Notes

- Ensure import paths use:
  - `from monitoring.dashboard_data import ...`
  - `from database.supabase_client import supabase`
- Add `__init__.py` in package folders if needed.
- Do not commit `.env`.

---

## License
For academic/demo use.
