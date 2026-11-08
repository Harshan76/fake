# TruthLens – Production Deployment Guide
### Target Architecture: Vercel (Frontend) + Render (Backend) + PostgreSQL (Database)

This guide provides end-to-end instructions for deploying TruthLens to production with high availability, persistent relational storage, and secure cross-origin communication.

---

## 🏗️ Architecture Overview

```
                      ┌─────────────────────────────────┐
                      │        Vercel (Frontend)        │
                      │  https://truthlens.vercel.app   │
                      └────────────────┬────────────────┘
                                       │ HTTPS REST Calls
                                       ▼ (CORS Authenticated)
                      ┌─────────────────────────────────┐
                      │         Render (Backend)        │
                      │   https://truthlens.onrender.com│
                      │   FastAPI + Scikit-Learn Model  │
                      └────────────────┬────────────────┘
                                       │ Connection Pool (SSL)
                                       ▼
                      ┌─────────────────────────────────┐
                      │    Hosted PostgreSQL Database   │
                      │  (Render Postgres / Supabase)   │
                      └─────────────────────────────────┘
```

---

## 1. Prerequisites & Repository Preparation

Ensure your Git repository is pushed to GitHub:

```bash
git init
git add .
git commit -m "feat: production deployment preparation for Vercel, Render, and PostgreSQL"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/truthlens.git
git push -u origin main
```

---

## 2. Step 1: Provision the PostgreSQL Database

TruthLens uses a persistent relational PostgreSQL database for storing verification logs, forensic evidence trails, and source attribution links.

### Option A: Using Render Managed PostgreSQL (Recommended)
1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **PostgreSQL**.
3. Fill in the details:
   - **Name**: `truthlens-db`
   - **Database**: `truthlens`
   - **User**: `truthlens_user`
   - **Region**: Choose the region closest to your users (e.g., `Oregon (US West)` or `Frankfurt (EU)`).
   - **Plan**: `Free` (or `Starter`).
4. Click **Create Database**.
5. Once provisioned, scroll to the **Connections** panel and copy the **Internal Database URL** (if deploying backend on Render in the same region) or **External Database URL** (starts with `postgres://...`).

> **Note on URL Normalization & Driver Compatibility:** Render provides URLs beginning with `postgres://`. TruthLens automatically normalizes this to `postgresql+psycopg://` in `app/config.py` to use the modern `psycopg` (version 3, installed via `psycopg[binary]>=3.2,<4`) driver. Both `postgres://` and `postgresql://` inputs are automatically mapped to `postgresql+psycopg://` with zero manual intervention.

---

## 3. Step 2: Deploy the Backend to Render

1. On the Render Dashboard, click **New +** -> **Web Service**.
2. Connect your GitHub repository (`truthlens`).
3. Configure the service settings:
   - **Name**: `truthlens-backend`
   - **Region**: Same as your database (e.g., `Oregon`).
   - **Root Directory**: Leave blank (root `.`).
   - **Runtime**: `Python 3`.
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - **Plan**: `Free` (or `Starter`).

4. Add **Environment Variables** under the **Environment** tab:

   | Variable | Value | Purpose |
   | :--- | :--- | :--- |
   | `ENVIRONMENT` | `production` | Enables production security & logging |
   | `DATABASE_URL` | *(Paste your PostgreSQL URL from Step 1)* | Persistent DB connection |
   | `CORS_ORIGINS` | `*` *(Update to Vercel URL in Step 4)* | Allowed CORS domains |
   | `SEARCH_API_KEY` | *(Optional: Google/Bing/Tavily API key)* | Fallback search API |
   | `PYTHON_VERSION`| `3.10.14` | Explicit Python runtime |

5. Configure **Health Check Path**:
   - Expand **Advanced Settings**.
   - Set **Health Check Path** to `/health`.
6. Click **Create Web Service**.
7. Once deployed, note your Render backend URL (e.g., `https://truthlens-backend.onrender.com`).
8. Test the live health endpoint in your browser:
   `https://truthlens-backend.onrender.com/health` -> Should return `{"status":"healthy","database":"connected","model_loaded":true}`.

---

## 4. Step 3: Deploy the Frontend to Vercel

The frontend is a dedicated, production-optimized Vite Single Page Application configured in `frontend/`.

1. Log in to [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New...** -> **Project**.
3. Import your GitHub repository (`truthlens`).
4. In the configuration screen:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Click *Edit* and select **`frontend`**
   - **Build Command**: `npm run build` *(auto-detected)*
   - **Output Directory**: `dist` *(auto-detected)*
   - **Install Command**: `npm install` *(auto-detected)*

5. Expand **Environment Variables** and add:

   | Variable | Value | Example |
   | :--- | :--- | :--- |
   | `VITE_API_BASE_URL` | Your Render Backend URL | `https://truthlens-backend.onrender.com` |

   > **Important:** Do NOT add a trailing slash to `VITE_API_BASE_URL`.

6. Click **Deploy**.
7. Once deployment finishes, Vercel gives you your production URL (e.g., `https://truthlens.vercel.app`).

---

## 5. Step 4: Finalize CORS Security on Render

Now that your frontend has a live production domain on Vercel:

1. Return to your [Render Dashboard](https://dashboard.render.com).
2. Select your `truthlens-backend` service -> **Environment**.
3. Edit `CORS_ORIGINS` to restrict API access strictly to your deployed frontend:
   ```env
   CORS_ORIGINS=https://truthlens.vercel.app,http://localhost:5173
   ```
4. Click **Save Changes**. Render will automatically restart the backend with the new security boundaries.

---

## 6. Verification Checklist

Test each feature on the production Vercel URL:

- [ ] **Homepage & Design**: Renders with `#F5F0E5` cream background, `#080F1B` navy text, and `#B87927` gold accents.
- [ ] **Health Endpoint**: `https://YOUR-BACKEND.onrender.com/health` returns HTTP 200 with `database: connected`.
- [ ] **Text Verification**: Submit a news headline or claim.
- [ ] **URL Article Verification**: Enter a public news URL; headline, author, and body are extracted and analyzed with SSRF protection active.
- [ ] **Screenshot & Image OCR**: Upload a WhatsApp rumor or screenshot. Tesseract OCR extracts text for confirmation before verification.
- [ ] **Voice Claim**: Click the microphone on desktop Chrome/Edge/Safari (over HTTPS). Speech-to-text transcribes speech live into an editable box.
- [ ] **Document Extraction**: Upload `.pdf`, `.docx`, or `.txt` press releases.
- [ ] **Explore More Links**: Click external source links in verification results; open in new tabs.
- [ ] **History Archive**: Past analyses are stored in PostgreSQL and persist across server restarts.

---

## 7. Known Production Considerations

1. **Render Free Tier Spin-down**:
   Free web services on Render spin down after 15 minutes of inactivity. The first request after idle can take 30–50 seconds to warm up. For instant responses, upgrade to a **Starter** instance (\$7/month).
2. **Browser HTTPS for Voice Recognition**:
   The Web Speech API requires HTTPS in modern browsers (Chrome, Edge, Safari). Vercel provides automatic SSL certificates, ensuring microphone access works out-of-the-box.
3. **Database Idle Sockets**:
   PostgreSQL connection pooling is configured with `pool_pre_ping=True` and `pool_recycle=300` in `app/database.py` to prevent broken connections from cloud socket timeouts.
