# Deploying PitchIQ (free)

The app only reads `artifacts/` and `reports/figures/`. Run the pipeline once
locally, commit those folders, and any host can serve the app.

## 0. First real run (on your laptop)

```bash
python -m pitchiq.pipeline all          # downloads data, tunes, backtests, writes artifacts/
git add artifacts reports
git commit -m "First real run"
git push
```
`artifacts/tuned_params.json` must be committed. The weekly refresh reuses those tuned parameters instead of re-tuning every week.

## 1. Live link: Streamlit Community Cloud (free, recommended)

1. Push the repo to GitHub (public).
2. Go to share.streamlit.io → **Create app** → choose the repo, branch `main`, main file `app/streamlit_app.py`.
3. Deploy. `app/requirements.txt` sits next to the entrypoint, so Community Cloud uses it (three light packages, fast builds) instead of the full pipeline requirements.
4. Each weekly commit from GitHub Actions triggers a redeploy automatically.

## 2. Weekly refresh: GitHub Actions

`.github/workflows/weekly-refresh.yml` runs every Tuesday 06:17 UTC (11:47 IST):
- restores cached raw data (completed seasons are never re-downloaded)
- `python -m pitchiq.pipeline refresh` → current-season data, features, leakage check, predictions, SHAP, simulation
- commits `artifacts/` and `reports/` if anything changed

Setup: repo **Settings → Actions → General → Workflow permissions → Read and write**. Run it once by hand from the **Actions** tab (`workflow_dispatch`) to confirm it works.

## 3. Docker / AWS (optional, if you want "deployed on AWS" on the resume)

```bash
docker build -t pitchiq-app .
docker run --rm -p 8501:8501 pitchiq-app      # http://localhost:8501
```

AWS's free tier changed for accounts created after 15 July 2025. New customers now get up to $200 in credits on a free plan that ends after 6 months or when the credits run out. Check your own account's Free Tier page before launching anything, and set a **billing alarm / budget** first.

Cheapest simple route, **one small EC2 instance running the container:**
1. Launch an Amazon Linux 2023 instance (smallest free-tier-eligible size shown for your account). Security group: inbound TCP 8501 from your IP (or 0.0.0.0/0 for a public demo), SSH from your IP only.
2. On the instance:
   ```bash
   sudo dnf install -y docker git && sudo systemctl enable --now docker
   git clone https://github.com/shrey200511/pitchiq.git && cd pitchiq
   sudo docker build -t pitchiq-app . && sudo docker run -d --restart unless-stopped -p 8501:8501 --name pitchiq pitchiq-app
   ```
3. Weekly pull of the refreshed artifacts (crontab, Tuesday 08:00 UTC, after the Action):
   ```
   0 8 * * 2 cd ~/pitchiq && git pull && sudo docker build -t pitchiq-app . && sudo docker rm -f pitchiq && sudo docker run -d --restart unless-stopped -p 8501:8501 --name pitchiq pitchiq-app
   ```
4. **Stop or terminate the instance** when you no longer need it.

Why not Lambda? Streamlit needs a long-lived WebSocket server, which Lambda doesn't provide. Lambda would suit a FastAPI JSON endpoint instead. That's a possible extension, not needed here.
