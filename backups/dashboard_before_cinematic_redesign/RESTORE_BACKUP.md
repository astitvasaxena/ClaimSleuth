# How to Restore the Original ClaimSleuth Dashboard

This document provides exact instructions to restore the ClaimSleuth dashboard to its state before the cinematic 3D redesign.

---

## 1. Backup Location and Contents

The backup is stored safely outside active template and static serving paths at:
`backups/dashboard_before_cinematic_redesign/`

Backed-up files:
- `index.html` (Original working single-page dashboard with standard 2-column layout, summary banner, scenario selection, adjudication actions, and collapsible technical details)
- `app.py` (FastAPI backend routes and template serving logic)

---

## 2. Restoration Instructions

### Option A: Using PowerShell (Windows)
Run this single command from the project root (`claim_sleuth` directory):

```powershell
Copy-Item "backups\dashboard_before_cinematic_redesign\index.html" "src\ui\templates\index.html" -Force
Copy-Item "backups\dashboard_before_cinematic_redesign\app.py" "src\ui\app.py" -Force
```

### Option B: Using Command Prompt / Bash
```bash
cp backups/dashboard_before_cinematic_redesign/index.html src/ui/templates/index.html
cp backups/dashboard_before_cinematic_redesign/app.py src/ui/app.py
```

### Option C: Manual File Explorer Copy
1. Open folder `backups\dashboard_before_cinematic_redesign\`.
2. Copy `index.html` and paste it into `src\ui\templates\index.html` (overwrite when prompted).
3. Copy `app.py` and paste it into `src\ui\app.py` (overwrite when prompted).

---

## 3. Verify Restoration

1. If the FastAPI dev server is running with `--reload` (e.g. `uvicorn app:app --reload`), refreshing `http://127.0.0.1:8000/` will immediately load the restored dashboard.
2. If restarting manually:
   ```powershell
   python src/ui/app.py
   ```
3. Run the automated test suite to confirm everything passes:
   ```powershell
   python -m pytest tests/ -v
   ```
