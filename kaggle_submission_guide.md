# Kaggle Notebook Submission Guide (Simple Flow)

This guide assumes you want to submit from a **Kaggle Notebook** (not local upload only).

## 1. Create notebook
1. Open competition page: `playground-series-s6e4`.
2. Go to **Code** tab.
3. Click **New Notebook**.
4. In notebook settings:
   - **Accelerator = GPU**
   - **Internet = Off** (recommended unless needed)

## 2. Add competition data
1. Right panel -> **Add data**.
2. Add the competition dataset (`playground-series-s6e4`).
3. Keep default mount path:
   - `/kaggle/input/playground-series-s6e4/`

## 3. Run prepared notebook
You now have a ready file in this repo:
- `kaggle_gpu_submission_workflow.ipynb`

Upload this notebook to Kaggle or copy its cells into your Kaggle notebook, then **Run All**.

Expected outputs:
- `/kaggle/working/submission_ready.csv`
- `/kaggle/working/model_report.json`
- optional zip: `/kaggle/working/submission_ready.zip`

## 4. Submit from notebook output
After run finishes:
1. Open the right panel -> **Output** files.
2. Click **Submit to Competition** next to `submission_ready.csv` (or upload zip in competition page).
3. Add description, for example:
   - `GPU CatBoost multi-seed ensemble (4 seeds), balanced class weights, 3-fold CV check, full-train inference.`
4. Confirm submission.

## 5. Check leaderboard + iterate
1. Go to **My Submissions**.
2. Compare score and keep notes by run:
   - seeds used
   - iterations
   - CV mean/std
3. Use daily limit wisely (10/day): test 2-4 meaningful variants, not random tweaks.

## Brand-building workflow
- Keep notebook clean and well documented (title, method, result).
- Publish notebook publicly after competition-safe timing.
- Link notebook + score screenshot in your GitHub and social posts.
- Emphasize: metric, rank movement, and engineering decisions.
