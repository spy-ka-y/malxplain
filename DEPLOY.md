# Deploying the MalXplain dashboard

The dashboard is hosted free on Streamlit Community Cloud. It redeploys
automatically on every push, so the hosted link always reflects the current state
of the repository.

## One time setup

1. Push this repository to GitHub (see the commands your team was given).
2. Go to https://share.streamlit.io and sign in with the GitHub account that owns
   the repository.
3. Click "New app", then select:
   - Repository: your-username/malxplain
   - Branch: main
   - Main file path: app.py
4. Click Deploy. The first build takes roughly three to five minutes while the
   dependencies install.

You will get a permanent public URL of the form
https://your-app-name.streamlit.app

## What works on the hosted version

Every page except "Live detection" reads precomputed result files, so they load
instantly and require no training.

"Live detection" loads the dataset and trains the reference detector on first use,
which takes about sixty to ninety seconds, then caches it.

## What does not work on the hosted version, and why that is fine

The local language model runs through Ollama on a specific machine. A hosted
deployment has no Ollama server, so the analyst finding falls back to the
deterministic template. The app detects this and states which path it used.

This is expected behaviour, not a failure. The hosted dashboard is for showing
results and progress. The locally hosted language model is demonstrated live,
which is also the stronger demonstration, because the point of the design is that
nothing leaves the machine.

## Keeping the dataset available

The CIC-MalMem-2022 CSV is about 19 MB, which is well inside GitHub's limits, so
committing it is the simplest option and makes the deployment self contained.
If you prefer not to commit it, run scripts/setup.py locally and note that the
hosted build will need the file fetched at startup instead.
