# Git Structure Snapshot

This document captures the current Git repository layout, branches, remotes, and key paths for the RACEN workspaces. It is meant as a stable reference so we can quickly recover context later.

---

## 1. Repositories in this workspace

Within `d:\RAVENs\GREST\RACEN_LOCAL` there are **two Git repositories**:

1. **windsurf-racen** (main backend + web UI monorepo)
   - Root: `d:\RAVENs\GREST\RACEN_LOCAL\Windsurf_Project`
   - Remote:
     - `origin https://github.com/sam9s/windsurf-racen.git`

2. **Grest_RACEN_Slack_Bot** (Slack bot)
   - Root: `d:\RAVENs\GREST\RACEN_LOCAL\Grest_RACEN_Slack_Bot`
   - Remote:
     - `origin https://github.com/sam9s/Grest_RACEN_Slack_Bot.git`

The parent folder `d:\RAVENs\GREST\RACEN_LOCAL` itself is **not** a Git repo (no `.git` there).

---

## 2. Repository: windsurf-racen (Windsurf_Project)

**Root:** `d:\RAVENs\GREST\RACEN_LOCAL\Windsurf_Project`

### 2.1. Current branch & status

- `git status -sb`:
  - `## feature/product-search...origin/feature/product-search [ahead 2]`
  - Modified (tracked):
    - `docs/Progress_Report_2025-11-09.md`
    - `docs/TASK.md`
    - `src/racen/assistant_meta.py`
  - Untracked:
    - `docs/infrastructure.md` (infra snapshot we created)
    - `tests/Stress_Test/` (new tests folder)

So **HEAD** is on branch **`feature/product-search`**, ahead of `origin/feature/product-search` by 2 commits.

### 2.2. Branches

`git branch -vv` (simplified):

- `main`
  - Tracks: `origin/main`
  - Last seen commit: docs and housekeeping (`docs: link Slack bot repo in README; ...`).

- `feature/product-search` (**HEAD**)
  - Tracks: `origin/feature/product-search`
  - Status: ahead by 2 commits.
  - Recent focus: product search / meta Q&A / Web UI import.

- `backup-intent-routing-2025-11-20`
  - Purpose: backup of intent routing + end-to-end query tests.

- `backup-product-specs-2025-11-20`
  - Purpose: backup of product specs loader and blog/shipping/warranty flow tests.

- `pipeline-snapshot-2025-11-11`
  - Tracks: `origin/pipeline-snapshot-2025-11-11` (ahead 1).
  - Purpose: ingest + `/ingest/status` stage field snapshot.

- `pipeline-snapshot-2025-11-16`
  - Tracks: `origin/pipeline-snapshot-2025-11-16`.
  - Purpose: pipeline snapshot before product search work (iPhone ingest + prompt tweaks).

### 2.3. Recent commits (HEAD branch)

From `git log -n 10 --oneline --decorate --graph` (simplified, newest first):

- `2a90bcf (HEAD -> feature/product-search)`
  - `tests(meta): add assistant_meta_queries.txt and report runner; generate assistant_meta_results.md (24 queries, 24 OK)`
- `380bf52`
  - `feat(meta): deterministic assistant meta Q&A (who are you, creator, location, capabilities, privacy, human/bot, language) with early handling in answer flow; add unit tests`
- `f8a42ce (origin/feature/product-search)`
  - `chore(webui): import RACEN WebUI into monorepo under racen-webui; mark TASK complete`
- `d8a60c2`
  - `backup: 51-query comprehensive report green; classifier-driven buying_advice routing; add 51-report scripts and output`
- `c8a0389`
  - `Backup before product family price Slack tweaks`
- `10b1757`
  - `Backup before product family price fallback logic`
- `ad46a55`
  - `backup: remove obsolete grest_iphone_missing mapping`
- `4f4c2f3`
  - `backup: web comparison routing & follow-up suppression stable`
- `3a235f4`
  - `Phase 1.1: Trustpilot+Mouthshut brand reputation, tests, and env docs`
- `37bef17 (backup-intent-routing-2025-11-20)`
  - `backup: intent routing + e2e query tests`

This confirms that **product search**, **meta Q&A**, **51-query reports**, and **Web UI import** live on `feature/product-search`.

### 2.4. Top-level tracked paths

From `git ls-tree --name-only HEAD`:

- `.gitignore`
- `Grest_Data` (data assets)
- `README.md`
- `docs` (project docs, including `TASK.md`, `Progress_Report_*.md`, `infrastructure.md`)
- `domain_product_families.yaml`
- `markitdown-main` (vendored/linked repo content, not a Git submodule here)
- `ottomator-agents-main` (vendored/linked content)
- `outputs` (reports, benchmarks)
- `racen-webui` (Next.js Web UI imported into monorepo)
- `requirements.txt`
- `scripts` (Answer API, step pipeline, test runners)
- `src` (RACEN core Python modules)
- `tests` (pytest + query runners)
- `windsurf-racen-local` (local compose and tooling)

There are **no active submodules** in this repo (current `git submodule status` output is empty).

---

## 3. Repository: Grest_RACEN_Slack_Bot

**Root:** `d:\RAVENs\GREST\RACEN_LOCAL\Grest_RACEN_Slack_Bot`

### 3.1. Current branch & status

- `git status -sb`:
  - `## pipeline-snapshot-2025-11-16...origin/pipeline-snapshot-2025-11-16`
  - Modified (tracked):
    - `slack-openai-bot/app.js` (this includes our "Thinking…" UX changes).

So **HEAD** is on branch **`pipeline-snapshot-2025-11-16`**, tracking `origin/pipeline-snapshot-2025-11-16`.

### 3.2. Branches

From `git branch -vv` (simplified):

- `main`
  - Tracks: `origin/main`
  - Latest tagged work:
    - Tag `racen-ingest-progress-2025-11-16-bot`: "Slack ingest UX: DM stage polling, stage pings, final summary, end-only unfurl, divider, and duplicate suppression".

- `pipeline-snapshot-2025-11-16` (**HEAD**)
  - Tracks: `origin/pipeline-snapshot-2025-11-16`.
  - Purpose: snapshot with Slack bot env path & routing stable (Answer API wiring, allowlists, ingest UX).

### 3.3. Recent commits (HEAD and main)

From `git log -n 10 --oneline --decorate --graph` (simplified, newest first):

- `c619849 (HEAD -> pipeline-snapshot-2025-11-16, origin/pipeline-snapshot-2025-11-16)`
  - `backup: slack bot env path & routing stable`
- `b1140ea`
  - `Slack: ingest allowlist 403 handling + not-authorized DM`
- `b98085f (tag: racen-ingest-progress-2025-11-16-bot, main)`
  - `Slack ingest UX: DM stage polling, stage pings, final summary, end-only unfurl, divider, and duplicate suppression`
- `dd63c83 (tag: snapshot-2025-11-16-1631, origin/main, origin/HEAD)`
  - `Slack: clean product URL unfurl; product URL allowlist override (QA-only)`
- `d343003 (tag: snapshot-2025-11-16-1212)`
  - `Slack bot: escalate on 3rd fallback; replace fallback body with concise support block; keep context across user+channel (10m)`
- `6c713a9`
  - `persona: change phrasing to 'senior' in system_prompt.md and lexicon.v1.yaml`
- `afeab32`
  - `init: Slack bot runtime with presets and Answer API integration (no secrets)`

Our **current local modifications** to `slack-openai-bot/app.js` (Thinking… indicator, link conversion, etc.) are on top of this.

### 3.4. Top-level tracked paths

From `git ls-tree --name-only HEAD`:

- `.gitignore`
- `package-lock.json`
- `slack-openai-bot` (main bot code, env loader, persona files)
- `windsurf-racen-local` (local tooling/compose for Slack repo)

No active submodules are reported here either.

---

## 4. Summary: where things live in Git

- **Backend + Web UI (monorepo)**
  - Repo: `windsurf-racen` at `Windsurf_Project`.
  - Active branch: `feature/product-search` (HEAD, ahead of origin by 2).
  - Contains:
    - FastAPI Answer API, pipeline steps, tests.
    - Docs: `TASK.md`, progress reports, `infrastructure.md`, this `git_structure.md`.
    - Web UI: `racen-webui` (Next.js) imported into monorepo.

- **Slack Bot**
  - Repo: `Grest_RACEN_Slack_Bot`.
  - Active branch: `pipeline-snapshot-2025-11-16` (HEAD, aligned with origin).
  - Contains:
    - `slack-openai-bot/app.js` (Slack event handling, Answer API calls, Thinking… UX, ingest UX).
    - `windsurf-racen-local` for local support tooling.

All of this is backed up to GitHub under the two remotes listed above. This document should be treated as the canonical Git layout snapshot so future work does not rely on memory for branches or repo boundaries.
