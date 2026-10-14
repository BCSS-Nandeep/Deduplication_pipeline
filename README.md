# DOPAMS-CCTNS identity deduplication pipeline

Application code only. The PostgreSQL database is a **read-only source**: nothing in this project creates,
alters or writes database objects or rows. Outputs, logs and checkpoints are local files under `output/`.

## Phase 2: image resolution

Reads `persons_unified` and `media_unified` (joined on `parent_entity_type = 'person'` and
`parent_entity_id = person_id`) from `dopams_cctns_v2`, locates each file on the media file server, and
writes a manifest saying which images are usable for face recognition and why the others are not.
It does not compare faces, score, or generate candidates.

```powershell
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m dedup_pipeline.image_resolution --help

# small trial
.\.venv\Scripts\python -m dedup_pipeline.image_resolution --output-dir output\trial --source-system V2 --only-with-media --limit-persons 200
# full run (downloads every VERIFIED_ACCESSIBLE person image once; resumable)
.\.venv\Scripts\python -m dedup_pipeline.image_resolution
.\.venv\Scripts\python -m pytest
```

Connection settings: `DEDUP_PG_HOST/PORT/USER/PASSWORD` (preferred, e.g. a read-only role), otherwise
`UNIFIED_PG_*` from `--env-file`, `DEDUP_ENV_FILE`, or `../Dopams_data_base/.env`. The database is pinned to
`dopams_cctns_v2`. `FILE_SERVER_BASE_URL` overrides the file-server address.

### How a file is resolved

* Path: `resolved_relative_path`, else `source_local_path`, else `original_attach_path` (as DOPAMS-BE does).
* URL and extension fallback mirror `DOPAMS-BE/src/routes/fileProxy.ts`: as stored, then
  `.jpg .jpeg .png .webp .pdf .doc .docx` when the path has no extension.
* File type comes from the file signature only (`mime_type` is NULL in the database and the server
  always answers `application/octet-stream`). JPEG, PNG and WebP are fully decoded; PDFs are skipped.
* Only `VERIFIED_ACCESSIBLE` records are fetched unless `--probe-unverified` is given.

### Phase 6: scoring and human review (`dedup_pipeline/scoring/`)

Scores every Phase 5 candidate from the identity fields (name, alias, relative name, phone, age/DOB, gender, address)
and face evidence, and builds the package a human verifier works from. **It never decides anything**: every candidate
leaves as `pending`, nothing is merged, and nothing is written to any database. Read-only database access, local files only.

```powershell
$env:FR_DOPAMS_BASE_URL = "http://<host>:<port>"; $env:FR_DOPAMS_API_KEY = "<key>"
.\.venv\Scripts\python -m dedup_pipeline.scoring run                     # -> output/scoring/runs/<timestamp>/
.\.venv\Scripts\python -m dedup_pipeline.scoring validate-worksheet <completed review_worksheet.csv>
```

`match_score = sum(weight x field_score) / sum(applicable weights)` and `evidence_coverage = applicable weight / total
weight` (rule version `score-v1`, weights in `ScoringConfig`, saved with every run). Missing, skipped or failed evidence
is excluded from both, never scored as 0. Face evidence enters through a documented piecewise-linear map of the raw cosine
(0.20 -> 0, 0.58 -> 1) whose anchors come from the FR-DOPAMS README calibration; the raw cosine (-1..1) is always kept.
The score is **not a probability** and is not calibrated. Inputs are validated first (Phase 5 and Phase 2 completion
markers, counts, ordering, database stamp); a partial image run is reported as a limitation. Face comparisons are cached
in `output/scoring/face_evidence_cache.jsonl` (reruns and re-weighting make no new service calls); service failures make
the run `incomplete` and are retried on the next run. Each run gets its own directory and earlier runs are never overwritten.

**Rule versions.** `score-v2` (default) evaluates every age at one explicit `--reference-date` (recorded in the config, summary and
report; if omitted, today's date is used once and recorded), compares DOBs as dates, estimates birth-year ranges from recorded ages
(an age of 30 at a 2026 reference date means born 1995 or 1996; estimates are capped below an exact DOB match), flags records whose age
contradicts their own DOB, and reviews candidates in groups (no hard conflict, hard conflict, insufficient evidence, technical
failure; score order inside each). It also ranks face-comparison images by the face service's own earlier results. `score-v1`
(`--rule-version score-v1`) reproduces the earlier scoring byte for byte; runs of different versions are never mixed or resumed
into each other, and scores of different rule versions must not be compared as if equivalent.

`score-v3` (`--rule-version score-v3 --reference-date YYYY-MM-DD`, selected explicitly; not the default until validated) anchors every
recorded age to the **FIR date** it is linked to (`persons_unified.age` + `accused_unified.person_id -> crimes_unified.fir_date`, read
read-only and converted to the `Asia/Kolkata` calendar date). The possible birth years follow from age and FIR date, so age 26 at a 2022 FIR
and age 30 at a 2026 FIR are consistent. An age with no usable FIR date (no link, missing or invalid date) is *unavailable evidence*; the
current date is never substituted. Several FIRs for one record are kept as separate observations with provenance and never resolved by
picking the newest or oldest; contradicting ones are flagged for human review. DOB comparisons stay separate from age estimates. Thresholds
(`birth_year_tolerance_years`, `birth_year_conflict_gap_years`, `observation_conflict_policy`, `fir_timezone`, FIR validity bounds) are
configuration. Use a separate output root (`--output-root output/scoring_v3`) so earlier runs stay untouched. `tools/compare_v2_v3.py`
compares a v2 and a v3 run and `tools/phase8_verify.py` independently re-derives the v3 age evidence. Extra v3 outputs per run:
`age_evidence_provenance.jsonl`, `age_evidence_summary.json`.

**Performance (Phase 9).** Scoring results are unchanged (byte-identical outputs for score-v1/v2/v3) and a run is about 1.3x faster: name
similarity is memoised and skips matching that cannot reach its floor, the worksheet and the HTML report are written in one pass over the
scored file, and the image manifest is parsed once. `--reuse-known-image-outcomes` (opt-in, off by default) skips face-service calls whose
rejection the stored results already determine exactly; it assumes image files have not changed since they were first judged. Measurements,
equivalence results and remaining opportunities: `output/audit/phase9/PHASE_9_PERFORMANCE_REPORT.md`; reproduce with `tools/phase9_*.py`.

### Face engine setup (Phase 10 / 10.1)

Face comparison runs **in-process** (`LocalFacePairComparer`: the stateless FR-DOPAMS comparison logic, no separate service). It is the only
face engine of the scoring command: there is no HTTP/API engine option and no fallback. The default rule version is
still `score-v2`; the face engine does not change any score, weight, threshold, candidate or review order.

**Setup (CPU, the validated configuration).** Use the dedicated environment; do not install these packages into `.venv` or `.fr-venv`:

    python -m venv .face-venv
    .face-venv\Scripts\python -m pip install -r requirements.txt -r requirements-face-local.txt
    .face-venv\Scripts\python -m dedup_pipeline.scoring face-preflight      # prints the engine identity, or says exactly what is missing

The model files `det_10g.onnx` and `w600k_r50.onnx` must exist in `~/.insightface/models/buffalo_l` (nothing is downloaded automatically).

**Running.** `.face-venv\Scripts\python -m dedup_pipeline.scoring run --reference-date YYYY-MM-DD --output-root output\scoring_<name>` (CPU; the model is
loaded once per process). Write new runs to a new `--output-root`; completed runs are never overwritten.

**Preflight.** Before any output directory or database connection is created, the run checks the Python packages, the model files, the provider
and that the model loads. Any problem stops the run with exit code 2 and one message listing every problem. There is no fallback to any other
engine or provider. `--no-face-service` needs no engine at all.

**Cache behaviour.** Stored face results carry a fingerprint (engine version, every setting, SHA-256 of the model files, versions of insightface /
onnxruntime / numpy / opencv / scikit-image / scipy / Pillow, and the provider actually used). Results are reused only when the fingerprint is
identical. Results made by any other engine (for example the earlier HTTP service) are never used, and a changed library, setting or model file means
recomputation, never a silent mix. The preflight output lists any installed library that differs from the validated set.

**GPU: not validated.** Only CPU has been replayed against the 525 stored real-service cases. A GPU provider (`--face-provider auto|cuda`) is
refused unless `--accept-unvalidated-provider` is given, and a GPU deployment needs its own replay first:
`.face-venv\Scripts\python tools\phase10_replay.py --out <dir> --provider cuda` on that hardware (requires `onnxruntime-gpu` and a CUDA setup), with
exact agreement before enabling it. GPU results have a different fingerprint and are never mixed with CPU results.

**Performance (Phase 11).** The local face engine skips the recognition pass the library runs on every detected face but the comparer never uses, keeps each image's answer (embedding or exact rejection) in memory for the run, computes an image once even when several pairs ask for it at the same moment, and the default is 8 pairs in flight (`face_concurrency`; 4 -> 8 gives 1.6x for about +275 MB). All outputs are identical to the earlier code (525-case replay and 24,155-candidate comparison); a fresh scoring run takes about 145 s instead of 265 s on the reference machine. Report: `output/audit/phase11/PHASE_11_PERFORMANCE_REPORT.md`.

Reports: `output/audit/phase10/PHASE_10_LOCAL_FACE_COMPARER_REPORT.md`, `output/audit/phase10_1/PHASE_10_1_ROLLOUT_REPORT.md`.
Reviewer package per run: `review_worksheet.csv` (empty decision columns), `review_report/` (read-only HTML),
`REVIEW_INSTRUCTIONS.md`, `scored_candidates.jsonl`, `deferred_unevaluated.jsonl` (never-evaluated pairs, not non-matches),
`run_summary.json`. The files contain personal values for authorised verification; treat them as sensitive.

### Phase 5: candidate generation (`dedup_pipeline/candidates/`)

Blocking only: finds likely duplicate-person pairs, writes them to local files, and stops. No scoring, no FR-DOPAMS
calls, no merging, no database writes (the session is verified read-only and the run refuses to start otherwise).

```powershell
.\.venv\Scripts\python -m dedup_pipeline.candidates --help
.\.venv\Scripts\python -m dedup_pipeline.candidates            # -> output/candidates/
```

Four independent strategies, grouped on the database side and streamed in bounded batches: `name_exact` (normalised
name, >= 2 tokens), `token_sorted_name` (same tokens, any order), `phone_exact` (last 10 digits of a plausible
number) and `alias_name` (one person's alias equals another's name or alias). Every threshold, block-size cap and
normalisation rule is a field of `CandidateConfig` (override with `--config file.json`). Blocks over their cap are
**deferred, listed and quantified** in `deferred_blocks.jsonl`, never truncated; a run that would exceed
`max_total_pairs` fails instead of dropping pairs. `source_policy` defaults to `all` (V1-V1, V1-V2, V2-V2 are all
kept and counted separately); `cross_source_only` / `same_source_only` are explicit options.

Output: `candidates.jsonl` (one record per unique pair, ids in canonical order, every rule that found it, source
record ids, optional Phase 2 image references), `deferred_blocks.jsonl`, `run_summary.json` (counts per strategy,
union, source pair and rule combination; exclusions; timings; memory) and `run_state.json`. **A run is complete only
if `run_summary.json` exists with `status: completed`**; it is deleted when a run starts and written last. Runs
resume from `work/checkpoint.json` after a crash or kill when the settings and the data stamp are unchanged;
finished runs are never resumed, and output is byte-identical for identical data. Names and numbers are not written
to the output unless `export_comparison_keys` is set.

### Phase 3: FR-DOPAMS client (`dedup_pipeline/face_recognition/`)

A client for the **existing** FR-DOPAMS API plus per-pair face-evidence logic. It never touches a database
and never changes FR-DOPAMS. Configure with `FR_DOPAMS_BASE_URL` (no default; also
`FR_DOPAMS_CONNECT_TIMEOUT`, `FR_DOPAMS_READ_TIMEOUT`, `FR_DOPAMS_RETRIES`).

```powershell
$env:FR_DOPAMS_BASE_URL = "http://<host>:<port>"
.\.venv\Scripts\python -m dedup_pipeline.face_recognition health   # read-only preflight (GET /health, GET /openapi.json)
```

**Pairwise comparison** uses the stateless `POST /compare` that was added to FR-DOPAMS (it enrolls nothing, writes
no match log and touches no database). It needs `FR_DOPAMS_API_KEY` (the service's `COMPARE_API_KEY`):
`compare_pair(a, b, index, loader, FrDopamsPairComparer(client))`. If an image has no usable face, the person's
next usable image (Phase 2 rank order, at most 3 per person) is tried, only for the failing side; every attempt is
recorded. Service failures are `error`, never a mismatch, and never trigger fallback.
`FrDopamsFaceChecker` (single-image check via `/search`) still writes a `match_logs` row per call, so it stays
refused unless `FR_DOPAMS_ALLOW_AUDIT_LOG_WRITES=1`; with `/compare` it is not needed.

Tests: `pytest` runs the unit and source-contract tests. `tests/live/` needs a running FR-DOPAMS and is skipped
otherwise; start one without database initialisation using `tools/run_fr_dopams_stateless.py` and build the
fixtures with `tools/make_live_fixtures.py` (both documented in their docstrings).

### Output (`output/image_resolution/`)

| File | Content |
|---|---|
| `image_manifest.jsonl` | one row per person-media record: status, detected type, size, usability, reason, rank, shared-path flag |
| `person_summary.jsonl` | one row per person: `usable_image_available`, `no_person_media_records`, `no_accessible_media` or `no_usable_image` |
| `shared_paths.jsonl` | paths referenced by more than one person (a lead to investigate, not proof of identity) |
| `run_summary.json` | counts and parameters (no credentials) |
| `checkpoint.jsonl` | resume state; transient errors are retried on the next run |

Reruns are safe: unchanged records are reused from the checkpoint and the outputs are byte-identical.
Face detectability is not checked here; `FaceChecker` is a hook for the FR-DOPAMS stage, and until it is
supplied `face_check_status` is `not_checked`.
