# Robust Multi-Speaker Maritime VHF Text-to-Speech Generation and Quality Evaluation Using Clean and Noisy Reference Speech

## Clickable Index

- [0. Project Scope, Application, Current State, and Machine Layout](#0-project-scope-application-current-state-and-machine-layout)
  - [0.1 Current thesis/application statement](#0-1-current-thesis-application-statement)
  - [0.2 Two-machine workflow](#0-2-two-machine-workflow)
  - [0.3 Current high-level workflow](#0-3-current-high-level-workflow)
  - [0.4 Current normalized dialogue corpus](#0-4-current-normalized-dialogue-corpus)
  - [0.5 Experimental conditions](#0-5-experimental-conditions)
  - [0.6 Active script index — what each script does, why it exists, and I/O](#0-6-active-script-index-what-each-script-does-why-it-exists-and-i-o)
  - [0.7 Current core artefacts and result links](#0-7-current-core-artefacts-and-result-links)
  - [0.8 Current status snapshot](#0-8-current-status-snapshot)
- [Quick Overview — What the Current System Does and Why](#quick-overview-what-the-current-system-does-and-why)
- [1. Current Pipeline](#1-current-pipeline)
  - [1.1 Expanded current pipeline: generation + quality control](#1-1-expanded-current-pipeline-generation-quality-control)
- [2. Speaker-Role Classification](#2-speaker-role-classification)
  - [2.1 What the classifier does](#2-1-what-the-classifier-does)
  - [2.2 Example role cues](#2-2-example-role-cues)
  - [2.3 \`OK\` and \`REVIEW\_REQUIRED\`](#2-3-ok-and-review-required)
  - [2.4 Evidence and validation](#2-4-evidence-and-validation)
- [3. Three Main Dialogue Scripts](#3-three-main-dialogue-scripts)
  - [\`add\_speaker\_labels.py\`](#add-speaker-labels-py)
  - [\`create\_pilot\_generation\_plan.py\`](#create-pilot-generation-plan-py)
  - [\`run\_pilot\_generation\_dialogue.py\`](#run-pilot-generation-dialogue-py)
  - [3.1 Current generation behavior from the attached scripts](#3-1-current-generation-behavior-from-the-attached-scripts)
  - [3.2 Implementation Status of the Attached Generation Scripts](#3-2-implementation-status-of-the-attached-generation-scripts)
- [4. TTS Experimental Conditions](#4-tts-experimental-conditions)
- [5. Dataset Organization and Current Contents](#5-dataset-organization-and-current-contents)
  - [5.1 Clean Dataset](#5-1-clean-dataset)
  - [5.2 Noisy Dataset](#5-2-noisy-dataset)
  - [5.3 Current Noisy Chatterbox Outputs](#5-3-current-noisy-chatterbox-outputs)
  - [5.4 Current Noisy Fish-Speech Outputs](#5-4-current-noisy-fish-speech-outputs)
  - [5.5 Clean vs Noisy Reference Use in the Pilot](#5-5-clean-vs-noisy-reference-use-in-the-pilot)
  - [5.6 Current Reference-Audio Partitioning Method](#5-6-current-reference-audio-partitioning-method)
- [6. Repository Structure](#6-repository-structure)
  - [6.1 Windows repository — canonical documentation/source tree](#6-1-windows-repository-canonical-documentation-source-tree)
  - [6.2 Linux data-station runtime](#6-2-linux-data-station-runtime)
  - [6.3 Fish-Speech repository/runtime](#6-3-fish-speech-repository-runtime)
  - [6.4 Canonical-location principle](#6-4-canonical-location-principle)
- [7. Important Project Files](#7-important-project-files)
  - [Documentation](#documentation)
  - [Dataset directories](#dataset-directories)
  - [Metadata](#metadata)
- [8. Commands — Complete Current Workflow](#8-commands-complete-current-workflow)
  - [8.1 Check the available command-line arguments](#8-1-check-the-available-command-line-arguments)
  - [8.2 Prepare / normalize the multi-scenario distress dataset](#8-2-prepare-normalize-the-multi-scenario-distress-dataset)
  - [8.3 Add Vessel / Coast Guard speaker-role labels](#8-3-add-vessel-coast-guard-speaker-role-labels)
  - [8.4 Create the pilot generation plan](#8-4-create-the-pilot-generation-plan)
  - [8.5 Start the Fish-Speech API server](#8-5-start-the-fish-speech-api-server)
  - [8.6 Validate the dialogue plan without generating audio](#8-6-validate-the-dialogue-plan-without-generating-audio)
  - [8.7 Generate a small audio test](#8-7-generate-a-small-audio-test)
  - [8.8 Run the complete dialogue generation](#8-8-run-the-complete-dialogue-generation)
  - [8.9 Test one Chatterbox sample](#8-9-test-one-chatterbox-sample)
  - [8.10 Test one Fish-Speech sample](#8-10-test-one-fish-speech-sample)
  - [8.11 Run ASR on the four current pilot output conditions](#8-11-run-asr-on-the-four-current-pilot-output-conditions)
  - [8.12 Calculate text metrics](#8-12-calculate-text-metrics)
  - [8.13 Run the current reference-audio partitioning method](#8-13-run-the-current-reference-audio-partitioning-method)
  - [8.14 Earlier / auxiliary SNR-noise analysis](#8-14-earlier-auxiliary-snr-noise-analysis)
  - [8.15 Earlier threshold-testing utility](#8-15-earlier-threshold-testing-utility)
  - [8.16 Diagnose failed generated samples with the current 3-way backend](#8-16-diagnose-failed-generated-samples-with-the-current-3-way-backend)
  - [8.17 Validate the speaker backend on real SAME/DIFFERENT controls](#8-17-validate-the-speaker-backend-on-real-same-different-controls)
  - [8.18 Legacy WhatsApp-only backend control](#8-18-legacy-whatsapp-only-backend-control)
- [9. Fish-Speech Server](#9-fish-speech-server)
- [10. Evaluation of generated audio](#10-evaluation-of-generated-audio)
- [IMPORTANT: Why the Old Threshold Method Was Replaced](#important-why-the-old-threshold-method-was-replaced)
  - [Earlier method](#earlier-method)
  - [Why this became a problem](#why-this-became-a-problem)
  - [Why the intended-vs-other margin was not enough](#why-the-intended-vs-other-margin-was-not-enough)
  - [Why the anchor was not enough](#why-the-anchor-was-not-enough)
  - [Practical false-pass behavior](#practical-false-pass-behavior)
  - [Why noisy/VHF audio made the problem harder](#why-noisy-vhf-audio-made-the-problem-harder)
  - [Why the project moved to `speaker_backend_3way.py`](#why-the-project-moved-to-speaker-backend-3way-py)
  - [10.1 Speaker-identity QC: why the method changed](#10-1-speaker-identity-qc-why-the-method-changed)
  - [10.1A Detailed Speaker-QC Evolution — Why the 3-Way Backend Was Needed](#10-1a-detailed-speaker-qc-evolution-why-the-3-way-backend-was-needed)
  - [10.2 Correct scientific interpretation](#10-2-correct-scientific-interpretation)
  - [10.3 Recommended future improvements](#10-3-recommended-future-improvements)
  - [10.4 Remaining limitations of the 3-way speaker backend](#10-4-remaining-limitations-of-the-3-way-speaker-backend)
  - [10.5 General QC architecture](#10-5-general-qc-architecture)
  - [10.6 Reference QC before TTS generation](#10-6-reference-qc-before-tts-generation)
  - [10.7 Current general failure meanings and solutions](#10-7-current-general-failure-meanings-and-solutions)
  - [10.8 Retry policy](#10-8-retry-policy)
  - [10.9 Current experimental interpretation](#10-9-current-experimental-interpretation)
- [11. Planned Evaluation Metrics](#11-planned-evaluation-metrics)
  - [11.1 Speaker-verification metrics to report separately](#11-1-speaker-verification-metrics-to-report-separately)
- [13. Design Principle](#13-design-principle)
  - [13.1 Four different questions must stay separate](#13-1-four-different-questions-must-stay-separate)
- [14. Completed Controlled Pilot — 15 Dialogues × 4 Conditions](#14-completed-controlled-pilot-15-dialogues-4-conditions)
  - [14.1 Five unique problematic Vessel references](#14-1-five-unique-problematic-vessel-references)
- [15. Failed-Sample Diagnostic Study](#15-failed-sample-diagnostic-study)
  - [15.1 Test 1 — can the backend recognize two real chunks from the same original recording?](#15-1-test-1-can-the-backend-recognize-two-real-chunks-from-the-same-original-recording)
  - [15.2 Tests 2/3 — compare failed generated attempts against A/B/full reference](#15-2-tests-2-3-compare-failed-generated-attempts-against-a-b-full-reference)
  - [15.3 Important detailed observations](#15-3-important-detailed-observations)
- [16. Real-Human Speaker-Backend Validation](#16-real-human-speaker-backend-validation)
  - [16.1 What this means](#16-1-what-this-means)
  - [16.2 Scientific limitation](#16-2-scientific-limitation)
  - [16.3 What is actually compared](#16-3-what-is-actually-compared)
- [17. Current Repository Storage and Cleanup Policy](#17-current-repository-storage-and-cleanup-policy)
  - [17.1 Keep as active/current](#17-1-keep-as-active-current)
  - [17.2 Generation-plan versioning](#17-2-generation-plan-versioning)
  - [17.3 Earlier single-output evaluation workbooks](#17-3-earlier-single-output-evaluation-workbooks)
  - [17.4 Earlier reference-partition threshold experiments](#17-4-earlier-reference-partition-threshold-experiments)
  - [17.5 What must stay data-station-only](#17-5-what-must-stay-data-station-only)
- [18. Next Experiment — Channel-Mismatch Diagnostic](#18-next-experiment-channel-mismatch-diagnostic)
- [19. Current Thesis Interpretation and Real-Life Use](#19-current-thesis-interpretation-and-real-life-use)
- [20. References, Literature, Standards, and Citation Map](#20-references-literature-standards-and-citation-map)
  - [Local literature and standards stored in this repository](#local-literature-and-standards-stored-in-this-repository)
  - [Maritime communication](#maritime-communication)
  - [Speech quality](#speech-quality)
  - [TTS models](#tts-models)
  - [Dataset background](#dataset-background)
  - [Speaker verification, calibration, PLDA, and score normalization](#speaker-verification-calibration-plda-and-score-normalization)
  - [Speaker-QC Reference Map — What Each Source Supports](#speaker-qc-reference-map-what-each-source-supports)
  - [Reference-audio partitioning and noise estimation](#reference-audio-partitioning-and-noise-estimation)

---


> **README status:** consolidated project record through **20 September 2026**.  
> **Canonical active dialogue generator:** `run_pilot_generation_dialogue.py`. Older references to `run_generation_dialogue.py` are historical/legacy only.


This repository contains the experimental pipeline for generating and evaluating maritime VHF-style distress communication with **Chatterbox** and **Fish-Speech**.
The current workflow focuses on:

- preparing maritime distress dialogue text,

- automatically assigning **VESSEL** and **COAST\_GUARD** communication roles,

- generating two-speaker dialogue with separate reference voices,

- comparing clean-reference and noisy-reference TTS conditions,

- transcribing generated speech with ASR,

- evaluating intelligibility and acoustic quality.

> **Working thesis title:** 
Robust Multi-Speaker Maritime VHF Text-to-Speech Generation and Quality Evaluation Using Clean and Noisy Reference Speech

---


<a id="0-project-scope-application-current-state-and-machine-layout"></a>
## 0. Project Scope, Application, Current State, and Machine Layout

<a id="0-1-current-thesis-application-statement"></a>
### 0.1 Current thesis/application statement

The project investigates **automated generation and multi-dimensional quality evaluation of multi-speaker maritime VHF radio dialogues using TTS**. Real publicly available maritime/VHF recordings and cleaner demonstration recordings are used as reference/conditioning audio. The objective is to determine whether TTS models can generate realistic domain-specific maritime communication while preserving:

```text
speaker identity
message/content fidelity
usable signal quality
relevant radio/channel characteristics
```

under both cleaner and noisy/VHF-like reference conditions.

The generated output can technically be called *synthetic speech* because it is produced by TTS, but this repository generally uses **TTS-generated maritime VHF speech/dialogue** to avoid implying that the original reference recordings are artificial.

A practical downstream use is the creation of controlled maritime-radio speech/dialogue material for speech-system research, simulation and dataset augmentation. In this thesis, however, **ASR is primarily an evaluation component**: generated speech is transcribed and compared with the intended target text to detect missing, substituted, repeated or hallucinated content. ASR training is a possible downstream application, not the primary task implemented here.

<a id="0-2-two-machine-workflow"></a>
### 0.2 Two-machine workflow

| Environment | Root | Main responsibility |
|---|---|---|
| Windows repository | `C:\Users\kda\Documents\ma-tts-tuning` | Canonical source code, metadata preparation, documentation, compact experiment results, reference material and Git history |
| Linux data station | `/data/kavya/conda_envs/chatterbox` | Chatterbox runtime, integrated dialogue generation, speaker-QC backend, failure diagnostics and large experiment artefacts |
| Fish-Speech runtime | `/data/kavya/fishspeech` | Fish-Speech API server and model checkpoints |

<a id="0-3-current-high-level-workflow"></a>
### 0.3 Current high-level workflow

```text
REAL maritime/demo reference recordings
                 +
        maritime dialogue text
                 ↓
prepare_distress_dataset.py
                 ↓
normalized multi-scenario workbook
                 ↓
add_speaker_labels.py
                 ↓
VESSEL / COAST_GUARD role-labelled dialogue
                 ↓
create_pilot_generation_plan.py
                 ↓
pilot_generation_plan.csv
                 ↓
run_pilot_generation_dialogue.py
        ┌────────┴────────┐
        ↓                 ↓
   Chatterbox        Fish-Speech
        ↓                 ↓
     turn-level generation + bounded retry
                 ↓
        MULTI-DIMENSIONAL QC
        ├── speaker identity
        │   └── ECAPA → calibrated cosine / PLDA / cohort → 2-of-3
        ├── signal quality
        ├── ASR/content fidelity
        ├── channel/noise diagnostics
        └── manual listening
                 ↓
          accepted turns
                 ↓
       concatenated dialogue
                 ↓
        experiment analysis
```

<a id="0-4-current-normalized-dialogue-corpus"></a>
### 0.4 Current normalized dialogue corpus

The current normalized workbook represents **500 dialogues across five scenario classes**:

```text
armed_attack    100
sinking         100
collision       100
fire            100
flooding        100
-------------------
total           500
```

Canonical current normalized workbook:

- [workspace/data/pilot/metadata/maritime_distress_normalized.xlsx](workspace/data/pilot/metadata/maritime_distress_normalized.xlsx)

Original/source workbook:

- [workspace/docs/metadata/maritime_distress_formatted (version 1).xlsx](workspace/docs/metadata/maritime_distress_formatted%20(version%201).xlsx)

The source workbook contains the earlier formatted corpus plus embedded scenario material that is normalized by `prepare_distress_dataset.py`.

<a id="0-5-experimental-conditions"></a>
### 0.5 Experimental conditions

Every selected dialogue can be expanded into four controlled TTS conditions:

| Condition | Model | Vessel reference condition |
|---|---|---|
| `chatterbox_clean_ref` | Chatterbox | clean/wideband reference |
| `fishspeech_clean_ref` | Fish-Speech | clean/wideband reference |
| `chatterbox_noisy_ref` | Chatterbox | noisy/native-VHF reference |
| `fishspeech_noisy_ref` | Fish-Speech | noisy/native-VHF reference |

Separate Vessel and Coast Guard reference voices are used. Role labels are control metadata and are not spoken.

<a id="0-6-active-script-index-what-each-script-does-why-it-exists-and-i-o"></a>
### 0.6 Active script index — what each script does, why it exists, and I/O

| Script | Status | Run mainly on | Purpose / how it works | Main input | Main output |
|---|---|---|---|---|---|
| [prepare_distress_dataset.py](workspace/docs/tools/prepare_distress_dataset.py) | Active | Windows | Normalizes the source workbook and merges scenario material into a consistent dialogue table. | Source maritime Excel / optional scenario JSON | `maritime_distress_normalized.xlsx` |
| [add_speaker_labels.py](workspace/docs/tools/add_speaker_labels.py) | Active | Windows | Rule-based semantic classifier assigns `VESSEL`, `COAST_GUARD` or `REVIEW_REQUIRED`; scores are rules, not probabilities. | Normalized workbook | `maritime_distress_speaker_labeled.xlsx` + review information |
| [create_pilot_generation_plan.py](workspace/docs/tools/create_pilot_generation_plan.py) | Active | Windows / data station | Expands labelled dialogues into model/reference conditions, selects role-specific references and target/output paths. | Speaker-labelled workbook + reference pools | `pilot_generation_plan.csv` |
| [run_pilot_generation_dialogue.py](workspace/docs/tools/run_pilot_generation_dialogue.py) | **Canonical active generator** | Data station | Parses labelled turns, generates each turn with Chatterbox/Fish-Speech, runs QC/retries and concatenates accepted turns. | Plan CSV, references, Fish API, speaker backend | Attempt WAVs, accepted turn WAVs, QC CSVs, final dialogue WAV, updated statuses |
| [run_pilot_generation.py](workspace/docs/tools/run_pilot_generation.py) | Earlier/helper | Windows/data station | Earlier pilot-generation helper retained for reproducibility; the dialogue generator is now the canonical multi-turn path. | Pilot metadata | Earlier pilot outputs |
| [generate_chatterbox_one.py](workspace/docs/tools/generate_chatterbox_one.py) | Helper | Windows/data station | Generates one Chatterbox sample to validate environment/reference conditioning independently. | Text + reference WAV | One WAV |
| [generate_fishspeech_one.py](workspace/docs/tools/generate_fishspeech_one.py) | Helper | Data station | Sends one TTS request to Fish-Speech for isolated testing. | Target text + Fish API (+ reference data depending on script version) | One WAV |
| [asr_transcribe.py](workspace/docs/tools/asr_transcribe.py) | Evaluation | Windows/data station | Transcribes generated audio with faster-whisper for content-fidelity analysis. | WAV directory | Summary CSV + per-audio transcript/segment files |
| [calc_text_metrics.py](workspace/docs/tools/calc_text_metrics.py) | Evaluation | Windows | Computes WER, CER, keyword and number correctness from target/reference text vs ASR output. | Evaluation workbook | Workbook with text metrics |
| [sort_reference_audio.py](workspace/docs/tools/sort_reference_audio.py) | Active preprocessing | Windows | Computes `R_high` and dynamic pause-floor SNR and partitions references into clean vs noisy/native-VHF. | Reference WAV collection | `classification_summary.csv` + Node E/F folders |
| [noise_cal.py](workspace/docs/tools/noise_cal.py) | Legacy/auxiliary | Windows | Earlier non-intrusive SNR/noise-floor inspection. | WAV | Text/console noise metrics |
| [test_noisy_thresholds.py](workspace/docs/tools/test_noisy_thresholds.py) | Legacy experiment | Windows | Sweeps older SNR/noise-dBFS rules to study noisy-reference separation. | Legacy inventory CSV | Threshold-test CSV |
| [speaker_backend_3way.py](workspace/docs/tools/speaker_backend_3way.py) | **Active speaker QC** | Data station | Uses shared ECAPA embeddings with calibrated cosine, PLDA and cohort-normalized cosine; final SAME requires at least 2/3 votes. | Speaker embeddings + fitted backend model | Component scores, votes, SAME/DIFFERENT |
| [diagnose_failed_samples.py](workspace/docs/tools/diagnose_failed_samples.py) | Active diagnostic | Data station | Finds failed dialogue rows/turn attempts, builds non-overlapping A/B reference controls and compares generated attempts against A/B/full reference. | Plan CSV + failed turn QC + backend model | `reference_test1_summary.csv`, `failed_attempt_tests_2_3.csv`, A/B chunks |
| [validate_speaker_backend_real_trials.py](workspace/docs/tools/validate_speaker_backend_real_trials.py) | Active diagnostic | Data station | Tests the backend on known-SAME real A/B chunks and known-DIFFERENT Coast Guard controls; summarizes method accuracy. | A/B controls + two different-speaker refs + backend model | `real_trials_results.csv`, `method_summary.csv` |
| [legacy/test_whatsapp_chunks.py](workspace/docs/tools/legacy/test_whatsapp_chunks.py) | Legacy | Data station/Windows | Earlier one-reference known-SAME diagnostic that motivated the generalized diagnostic script. | WhatsApp A/B chunks | Console/backend comparison |

Legacy early experiment scripts retained inside the old noisy-output folders include `run_chatterbox_baseline.py`, `run_chatterbox_prompt.py`, `run_chatterbox_reference.py`, and Fish-Speech `run_webui.py`. They are historical baselines, not the current main pipeline.

<a id="0-7-current-core-artefacts-and-result-links"></a>
### 0.7 Current core artefacts and result links

| Artefact | Role |
|---|---|
| [Normalized corpus](workspace/data/pilot/metadata/maritime_distress_normalized.xlsx) | Current 500-dialogue, five-scenario normalized source for role labelling |
| `workspace/data/pilot/metadata/metadata/maritime_distress_speaker_labeled.xlsx` | Current tree location of speaker-labelled workbook; recommended to flatten one level later |
| [Pilot generation plan](workspace/data/pilot/metadata/pilot_generation_plan.csv) | Generation routing/configuration; ensure a newly generated five-scenario plan is not confused with the older 400-row plan |
| [Reference partition summary](workspace/docs/sorted_pipeline_data/classification_summary.csv) | Current clean/noisy partition results using `R_high + dynamic SNR` |
| [Failed-reference Test-1 summary](workspace/data/diagnostics/failed_reference_speaker_tests/reference_test1_summary.csv) | A/B known-SAME reference controls |
| [Failed generated-attempt diagnostics](workspace/data/diagnostics/failed_reference_speaker_tests/failed_attempt_tests_2_3.csv) | Generated attempts compared with verified reference chunks |
| [Real-backend trial results](workspace/data/diagnostics/backend_real_validation/real_trials_results.csv) | All 15 known-SAME/known-DIFFERENT backend control trials |
| [Backend method summary](workspace/data/diagnostics/backend_real_validation/method_summary.csv) | Accuracy summary for calibrated cosine, PLDA, cohort and final vote |
| Data-station `workspace/models/speaker_backend_3way.pkl` | Fitted 3-way backend model used by current speaker QC; currently runtime-side rather than visible in the latest Windows tree |

<a id="0-8-current-status-snapshot"></a>
### 0.8 Current status snapshot

Completed so far:

```text
✓ multi-scenario dialogue normalization
✓ semantic Vessel / Coast Guard role labelling
✓ four-condition generation-plan design
✓ Chatterbox + Fish-Speech multi-turn generation
✓ clean/noisy reference partitioning with R_high + dynamic SNR
✓ signal QC and bounded retry logic
✓ ECAPA speaker embeddings
✓ raw-cosine threshold experiments
✓ intended / competing-role / anchor experiments
✓ replacement with calibrated 3-way speaker backend
✓ 60-run controlled pilot analysis
✓ failed-sample diagnostic across five problematic references
✓ manual confirmation that all five A/B reference pairs are same physical speaker
✓ real-human SAME/DIFFERENT backend sanity validation
```

Next unresolved diagnostic:

```text
reference = noisy/VHF/channel-limited
TTS output = often cleaner / spectrally different
        ↓
Does channel mismatch itself cause part of the speaker-verifier rejection?
```

The planned next experiment is therefore a **channel-matching diagnostic** on failed generated attempts before changing the 2-of-3 speaker decision rule or modifying TTS references.

---

<a id="quick-overview-what-the-current-system-does-and-why"></a>
## Quick Overview — What the Current System Does and Why

The project has four different jobs that must not be confused:

```text
1. SPEAKER ROLE FROM TEXT
   "Is this turn Vessel or Coast Guard?"
   → add_speaker_labels.py

2. GENERATION PLAN
   "Which TTS model, reference WAV, transcript, seed and output path should be used?"
   → create_pilot_generation_plan.py

3. AUDIO GENERATION
   "Generate each labelled turn and concatenate the dialogue."
   → run_pilot_generation_dialogue.py

4. QUALITY CONTROL
   "Did the generated turn keep the correct speaker, say the correct words,
    avoid clipping/noise, and remain usable?"
   → speaker_backend_3way.py + signal QC + ASR/content QC + manual listening
```

The key lesson from the pilot experiments is:

> A TTS request finishing successfully does **not** mean the generated turn is correct.

A turn can still contain the wrong physical speaker identity, speaker drift, missing words, repeated words such as `Over`, excessive radio/static noise, clipping, long silence, or hallucinated speech. The current design therefore treats **generation** and **quality control** as separate stages.

The speaker-identity QC also evolved over time:

```text
Initial generation
    ↓
No automatic speaker-consistency check
    ↓
ECAPA-TDNN embedding + raw cosine similarity
    ↓
Problem: manually chosen thresholds behaved differently for clean and noisy/VHF audio
    ↓
Reference / competing-speaker / anchor comparisons
    ↓
Problem: margins and anchors could still allow false passes
    ↓
speaker_backend_3way.py
    ↓
ECAPA embedding interpreted by:
    1. calibrated cosine
    2. PLDA
    3. cohort-normalized cosine
    ↓
2-of-3 SAME / DIFFERENT decision
```

The detailed explanation of this progression is in **Section 10**.

---

<a id="1-current-pipeline"></a>
## 1. Current Pipeline

```

Maritime dialogue Excel

        │

        ▼

add_speaker_labels.py

        │

        │ rule-based semantic speaker-role classification

        ▼

speaker_labeled_text

VESSEL: ...

COAST_GUARD: ...

        │

        ▼

create_pilot_generation_plan.py

        │

        ▼

pilot_generation_plan.csv

        │

        ▼

run_pilot_generation_dialogue.py

        │

        ├── Chatterbox

        └── Fish-Speech

        │

        ▼

individual turn WAV files

        │

        ▼

final concatenated dialogue WAV

        │

        ▼

ASR + text/acoustic evaluation

```

The visual project pipeline is also available in
[workspace/docs/main\_pipeline.png](workspace/docs/main_pipeline.png).

<a id="1-1-expanded-current-pipeline-generation-quality-control"></a>
### 1.1 Expanded current pipeline: generation + quality control

The generation pipeline above is the **audio-creation path**. The current project also needs a separate quality-control path because a TTS call can finish successfully while still producing the wrong speaker, missing words, repeated words, excessive noise, or clipping.

```
Speaker-labelled dialogue
        │
        ▼
create_pilot_generation_plan.py
        │
        ▼
pilot_generation_plan.csv
        │
        ▼
run_pilot_generation_dialogue.py
        │
        ├── Chatterbox
        └── Fish-Speech
        │
        ▼
individual turn WAV files
        │
        ├── signal QC
        │     clipping / peak / duration / silence / noise
        │
        ├── speaker-identity QC
        │     ECAPA embedding
        │     └── speaker_backend_3way.py
        │         ├── calibrated cosine
        │         ├── PLDA
        │         └── cohort-normalized cosine
        │              ↓
        │           2-of-3 vote
        │
        ├── content QC
        │     ASR vs target text
        │     missing / repeated / hallucinated words
        │     names / numbers / positions / "Over"
        │
        └── manual listening
              urgency / authority / accent / VHF realism
        │
        ▼
accepted turn WAVs
        │
        ▼
final concatenated dialogue WAV
```

The important separation is:

```
speaker-role classification
≠
speaker-identity verification
```

`add_speaker_labels.py` decides **which communication role the text belongs to**. The later speaker-verification backend checks whether a generated Vessel or Coast Guard turn still sounds like the **intended physical speaker**.

---

---

<a id="2-speaker-role-classification"></a>
## 2. Speaker-Role Classification

The source maritime Excel contains dialogue text but does not contain reliable speaker metadata for every radio turn. Before TTS generation, the dialogue therefore passes through a **rule-based semantic speaker-role classifier**.

<a id="2-1-what-the-classifier-does"></a>
### 2.1 What the classifier does

\`add\_speaker\_labels.py\`:

1. reads the source dialogue from \`clarity\_formatted\_text\`,

2. separates radio transmissions using turn-ending markers such as \`Over\` and \`Out\`,

3. searches each transmission for maritime role-specific lexical and communicative cues,

4. calculates separate deterministic scores for \`VESSEL\` and \`COAST\_GUARD\`,

5. assigns a role only when the rule criteria are sufficiently strong,

6. marks unclear cases as \`REVIEW\_REQUIRED\`,

7. writes the result to a new \`speaker\_labeled\_text\` column.

Example:

```

Input:

Mayday. Mayday. This is tanker BW BOSTON. Over.

Output:

VESSEL: Mayday. Mayday. This is tanker BW BOSTON. Over.

```

Another example:

```

Input:

Maintain radio silence. A naval vessel is en route. Over.

Output:

COAST_GUARD: Maintain radio silence. A naval vessel is en route. Over.

```

<a id="2-2-example-role-cues"></a>
### 2.2 Example role cues

Typical **VESSEL** evidence includes:

```

Mayday

MMSI

this is tanker / motor vessel / yacht

under attack

pirates / intruders / citadel

taking on water

fire on board

engine failure

running low on fuel

request immediate assistance

on-board status reports

```

Typical **COAST\_GUARD** evidence includes:

```

this is Coast Guard

what is your position

are there any injuries

confirm / report / provide

stand by

maintain radio silence

continue evasive manoeuvres

assistance is on the way

rescue asset is en route

dispatch information

ETA information

```

These rules describe the **communication role expressed by the text**. They are not acoustic speaker-recognition rules.

<a id="2-3-ok-and-review-required"></a>
### 2.3 \`OK\` and \`REVIEW\_REQUIRED\`

The internal rule scores are **not probabilities** and must not be interpreted as calibrated confidence values.

```

OK

= one role satisfies the required rule threshold and score margin.

REVIEW_REQUIRED

= evidence is too weak, mixed, or ambiguous for automatic assignment.

```

For example:

```

VESSEL score      = 12

COAST_GUARD score = 0

→ VESSEL

→ OK

```

while an ambiguous transmission such as:

```

Roger. Over.

```

may provide insufficient role evidence and is therefore marked:

```

REVIEW_REQUIRED

```

The classifier deliberately avoids assuming strict speaker alternation. These sequences are valid:

```

VESSEL

VESSEL

COAST_GUARD

COAST_GUARD

VESSEL

```

<a id="2-4-evidence-and-validation"></a>
### 2.4 Evidence and validation

The rule definitions are motivated by the standardized structure of maritime safety communication, particularly the **IMO Standard Marine Communication Phrases (SMCP)**.
For thesis evaluation, the rule weights themselves should not be reported as prediction probabilities. The classifier should be validated against manually annotated speaker-role ground truth using metrics such as:

```

accuracy

precision

recall

F1-score

automatic-classification coverage

REVIEW_REQUIRED rate

```

---

<a id="3-three-main-dialogue-scripts"></a>
## 3. Three Main Dialogue Scripts

<a id="add-speaker-labels-py"></a>
### \`add\_speaker\_labels.py\`

Path:
[workspace/docs/tools/add\_speaker\_labels.py](workspace/docs/tools/add_speaker_labels.py)
Purpose:

```

WHO is speaking?

```

It performs the only automatic speaker-role inference in the current dialogue pipeline and produces:

```

speaker_labeled_text

speaker_label_status

speaker_label_report

```

<a id="create-pilot-generation-plan-py"></a>
### \`create\_pilot\_generation\_plan.py\`

Path:
[workspace/docs/tools/create\_pilot\_generation\_plan.py](workspace/docs/tools/create_pilot_generation_plan.py)
Purpose:

```

HOW should each sample be generated?

```

It does **not** infer speakers.
It reads the already labeled dialogue and prepares:

```

model

condition

dialogue_text

vessel_reference_audio

coast_guard_reference_audio

output_audio

```

The output is:
[workspace/data/pilot/metadata/pilot\_generation\_plan.csv](workspace/data/pilot/metadata/pilot_generation_plan.csv)

<a id="run-pilot-generation-dialogue-py"></a>
### \`run\_pilot\_generation\_dialogue.py\`

Path:
[workspace/docs/tools/run\_pilot\_generation\_dialogue.py](workspace/docs/tools/run_pilot_generation_dialogue.py)
Purpose:

```

GENERATE the actual audio.

```

For every turn it:

1. reads the existing \`VESSEL\` or \`COAST\_GUARD\` label,

2. removes the label from the spoken text,

3. selects the corresponding reference voice,

4. generates the turn with Chatterbox or Fish-Speech,

5. saves the individual turn WAV,

6. concatenates all turn WAVs into the final dialogue.

The labels are control metadata and are **not spoken** by the TTS models.

<a id="3-1-current-generation-behavior-from-the-attached-scripts"></a>
### 3.1 Current generation behavior from the attached scripts

The current uploaded generation implementation uses the following additional rules.

#### Role-aware parsing

The generator accepts explicit labels such as:

```
VESSEL: ...
COAST_GUARD: ...
```

and the robust parser also supports:

```
VESSEL: ... COAST_GUARD: ... VESSEL: ...
```

inside one Excel cell. It does **not** infer or alternate speakers. Both `VESSEL` and `COAST_GUARD` must be present before generation continues.

#### Chatterbox settings

Current default role settings in the uploaded generator are:

```
VESSEL
  exaggeration = 0.70
  cfg_weight   = 0.35
  seed         = 42001

COAST_GUARD
  exaggeration = 0.35
  cfg_weight   = 0.50
  seed         = 43001
```

The fixed role seeds are used to reduce random voice drift between turns. A fixed seed improves reproducibility but does **not** guarantee perfect speaker identity.

#### Fish-Speech settings

Current settings in the uploaded generator are:

```
chunk_length       = 300
max_new_tokens     = 2048
top_p              = 0.80
temperature        = 0.55
repetition_penalty = 1.10
use_memory_cache   = on

VESSEL seed        = 42001
COAST_GUARD seed   = 43001
```

Fish-Speech uses the exact reference WAV together with the transcript of that same WAV. The transcript is conditioning metadata and must match the audio actually spoken in the reference recording.

#### Dialogue text vs Fish-Speech target text

The generation plan keeps two text fields:

```
dialogue_text
→ original speaker-labelled dialogue
→ used by Chatterbox

target_text
→ Fish-Speech version with stable role tags
→ used by Fish-Speech
```

Current stable Fish-Speech role tags are:

```
VESSEL:
[urgent, controlled distress]

COAST_GUARD:
[calm, clear, authoritative]
```

Using one stable style per role is intentional. Earlier experiments used larger turn-to-turn emotion changes such as `panicked`, `fearful`, `tense`, and `anxious`; these could change pitch, pace, and timbre enough to sound like a different speaker.

#### Reference selection rule

Two Coast Guard references are currently retained:

```
workspace/docs/Noisy_dataset/reference_voices/Northern Belle_ coast_guard.wav
workspace/docs/Noisy_dataset/reference_voices/coast_guard_noisy.wav
```

Only **one Coast Guard reference is selected for one dialogue at a time**. The same selected Coast Guard reference should then be used for all four experimental conditions of that dialogue so that model comparisons remain fair.
The intended plan logic is:

```
coast_guard_reference_audio = COAST_GUARD_REFS[
    (text_id - 1) % len(COAST_GUARD_REFS)
]
```

Do **not** assign the whole list:

```
coast_guard_reference_audio = COAST_GUARD_REFS   # wrong
```

because `get_reference_text()` expects one file path, not a Python list.

#### Reference transcript formatting rule

Adjacent Python strings are joined without adding spaces automatically. To avoid malformed transcripts such as:

```
coast guardreceived your mayday
```

store multi-part transcripts with `" ".join([...])`:

```
"workspace/docs/Noisy_dataset/reference_voices/coast_guard_noisy.wav":
    " ".join([
        "mayday yacht delias this is belfast coast guard",
        "received your mayday regarding the motorboat voyager",
        "standby on this channel over",
    ]),
```

This guarantees exactly one space between transcript fragments.

#### Chatterbox clipping protection

Some otherwise good Chatterbox attempts have failed signal QC because their waveform reaches full scale. Peak normalization should be applied before saving and before QC:

```
wav = ensure_2d_audio(wav)
peak = wav.abs().max()

if peak > 0:
    target_peak = 10 ** (-1.0 / 20.0)  # approximately -1 dBFS
    wav = wav * (target_peak / peak)
```

This addresses `EXCESSIVE_CLIPPING`. It does **not** repair a wrong speaker, missing content, or excessive generated noise.

#### Script-name note

The uploaded current generator is named:

```
run_pilot_generation_dialogue.py
```

Some older README commands and repository snapshots use:

```
run_pilot_generation_dialogue.py
```

Use the actual filename present in the working environment. Before a long run, verify it with:

```
python run_pilot_generation_dialogue.py -h
```

or the corresponding local filename.

---

---


<a id="3-2-implementation-status-of-the-attached-generation-scripts"></a>
### 3.2 Implementation Status of the Attached Generation Scripts

This section distinguishes what is already implemented in the attached scripts from what is still a required correction or a recommended next step.

| Item | Status in attached scripts | Meaning |
|---|---|---|
| Explicit `VESSEL:` / `COAST_GUARD:` parsing | **Implemented** in `run_pilot_generation_dialogue.py` | The generator does not guess or alternate speakers. |
| Inline labels inside one Excel cell | **Implemented** in the generator parser | `VESSEL: ... COAST_GUARD: ...` can be split into separate turns. |
| Separate Vessel and Coast Guard references | **Implemented** | Each role receives its own reference path from the generation-plan CSV. |
| Stable role seeds | **Implemented** | Vessel uses seed `42001`; Coast Guard uses `43001` in the attached generator. |
| Fish-Speech reference transcript | **Implemented** | Fish-Speech receives the transcript belonging to the selected reference WAV. |
| Fish-Speech stable role emotion tags | **Implemented in the plan design** | Vessel and Coast Guard use stable role styles rather than large turn-to-turn emotion changes. |
| Fish-Speech `max_new_tokens=2048`, lower temperature/top-p | **Implemented** | Intended to reduce truncation and excessive sampling variation. |
| Dry-run dialogue/reference validation | **Implemented** | Turns and references can be checked before synthesis. |
| One Coast Guard reference selected at a time | **Required correction in the attached `create_pilot_generation_plan.py` snapshot** | The uploaded snapshot still assigns the whole `COAST_GUARD_REFS` list. It must select one path by index. |
| Safe reference-transcript spacing | **Required correction in the attached plan-script snapshot** | Adjacent Python strings currently join without spaces in several transcripts. Use `" ".join([...])`. |
| Chatterbox peak normalization before save | **Recommended; not implemented in the attached generator snapshot** | Needed to prevent otherwise good attempts from failing with `EXCESSIVE_CLIPPING`. |
| 3-way speaker verification during generation | **Separate QC backend, not part of the attached generator** | `speaker_backend_3way.py` evaluates speaker identity after or around generation. |
| ASR content gate | **Separate evaluation stage** | Needed for missing/repeated words, names, numbers, positions and `Over`. |
| Explicit generated-noise gate | **Recommended QC addition** | Particularly important for Fish-Speech noisy-reference outputs. |

#### Important correction: select one Coast Guard reference, not the entire list

The attached `create_pilot_generation_plan.py` snapshot currently contains the incorrect assignment:

```python
coast_guard_reference_audio = COAST_GUARD_REFS
```

`COAST_GUARD_REFS` is a Python list, while `get_reference_text()` expects one path string. The intended logic is:

```python
coast_guard_reference_audio = COAST_GUARD_REFS[
    (text_id - 1) % len(COAST_GUARD_REFS)
]
```

This keeps both Coast Guard recordings available across the dataset while using **only one Coast Guard reference inside one dialogue**.

Example:

```text
Text 55 → Coast Guard reference A for all four TTS conditions
Text 56 → Coast Guard reference B for all four TTS conditions
Text 57 → Coast Guard reference A for all four TTS conditions
...
```

That design makes the Chatterbox/Fish-Speech comparison fair because the Coast Guard conditioning speaker does not change between the four conditions of the same source dialogue.

#### Important correction: reference-transcript spacing

Python does not insert spaces between adjacent quoted strings:

```python
"coast guard"
"received your mayday"
```

becomes:

```text
coast guardreceived your mayday
```

For Fish-Speech reference transcripts, use:

```python
" ".join([
    "mayday yacht delias this is belfast coast guard",
    "received your mayday regarding the motorboat voyager",
    "standby on this channel over",
])
```

The reference transcript should describe the **reference WAV**, not the new target dialogue.

#### Recommended Chatterbox signal fix

The attached generation script currently saves Chatterbox output directly after synthesis. Some pilot attempts reached full-scale amplitude and failed `EXCESSIVE_CLIPPING`.

Recommended addition before `ta.save(...)`:

```python
wav = ensure_2d_audio(wav)

peak = wav.abs().max()
if peak > 0:
    target_peak = 10 ** (-1.0 / 20.0)  # approximately -1 dBFS
    wav = wav * (target_peak / peak)
```

This fixes output level only. It does **not** repair wrong speaker identity, content errors, or generated noise.

---

<a id="4-tts-experimental-conditions"></a>
## 4. TTS Experimental Conditions

The pilot generation currently compares four conditions:
\| Condition | TTS model | Reference condition |
\|---|---|---|
\| \`chatterbox\_clean\_ref\` | Chatterbox | clean reference |
\| \`fishspeech\_clean\_ref\` | Fish-Speech | clean reference |
\| \`chatterbox\_noisy\_ref\` | Chatterbox | noisy reference |
\| \`fishspeech\_noisy\_ref\` | Fish-Speech | noisy reference |
Separate Vessel and Coast Guard reference voices are used so that each communication role receives the intended speaker identity.

---

<a id="5-dataset-organization-and-current-contents"></a>
## 5. Dataset Organization and Current Contents

The project currently maintains two main reference-audio groups:

```

clean_dataset

Noisy_dataset

```

These are kept separate because the experimental conditions compare TTS generation using cleaner reference speech against TTS generation using naturally noisy / VHF-like reference speech.
The folder in which a file is stored is the **source dataset label**. Acoustic measures such as estimated SNR and noise dBFS are evaluated separately and can later be used to determine whether a file behaves acoustically as clean, noisy, or borderline.

<a id="5-1-clean-dataset"></a>
### 5.1 Clean Dataset

Path:
[workspace/docs/clean\_dataset/](workspace/docs/clean_dataset/)
Main structure:

```

workspace/docs/clean_dataset/

│

├── reference_voices/

│   ├── (41) Practical Boating MBM How to do a mayday DSC Motor Boats Monthly MBM.wav

│   ├── (54) Example Mayday Message.wav

│   ├── (135) ScotSail VHF Radio Licence - Mayday Relay Voice Call.wav

│   ├── (135) VHF Mayday call.wav

│   └── WhatsApp Audio 2026-08-01 at 08.13.34.wav

│

├── chatterbox_clean_output/

│   └── chatterbox_clean_asr_output/

│

└── fishspeech_clean_output/

    └── fishspeech_clean_asr_output/

```

Current clean reference inventory contains **5 different reference recordings**.
These recordings are intentionally separate recordings rather than repeated copies of one voice. They are used as the clean-reference pool in:

```

chatterbox_clean_ref

fishspeech_clean_ref

```

The pilot-plan script rotates through these clean references for the Vessel/source voice.
Clean-reference paths used by the current plan script:

```

workspace/docs/clean_dataset/reference_voices/

```

The generated audio for the current pilot experiment is written under:

```

workspace/data/pilot/chatterbox_clean_ref/

workspace/data/pilot/fishspeech_clean_ref/

```

The older dataset-level output folders are still kept under:

```

workspace/docs/clean_dataset/chatterbox_clean_output/

workspace/docs/clean_dataset/fishspeech_clean_output/

```

These are useful for earlier baseline outputs and ASR inspection, but the active dialogue-pilot outputs are stored under \`workspace/data/pilot/\`.

<a id="5-2-noisy-dataset"></a>
### 5.2 Noisy Dataset

Path:
[workspace/docs/Noisy\_dataset/](workspace/docs/Noisy_dataset/)
Main structure:

```

workspace/docs/Noisy_dataset/

│

├── reference_voices/

├── chatterbox_output_noisy/

│   └── chatterbox_noisy_asr_output/

└── fishspeech_output_noisy/

    └── fishspeech_noisy_asr_output/

```

The current noisy-reference folder contains the following recordings:

```

(135) City Sailing SRC VHF Radio Course May Day Distress Call tutorial.wav

(135) Mayday Relay for sinking vessel on HF 8291 kHz in Australian bush.wav

(135) Mein Schiff 7_ Erfolgreiche Übung auf der Ostsee.wav

(51) Audio captures captain’s distress call after Iran fires on Indian tanker in Hormuz.wav

(51) Mayday Call from Fishing Vessel Off New Hampshire.wav

(51) Yacht's mayday call_ 'We are sinking'.wav

(51)Mayday call jet blue airline landing medical emergency but atc stop him #airplane #atcmemes.wav

(54) Boaters mayday call.wav

(54) Mayday call from capsized vessel on Saginaw Bay.wav

coast_guard_noisy.wav

French coastguard's mayday call after boat capsized – audio.wav

Northern Belle_ coast_guard.wav

Northern_Belle.wav

WhatsApp_audio_1.wav

```

Current noisy reference inventory therefore contains **14 recordings**.
The noisy group contains more heterogeneous material than the clean group, including:

```

real distress-call recordings

VHF / radio-like recordings

maritime training or tutorial recordings

Coast Guard/operator recordings

newly added Coast Guard-specific reference voices

```

The currently designated Coast Guard reference pool is:

```

workspace/docs/Noisy_dataset/reference_voices/Northern Belle_ coast_guard.wav

workspace/docs/Noisy_dataset/reference_voices/coast_guard_noisy.wav

```

These are kept separate from the Vessel/source reference pool.
The remaining noisy references are used as the rotating Vessel/source references for:

```

chatterbox_noisy_ref

fishspeech_noisy_ref

```

<a id="5-3-current-noisy-chatterbox-outputs"></a>
### 5.3 Current Noisy Chatterbox Outputs

Path:
[workspace/docs/Noisy\_dataset/chatterbox\_output\_noisy/](workspace/docs/Noisy_dataset/chatterbox_output_noisy/)
The corresponding ASR results are stored under:
[workspace/docs/Noisy\_dataset/chatterbox\_output\_noisy/chatterbox\_noisy\_asr\_output/](workspace/docs/Noisy_dataset/chatterbox_output_noisy/chatterbox_noisy_asr_output/)
This folder contains:

```

asr_transcripts_summary.csv

per-audio transcript .txt files

per-audio segment .csv files

```

<a id="5-4-current-noisy-fish-speech-outputs"></a>
### 5.4 Current Noisy Fish-Speech Outputs

Path:
[workspace/docs/Noisy\_dataset/fishspeech\_output\_noisy/](workspace/docs/Noisy_dataset/fishspeech_output_noisy/)
The corresponding ASR results are stored under:
[workspace/docs/Noisy\_dataset/fishspeech\_output\_noisy/fishspeech\_noisy\_asr\_output/](workspace/docs/Noisy_dataset/fishspeech_output_noisy/fishspeech_noisy_asr_output/)
This folder also contains:

```

asr_transcripts_summary.csv

per-audio transcript .txt files

per-audio segment .csv files

```

<a id="5-5-clean-vs-noisy-reference-use-in-the-pilot"></a>
### 5.5 Clean vs Noisy Reference Use in the Pilot

The active generation plan uses the following mapping:

```

Chatterbox + clean reference

    → CLEAN_REFS

Fish-Speech + clean reference

    → CLEAN_REFS

Chatterbox + noisy reference

    → NOISY_REFS

Fish-Speech + noisy reference

    → NOISY_REFS

```

For every generated dialogue, the script additionally selects a separate Coast Guard reference from:

```

COAST_GUARD_REFS

```

This means the dialogue-generation stage does **not** assume that one reference voice should represent both communication roles.
Conceptually:

```

VESSEL

    ↓

clean or noisy Vessel/source reference

    ↓

Chatterbox / Fish-Speech

COAST_GUARD

    ↓

Coast Guard-specific reference

    ↓

Chatterbox / Fish-Speech

```

<a id="5-6-current-reference-audio-partitioning-method"></a>
### 5.6 Current Reference-Audio Partitioning Method

The earlier classification based on \`SNR\` and \`noise\_dBFS\` is **no longer the active reference-audio partitioning rule**.
The current method uses two complementary acoustic indicators:

1. **High-frequency energy ratio** \`R\_high\`

2. **Dynamic pause-floor signal-to-noise ratio** \`SNR\_dB\`

The method is implemented by:
[workspace/docs/tools/sort\_reference\_audio.py](workspace/docs/tools/sort_reference_audio.py)
Supporting methodology:
[workspace/docs/snr\_outputs/reference\_audio\_partitioning\_pipeline.md](workspace/docs/snr_outputs/reference_audio_partitioning_pipeline.md)
The output is written under:
[workspace/docs/sorted\_pipeline\_data/](workspace/docs/sorted_pipeline_data/)
with the following structure:

```

sorted_pipeline_data/

│

├── Node_E_Cleaned_Reference/

├── Node_F_Noisy_Reference/

└── classification_summary.csv

```

#### High-frequency energy ratio

A Fourier-domain bandwidth check is used to determine whether a recording still contains substantial speech energy above the typical narrowband/VHF region.
The energy bands are:

```

E_low  = spectral energy at frequencies <= 4000 Hz

E_high = spectral energy from 4000 Hz to 8000 Hz

```

The ratio is:

```

R_high = E_high / (E_low + epsilon)

```

where:

```

epsilon = 1e-10

```

Interpretation:

```

Higher R_high

→ more high-frequency speech content

→ more consistent with wideband/direct-microphone speech

Lower R_high

→ high-frequency attenuation already present

→ more consistent with narrowband/VHF-like audio

```

The current threshold is:

```

R_high >= 0.005

```

#### Dynamic pause-floor SNR

The second metric estimates the separation between active speech and the background/pause floor.
Short-time RMS energy is calculated using:

```

window = 30 ms

hop    = 10 ms

```

The 15th percentile of the RMS distribution is used as a dynamic estimate of the background noise floor:

```

noise_threshold = percentile_15(RMS)

```

Frames are then separated approximately as:

```

speech frames:

RMS > 2.5 \* noise_threshold

noise frames:

RMS <= noise_threshold

```

The dynamic SNR is:

```

SNR_dB = 10 \* log10(P_speech / P_noise)

```

The current threshold is:

```

SNR_dB >= 22.0 dB

```

#### Current dual-condition decision gate

A reference recording is assigned to **Node E — Clean Reference** only if **both** conditions are satisfied:

```

R_high >= 0.005

AND

SNR_dB >= 22.0 dB

```

Otherwise it is assigned to:

```

Node F — Noisy / Native-VHF Reference

```

The practical interpretation is:
\| Condition | Classification | Treatment |
\|---|---|---|
\| \`R\_high >= 0.005\` and \`SNR\_dB >= 22 dB\` | **Node E — Clean** | Suitable for clean-reference branch; VHF DSP may be applied downstream |
\| \`R\_high < 0.005\` | **Node F — VHF/narrowband** | Existing channel bandwidth limitation is already present |
\| \`R\_high >= 0.005\` but \`SNR\_dB < 22 dB\` | **Node F — Noisy** | Wideband content exists, but background/pause-floor quality is insufficient |
The two metrics are intentionally combined because they measure different properties:

```

R_high

→ spectral bandwidth / high-frequency retention

SNR_dB

→ speech-to-background-noise separation

```

Therefore, a recording is considered suitable for the clean reference branch only when it is **both sufficiently wideband and sufficiently clean**.
The current \`classification\_summary.csv\` records:

```

filename

verdict

high_freq_ratio

snr_db

reason

```

---

<a id="6-repository-structure"></a>
## 6. Repository Structure

The repository is maintained on Windows, while computationally heavy TTS generation and speaker verification run on the Linux data station. The tree below intentionally shows **project-relevant files only**; Conda internals, Python libraries, caches and downloaded model-package internals are excluded.

<a id="6-1-windows-repository-canonical-documentation-source-tree"></a>
### 6.1 Windows repository — canonical documentation/source tree

Repository root:

```text
C:\Users\kda\Documents\ma-tts-tuning
```

Current project-relevant structure:

```text
ma-tts-tuning/
├── README.md
├── .gitignore
├── admin/
│   └── README.md
├── lit/
│   ├── 1904.02882v1.pdf
│   ├── T-REC-P.80-199303-S__PDF-E.pdf
│   ├── VAD-review.pdf
│   ├── Thesis_proposal.docx
│   ├── report.docx
│   └── README.md
├── meetings/
│   └── MeetingNotes.md
└── workspace/
    ├── data/
    │   ├── diagnostics/
    │   │   ├── backend_real_validation/
    │   │   │   ├── method_summary.csv
    │   │   │   └── real_trials_results.csv
    │   │   └── failed_reference_speaker_tests/
    │   │       ├── reference_test1_summary.csv
    │   │       ├── failed_attempt_tests_2_3.csv
    │   │       └── references/
    │   │           └── <five verified-reference folders>/chunk_A.wav + chunk_B.wav
    │   └── pilot/
    │       ├── chatterbox_clean_ref/
    │       ├── chatterbox_noisy_ref/
    │       ├── fishspeech_clean_ref/
    │       ├── fishspeech_noisy_ref/
    │       └── metadata/
    │           ├── maritime_distress_normalized.xlsx
    │           ├── pilot_generation_plan.csv
    │           └── metadata/
    │               ├── dialogue_text_review.csv
    │               ├── maritime_distress_speaker_labeled.xlsx
    │               └── pilot_generation_plan_old_roger_issue.csv
    └── docs/
        ├── main_pipeline.png
        ├── mixed_dataset_methods_tts.md
        ├── model_comparison.md
        ├── clean_dataset/
        │   └── reference_voices/
        ├── Noisy_dataset/
        │   └── reference_voices/
        ├── metadata/
        ├── snr_outputs/
        ├── sorted_pipeline_data/
        └── tools/
            ├── prepare_distress_dataset.py
            ├── add_speaker_labels.py
            ├── create_pilot_generation_plan.py
            ├── run_pilot_generation.py
            ├── run_pilot_generation_dialogue.py
            ├── generate_chatterbox_one.py
            ├── generate_fishspeech_one.py
            ├── asr_transcribe.py
            ├── calc_text_metrics.py
            ├── noise_cal.py
            ├── sort_reference_audio.py
            ├── test_noisy_thresholds.py
            ├── speaker_backend_3way.py
            ├── diagnose_failed_samples.py
            ├── validate_speaker_backend_real_trials.py
            └── legacy/
                └── test_whatsapp_chunks.py
```

The latest Windows tree therefore already contains the speaker-backend implementation and the two main diagnostic scripts. The full diagnostic WAV/CSV material is also mirrored under `workspace/data/diagnostics/`.

<a id="6-2-linux-data-station-runtime"></a>
### 6.2 Linux data-station runtime

Primary runtime root:

```text
/data/kavya/conda_envs/chatterbox
```

Project-relevant files under that location include:

```text
/data/kavya/conda_envs/chatterbox/
├── run_pilot_generation_dialogue.py
├── create_pilot_generation_plan.py
├── generate_chatterbox_one.py
├── generate_fishspeech_one.py
├── speaker_backend_3way.py
├── diagnose_failed_samples.py
├── validate_speaker_backend_real_trials.py
├── test_whatsapp_chunks.py                  # legacy diagnostic
├── workspace/
│   ├── data/
│   │   ├── pilot/                           # complete generation artefacts
│   │   └── diagnostics/
│   │       ├── backend_real_validation/
│   │       └── failed_reference_speaker_tests/
│   ├── docs/
│   │   ├── clean_dataset/reference_voices/
│   │   └── Noisy_dataset/reference_voices/
│   └── models/
│       ├── speaker_backend_3way.pkl
│       └── spkrec-ecapa-voxceleb/            # downloaded/cached model links
└── <Conda environment internals omitted>
```

The data-station directory is also a Conda environment, so it contains `bin/`, `lib/`, `include/`, `share/`, `conda-meta/`, installed Chatterbox package code and other environment files. Those are **runtime dependencies, not thesis repository content**, and should not be copied into Git.

<a id="6-3-fish-speech-repository-runtime"></a>
### 6.3 Fish-Speech repository/runtime

Fish-Speech is served separately from:

```text
/data/kavya/fishspeech
```

The generation pipeline calls its HTTP API at:

```text
http://127.0.0.1:8080/v1/tts
```

<a id="6-4-canonical-location-principle"></a>
### 6.4 Canonical-location principle

```text
Windows repository
→ source code, documentation, compact result CSVs, reference manifests,
  selected reproducibility artefacts and thesis history

Data station
→ full Conda runtime, downloaded model caches, complete turn attempts,
  large WAV collections and computational experiment outputs
```

Do not mirror the complete Conda environment into the Windows repository.

---

<a id="7-important-project-files"></a>
## 7. Important Project Files

<a id="documentation"></a>
### Documentation

\| File | Purpose |
\|---|---|
\| [workspace/docs/main\_pipeline.png](workspace/docs/main_pipeline.png) | Main pipeline diagram |
\| [workspace/docs/mixed\_dataset\_methods\_tts.md](workspace/docs/mixed_dataset_methods_tts.md) | Dataset/fine-tuning experiment notes |
\| [workspace/docs/model\_comparison.md](workspace/docs/model_comparison.md) | Model comparison observations |
\| [meetings/MeetingNotes.md](meetings/MeetingNotes.md) | Project meeting notes |

<a id="dataset-directories"></a>
### Dataset directories

\| Directory | Purpose |
\|---|---|
\| [workspace/docs/clean\_dataset/](workspace/docs/clean_dataset/) | Clean-source reference voices, earlier clean outputs, and clean ASR folders |
\| [workspace/docs/clean\_dataset/reference\_voices/](workspace/docs/clean_dataset/reference_voices/) | Current clean reference-voice pool |
\| [workspace/docs/Noisy\_dataset/](workspace/docs/Noisy_dataset/) | Noisy/VHF-like references, generated outputs, and ASR folders |
\| [workspace/docs/Noisy\_dataset/reference\_voices/](workspace/docs/Noisy_dataset/reference_voices/) | Current noisy reference-voice pool including Coast Guard references |
\| [workspace/docs/Noisy\_dataset/chatterbox\_output\_noisy/](workspace/docs/Noisy_dataset/chatterbox_output_noisy/) | Existing noisy-source Chatterbox outputs |
\| [workspace/docs/Noisy\_dataset/fishspeech\_output\_noisy/](workspace/docs/Noisy_dataset/fishspeech_output_noisy/) | Existing noisy-source Fish-Speech outputs |

<a id="metadata"></a>
### Metadata

\| File | Purpose |
\|---|---|
\| [workspace/docs/metadata/maritime\_distress\_formatted (version 1).xlsx](workspace/docs/metadata/maritime_distress_formatted%20(version%201).xlsx) | Canonical source maritime dialogue workbook |
\| [workspace/data/pilot/metadata/maritime\_distress\_normalized.xlsx](workspace/data/pilot/metadata/maritime_distress_normalized.xlsx) | Current normalized 500-dialogue / five-scenario workbook |
\| [workspace/data/pilot/metadata/metadata/maritime\_distress\_speaker\_labeled.xlsx](workspace/data/pilot/metadata/metadata/maritime_distress_speaker_labeled.xlsx) | Current tree location of the speaker-labelled workbook; recommended to flatten later |
\| [workspace/data/pilot/metadata/pilot\_generation\_plan.csv](workspace/data/pilot/metadata/pilot_generation_plan.csv) | Current generation-plan filename; preserve experiment-specific archived copies |
\| [workspace/docs/sorted\_pipeline\_data/classification\_summary.csv](workspace/docs/sorted_pipeline_data/classification_summary.csv) | **Current** clean/noisy reference partition result (`R_high + dynamic SNR`) |
\| [workspace/docs/metadata/noisy\_threshold\_test\_results.csv](workspace/docs/metadata/noisy_threshold_test_results.csv) | Legacy SNR/noise-dBFS threshold-test results |
\| [workspace/docs/metadata/tts\_dataset\_metadata\_step4\_filled.xlsx](workspace/docs/metadata/tts_dataset_metadata_step4_filled.xlsx) | Legacy earlier single-output evaluation workbook |
\| [workspace/docs/metadata/tts\_dataset\_metadata\_with\_metrics.xlsx](workspace/docs/metadata/tts_dataset_metadata_with_metrics.xlsx) | Legacy earlier metric-enriched evaluation workbook |

---

<a id="8-commands-complete-current-workflow"></a>
## 8. Commands — Complete Current Workflow

Run the Windows-side commands from the repository root:

```
C:\Users\kda\Documents\ma-tts-tuning
```

The current end-to-end order is:

```
1. prepare_distress_dataset.py
2. add_speaker_labels.py
3. review REVIEW_REQUIRED turns
4. create_pilot_generation_plan.py
5. start Fish-Speech API server
6. run_pilot_generation_dialogue.py --dry_run
7. generate a small test
8. run full dialogue generation
9. ASR transcription
10. text/acoustic evaluation
```

<a id="8-1-check-the-available-command-line-arguments"></a>
### 8.1 Check the available command-line arguments

Use `-h` whenever a local script version has changed:

```
python .\workspace\docs\tools\prepare_distress_dataset.py -h
python .\workspace\docs\tools\add_speaker_labels.py -h
python .\workspace\docs\tools\create_pilot_generation_plan.py -h
```

On the data station, for the dialogue generator:

```
python run_pilot_generation_dialogue.py -h
```

---

<a id="8-2-prepare-normalize-the-multi-scenario-distress-dataset"></a>
### 8.2 Prepare / normalize the multi-scenario distress dataset

This is the first step for the current mixed workbook.

```
python .\workspace\docs\tools\prepare_distress_dataset.py `
  --input_xlsx ".\workspace\docs\metadata\maritime_distress_formatted (version 1).xlsx" `
  --output_xlsx ".\workspace\data\pilot\metadata\maritime_distress_normalized.xlsx"
```

For the current workbook, `armed_attack` is the default base scenario. It can also be stated explicitly:

```
python .\workspace\docs\tools\prepare_distress_dataset.py `
  --input_xlsx ".\workspace\docs\metadata\maritime_distress_formatted (version 1).xlsx" `
  --base_scenario armed_attack `
  --output_xlsx ".\workspace\data\pilot\metadata\maritime_distress_normalized.xlsx"
```

If future distress scenarios are supplied as separate JSON task files, add them with repeated `--extra_json` arguments:

```
python .\workspace\docs\tools\prepare_distress_dataset.py `
  --input_xlsx ".\workspace\docs\metadata\maritime_distress_formatted (version 1).xlsx" `
  --base_scenario armed_attack `
  --extra_json ".\workspace\docs\metadata\fire.json" `
  --extra_json ".\workspace\docs\metadata\flooding.json" `
  --extra_json ".\workspace\docs\metadata\grounding.json" `
  --output_xlsx ".\workspace\data\pilot\metadata\maritime_distress_normalized.xlsx"
```

Output:

```
workspace/data/pilot/metadata/maritime_distress_normalized.xlsx
```

The workbook contains a normalized `dialogues` sheet and a scenario-count `summary` sheet.

---

<a id="8-3-add-vessel-coast-guard-speaker-role-labels"></a>
### 8.3 Add Vessel / Coast Guard speaker-role labels

Use the normalized workbook produced in Step 8.2:

```
python .\workspace\docs\tools\add_speaker_labels.py `
  --input_xlsx ".\workspace\data\pilot\metadata\maritime_distress_normalized.xlsx" `
  --output_xlsx ".\workspace\data\pilot\metadata\maritime_distress_speaker_labeled.xlsx"
```

Output:

```
workspace/data/pilot/metadata/maritime_distress_speaker_labeled.xlsx
```

Before creating the TTS plan, inspect all turns marked:

```
REVIEW_REQUIRED
```

The speaker-label workbook contains the role decision and the detailed `speaker_label_report`.

---

<a id="8-4-create-the-pilot-generation-plan"></a>
### 8.4 Create the pilot generation plan

For the current multi-scenario version, select up to **100 dialogues per scenario**:

```
python .\workspace\docs\tools\create_pilot_generation_plan.py `
  --input_xlsx ".\workspace\data\pilot\metadata\maritime_distress_speaker_labeled.xlsx" `
  --num_texts_per_scenario 100 `
  --output_csv ".\workspace\data\pilot\metadata\pilot_generation_plan.csv"
```

Output:

```
workspace/data/pilot/metadata/pilot_generation_plan.csv
```

The plan expands each accepted source dialogue into the four experimental conditions:

> **Version-control warning:** an older uploaded 400-row plan represented 100 dialogues × 4 conditions and had a blank `scenario_type` column. Treat that file as legacy. A full plan generated from the current 500-dialogue, five-scenario normalized workbook would contain up to 2000 rows when all 100 dialogues per scenario are selected.


```
chatterbox_clean_ref
fishspeech_clean_ref
chatterbox_noisy_ref
fishspeech_noisy_ref
```

For example, with the current five scenarios and 100 dialogues per scenario:

```
5 scenarios × 100 dialogues × 4 conditions = 2000 generation rows
```

> If your local copy of `create_pilot_generation_plan.py -h` still shows `--num_texts` instead of `--num_texts_per_scenario`, then you are running the older script and should replace it with the current multi-scenario version.

---

<a id="8-5-start-the-fish-speech-api-server"></a>
### 8.5 Start the Fish-Speech API server

Run this on the data station from the Fish-Speech repository:

```
cd /data/kavya/fishspeech

python tools/api_server.py \
  --llama-checkpoint-path checkpoints/s2-pro \
  --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
  --listen 0.0.0.0:8080
```

Expected endpoint:

```
http://127.0.0.1:8080/v1/tts
```

Keep this server running while Fish-Speech rows are generated.

---

<a id="8-6-validate-the-dialogue-plan-without-generating-audio"></a>
### 8.6 Validate the dialogue plan without generating audio

Run from the Chatterbox/project environment:

```
cd /data/kavya/conda_envs/chatterbox
```

Then run a dry check:

```
python run_pilot_generation_dialogue.py \
  --plan workspace/data/pilot/metadata/pilot_generation_plan.csv \
  --limit 20 \
  --dry_run
```

The dry run checks:

```
dialogue parsing
VESSEL / COAST_GUARD labels
selected reference voice
model condition
output path
```

It writes metadata but does **not** synthesize the final WAV files.

---

<a id="8-7-generate-a-small-audio-test"></a>
### 8.7 Generate a small audio test

After the dry run succeeds:

```
python run_pilot_generation_dialogue.py \
  --plan workspace/data/pilot/metadata/pilot_generation_plan.csv \
  --limit 20 \
  --fish_url http://127.0.0.1:8080/v1/tts
```

Use a small test before running the complete plan.
If existing outputs need to be regenerated:

```
python run_pilot_generation_dialogue.py \
  --plan workspace/data/pilot/metadata/pilot_generation_plan.csv \
  --limit 20 \
  --fish_url http://127.0.0.1:8080/v1/tts \
  --overwrite
```

---

<a id="8-8-run-the-complete-dialogue-generation"></a>
### 8.8 Run the complete dialogue generation

For the current five-scenario normalized corpus with 100 dialogues per scenario and four conditions:

```
python run_pilot_generation_dialogue.py \
  --plan workspace/data/pilot/metadata/pilot_generation_plan.csv \
  --limit 2000 \
  --fish_url http://127.0.0.1:8080/v1/tts
```

If the generation plan contains a different number of rows, use the actual CSV row count or omit/change the limit according to the local script behavior.
Generated dialogue outputs are stored under:

```
workspace/data/pilot/chatterbox_clean_ref/
workspace/data/pilot/chatterbox_noisy_ref/
workspace/data/pilot/fishspeech_clean_ref/
workspace/data/pilot/fishspeech_noisy_ref/
```

Each dialogue also has individual turn WAV files and `turns_metadata.csv`.

---

<a id="8-9-test-one-chatterbox-sample"></a>
### 8.9 Test one Chatterbox sample

This helper command is useful for checking the Chatterbox environment independently of the full dialogue pipeline:

```
python .\workspace\docs\tools\generate_chatterbox_one.py `
  --text "MAYDAY MAYDAY MAYDAY. This is motor vessel F R B Zero Niner. OVER." `
  --reference_audio ".\workspace\docs\clean_dataset\reference_voices\WhatsApp Audio 2026-08-01 at 08.13.34.wav" `
  --output_audio ".\workspace\data\pilot\test_chatterbox.wav"
```

---

<a id="8-10-test-one-fish-speech-sample"></a>
### 8.10 Test one Fish-Speech sample

Start the Fish-Speech API first, then run the helper script from the environment where it is installed:

```
python generate_fishspeech_one.py \
  --text "MAYDAY MAYDAY MAYDAY. This is motor vessel F R B Zero Niner. OVER." \
  --output_audio "reference_outputs/test_fishspeech.wav" \
  --url "http://127.0.0.1:8080/v1/tts"
```

If your local helper version exposes reference-audio/reference-text arguments, check them with:

```
python generate_fishspeech_one.py -h
```

---

<a id="8-11-run-asr-on-the-four-current-pilot-output-conditions"></a>
### 8.11 Run ASR on the four current pilot output conditions

#### Chatterbox clean-reference outputs

```
python .\workspace\docs\tools\asr_transcribe.py `
  --audio_dir ".\workspace\data\pilot\chatterbox_clean_ref" `
  --out_dir ".\workspace\data\pilot\chatterbox_clean_ref\asr_output" `
  --device cpu `
  --compute_type int8
```

#### Chatterbox noisy-reference outputs

```
python .\workspace\docs\tools\asr_transcribe.py `
  --audio_dir ".\workspace\data\pilot\chatterbox_noisy_ref" `
  --out_dir ".\workspace\data\pilot\chatterbox_noisy_ref\asr_output" `
  --device cpu `
  --compute_type int8
```

#### Fish-Speech clean-reference outputs

```
python .\workspace\docs\tools\asr_transcribe.py `
  --audio_dir ".\workspace\data\pilot\fishspeech_clean_ref" `
  --out_dir ".\workspace\data\pilot\fishspeech_clean_ref\asr_output" `
  --device cpu `
  --compute_type int8
```

#### Fish-Speech noisy-reference outputs

```
python .\workspace\docs\tools\asr_transcribe.py `
  --audio_dir ".\workspace\data\pilot\fishspeech_noisy_ref" `
  --out_dir ".\workspace\data\pilot\fishspeech_noisy_ref\asr_output" `
  --device cpu `
  --compute_type int8
```

The ASR script creates the summary transcript CSV and per-audio transcript/segment files.

---

<a id="8-12-calculate-text-metrics"></a>
### 8.12 Calculate text metrics

For the existing evaluation workbook:

```
python .\workspace\docs\tools\calc_text_metrics.py `
  --input ".\workspace\docs\metadata\tts_dataset_metadata_step4_filled.xlsx" `
  --sheet "Metadata Updated" `
  --output ".\workspace\docs\metadata\tts_dataset_metadata_with_metrics.xlsx"
```

Output:

```
workspace/docs/metadata/tts_dataset_metadata_with_metrics.xlsx
```

The script calculates metrics such as:

```
WER
CER
keyword accuracy
number accuracy
```

---

<a id="8-13-run-the-current-reference-audio-partitioning-method"></a>
### 8.13 Run the current reference-audio partitioning method

The active reference classification method uses:

```
R_high >= 0.005
AND
SNR_dB >= 22.0 dB
```

The current `sort_reference_audio.py` implementation contains `INPUT_AUDIO_DIR` and `OUTPUT_SORTED_DIR` in the script itself. After checking those paths, run:

```
python .\workspace\docs\tools\sort_reference_audio.py
```

Expected output structure:

```
workspace/docs/sorted_pipeline_data/
├── Node_E_Cleaned_Reference/
├── Node_F_Noisy_Reference/
└── classification_summary.csv
```

The current implementation and decision rule are documented in:
[workspace/docs/snr\_outputs/reference\_audio\_partitioning\_pipeline.md](workspace/docs/snr_outputs/reference_audio_partitioning_pipeline.md)

---

<a id="8-14-earlier-auxiliary-snr-noise-analysis"></a>
### 8.14 Earlier / auxiliary SNR-noise analysis

`noise_cal.py` is retained for auxiliary/older analysis. For example, to calculate the old SNR/noise output for all Fish-Speech noisy prototype WAVs:

```
Get-ChildItem ".\workspace\docs\Noisy_dataset\fishspeech_output_noisy" -Filter "*.wav" | ForEach-Object {
    python .\workspace\docs\tools\noise_cal.py $_.FullName |
      Out-File ".\workspace\docs\snr_outputs\fishspeech_noisy_snr\$($_.BaseName)_noise.txt"
}
```

This is **not** the current clean/noisy partitioning gate. The active partitioning gate is the `R_high + SNR_dB` method in `sort_reference_audio.py`.

---

<a id="8-15-earlier-threshold-testing-utility"></a>
### 8.15 Earlier threshold-testing utility

The older threshold-testing script is retained for experiment history:

```powershell
python .\workspace\docs\tools\test_noisy_thresholds.py
```

Historical input used by this experiment:

```text
workspace/docs/metadata/noisy_reference_inventory.csv
```

The latest Windows tree no longer contains that clean CSV copy; only the exploratory `noisy_reference_inventory.csv.xlsx` workbook is visible. Therefore restore/archive the original CSV before attempting exact reproduction of this legacy threshold sweep.

Historical output:

```text
workspace/docs/metadata/noisy_threshold_test_results.csv
```

This is **not** the current clean/noisy decision rule. The active partition result is `workspace/docs/sorted_pipeline_data/classification_summary.csv`.

---

<a id="8-16-diagnose-failed-generated-samples-with-the-current-3-way-backend"></a>
### 8.16 Diagnose failed generated samples with the current 3-way backend

Run on the data station:

```bash
cd /data/kavya/conda_envs/chatterbox
python diagnose_failed_samples.py
```

Defaults used by the diagnostic script:

```text
plan:
workspace/data/pilot/metadata/pilot_generation_plan.csv

backend model:
workspace/models/speaker_backend_3way.pkl

output directory:
workspace/data/diagnostics/failed_reference_speaker_tests

candidate chunk duration:
6.0 s
```

Equivalent explicit command:

```bash
cd /data/kavya/conda_envs/chatterbox
python diagnose_failed_samples.py \
  --plan workspace/data/pilot/metadata/pilot_generation_plan.csv \
  --backend-model workspace/models/speaker_backend_3way.pkl \
  --out-dir workspace/data/diagnostics/failed_reference_speaker_tests \
  --chunk-sec 6.0
```

Main outputs:

- [reference_test1_summary.csv](workspace/data/diagnostics/failed_reference_speaker_tests/reference_test1_summary.csv)
- [failed_attempt_tests_2_3.csv](workspace/data/diagnostics/failed_reference_speaker_tests/failed_attempt_tests_2_3.csv)
- generated A/B diagnostic chunks under [references/](workspace/data/diagnostics/failed_reference_speaker_tests/references/)

<a id="8-17-validate-the-speaker-backend-on-real-same-different-controls"></a>
### 8.17 Validate the speaker backend on real SAME/DIFFERENT controls

Run on the data station:

```bash
cd /data/kavya/conda_envs/chatterbox
python validate_speaker_backend_real_trials.py
```

The script uses the five manually verified A/B same-speaker controls and two Coast Guard references as different-speaker controls. It writes:

- [real_trials_results.csv](workspace/data/diagnostics/backend_real_validation/real_trials_results.csv)
- [method_summary.csv](workspace/data/diagnostics/backend_real_validation/method_summary.csv)

<a id="8-18-legacy-whatsapp-only-backend-control"></a>
### 8.18 Legacy WhatsApp-only backend control

The earlier diagnostic is retained only for history:

```bash
cd /data/kavya/conda_envs/chatterbox
python test_whatsapp_chunks.py
```

Repository copy:

- [workspace/docs/tools/legacy/test_whatsapp_chunks.py](workspace/docs/tools/legacy/test_whatsapp_chunks.py)

The generalized `diagnose_failed_samples.py` supersedes this single-reference test.

---

<a id="9-fish-speech-server"></a>
## 9. Fish-Speech Server

The Fish-Speech server command is documented in **Section 8.5**.
Use:

```
/data/kavya/fishspeech
```

as the Fish-Speech repository location, and keep the API server active while running Fish-Speech generation.

---

<a id="10-evaluation-of-generated-audio"></a>
## 10. Evaluation of generated audio

<a id="important-why-the-old-threshold-method-was-replaced"></a>
## IMPORTANT: Why the Old Threshold Method Was Replaced

This is one of the main methodological changes in the project.

<a id="earlier-method"></a>
### Earlier method

The earlier speaker-consistency check used:

```text
ECAPA-TDNN speaker embedding
        ↓
raw cosine similarity
        ↓
manually selected threshold
        ↓
SAME / DIFFERENT decision
```

Development values included approximate rules such as:

```text
clean-reference similarity       >= 0.45
noisy Vessel similarity          >= 0.30
same-role anchor similarity      >= 0.55
later stricter anchor considered >= 0.70
```

These values were useful for debugging, but they were not calibrated probabilities.

<a id="why-this-became-a-problem"></a>
### Why this became a problem

There was no single raw-cosine threshold that behaved consistently for clean, noisy, VHF-like, and HF-like references.

For example:

```text
same speaker + clean reference
→ cosine around 0.70 or higher

same-ish speaker + noisy/VHF reference
→ cosine around 0.30–0.40
```

If one threshold such as `0.45` was used for everything, noisy-reference generations could be rejected even when the speaker was reasonably similar.

If the noisy threshold was lowered to `0.30`, weaker or audibly drifting voices could also pass.

So the threshold had two opposite risks:

```text
threshold too high
→ false reject

threshold too low
→ false accept
```

<a id="why-the-intended-vs-other-margin-was-not-enough"></a>
### Why the intended-vs-other margin was not enough

The earlier QC also calculated:

```text
margin =
similarity to intended reference
-
similarity to other-role reference
```

Example:

```text
generated ↔ intended Vessel = 0.35
generated ↔ Coast Guard     = 0.02

margin = 0.33
```

The margin is large, but the absolute intended-speaker score is still only `0.35`.

Therefore a large margin means:

```text
"this generated voice is closer to the Vessel than to this Coast Guard"
```

but it does not necessarily mean:

```text
"this generated voice is a strong match to the true Vessel speaker"
```

<a id="why-the-anchor-was-not-enough"></a>
### Why the anchor was not enough

The first accepted generated turn of a role was used as an anchor:

```text
Vessel Turn 1
     ↓
accepted
     ↓
anchor
```

Later turns were compared with that anchor.

This helps detect within-dialogue drift, but if Turn 1 is already an imperfect clone:

```text
true Vessel reference
       ↓
generated Turn 1 = slightly wrong
       ↓
Turn 1 becomes anchor
```

later turns may become consistent with the imperfect generated voice instead of staying faithful to the original reference speaker.

<a id="practical-false-pass-behavior"></a>
### Practical false-pass behavior

During pilot testing, some noisy Fish-Speech Vessel turns produced anchor similarities around:

```text
0.57–0.60
```

and could pass an anchor threshold around:

```text
0.55
```

while manual listening still detected obvious speaker drift.

Stronger same-speaker generations could be around:

```text
0.77–0.84
```

The important observation was:

```text
automatic threshold QC → PASS
manual listening        → obvious speaker drift
```

This showed that the earlier threshold method could produce false passes.

<a id="why-noisy-vhf-audio-made-the-problem-harder"></a>
### Why noisy/VHF audio made the problem harder

The reference and generated speech can have very different acoustic conditions:

```text
REFERENCE
radio/VHF/HF filtering
compression
static
interference
band-limited speech

GENERATED TTS
cleaner synthetic speech
different spectral balance
different noise level
```

Speaker embeddings are designed to represent identity, but channel mismatch and noise can still affect the embedding and therefore the cosine score.

This is why the same raw threshold is difficult to justify across all recording conditions.

<a id="why-the-project-moved-to-speaker-backend-3way-py"></a>
### Why the project moved to `speaker_backend_3way.py`

Because of these limitations, the project moved from:

```text
ECAPA cosine
+
hand-selected threshold
```

to:

```text
speaker_backend_3way.py
```

which uses:

```text
1. calibrated cosine
2. PLDA
3. cohort-normalized cosine
```

followed by:

```text
2-of-3 SAME / DIFFERENT decision
```

The improvement is that final speaker identity is no longer controlled by one manually selected raw-cosine threshold alone.

Supporting references:

- [ECAPA-TDNN — Desplanques et al. (2020)](https://arxiv.org/abs/2005.07143)
- [BOSARIS score calibration — Brümmer & de Villiers (2013)](https://arxiv.org/abs/1304.2865)
- [PLDA — Ioffe (2006)](https://doi.org/10.1007/11744085_41)
- [Cohort normalization — Rosenberg et al. (1992)](https://www.isca-archive.org/icslp_1992/rosenberg92_icslp.html)
- [S-norm — Shum et al. (2010)](https://www.isca-archive.org/odyssey_2010/shum10_odyssey.html)
- [Score normalization under mismatch — Matějka et al. (2017)](https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html)

---

<a id="10-1-speaker-identity-qc-why-the-method-changed"></a>
### 10.1 Speaker-identity QC: why the method changed

Speaker-role labels alone are not enough. Each turn is synthesized independently, so the model may produce a noticeably different Vessel or Coast Guard voice in a later turn even when the same reference WAV is provided.
The speaker-QC method therefore evolved in three stages.

#### Stage 1 — generation without speaker verification

Originally the pipeline generated and concatenated turns successfully, but there was no reliable automatic answer to:

```
Does Vessel Turn 7 still sound like the same physical Vessel speaker as Vessel Turn 1?
```

A successful TTS request was therefore not the same as a successful dialogue.

#### Stage 2 — ECAPA-TDNN + raw cosine checks

ECAPA-TDNN was then used to convert audio into speaker embeddings. The generated turn was compared with:

```
1. intended role reference
2. competing role reference
3. first accepted same-role generated turn (anchor)
```

This produced useful quantities such as:

```
similarity to intended reference
similarity to other reference
margin = intended similarity - other similarity
similarity to same-role anchor
```

This was useful for debugging, but the final decision still depended on hand-selected cosine thresholds. Examples used during development included approximately:

```
clean-reference similarity threshold  ≈ 0.45
noisy Vessel similarity threshold     ≈ 0.30
anchor similarity threshold           ≈ 0.55
later stricter anchor considered      ≈ 0.70
```

These values are **legacy debugging heuristics**, not calibrated probabilities and not final scientific decision boundaries.

##### Why raw cosine thresholds were not enough

Clean and noisy/VHF references did not produce the same score range. A reasonably similar speaker under noisy conditioning could have a lower cosine score than a clean-reference generation. Lowering the threshold helped recall, but could also allow audibly drifting voices to pass.
A large intended-vs-other margin was also insufficient by itself. For example:

```
intended similarity = 0.35
other-role similarity = 0.02
margin = 0.33
```

The large margin mainly says the output is **closer to Vessel than to this Coast Guard reference**; it does not prove that `0.35` is a strong absolute identity match.
The anchor comparison also has a limitation. If the first generated Vessel turn is already imperfect, later turns can become consistent with the imperfect anchor instead of remaining close to the real reference speaker.
This led to cases where:

```
automatic QC = PASS
human listening = obvious speaker drift
```

That false-pass behavior motivated the current 3-way backend.


<a id="10-1a-detailed-speaker-qc-evolution-why-the-3-way-backend-was-needed"></a>
### 10.1A Detailed Speaker-QC Evolution — Why the 3-Way Backend Was Needed

This section records the complete reasoning behind the move from simple ECAPA cosine checks to `speaker_backend_3way.py`.

#### 10.1A.1 First problem: generation succeeded, but speaker identity could drift

The original generation pipeline synthesized every radio turn independently:

```text
VESSEL Turn 1 → generate
COAST_GUARD Turn 2 → generate
VESSEL Turn 3 → generate
COAST_GUARD Turn 4 → generate
VESSEL Turn 5 → generate
...
→ concatenate
```

Using the same Vessel reference for several turns does not force the TTS model to reproduce exactly the same vocal identity every time. Different stochastic generations can change:

```text
pitch
timbre
accent
speaking rate
vocal tension
prosody
```

Therefore a later Vessel turn could sound like another person even though the correct Vessel reference path was supplied.

The original pipeline could answer:

```text
"Did the model return a WAV file?"
```

but it could not reliably answer:

```text
"Is this still the intended Vessel speaker?"
```

This problem was especially noticeable in some Fish-Speech noisy-reference outputs, but the general problem applies to independently generated turns from either model.

#### 10.1A.2 First speaker-verification improvement: ECAPA-TDNN embeddings

The next step used **ECAPA-TDNN** to convert speech into a fixed-dimensional speaker embedding.

Reference:

- [Desplanques, Thienpondt & Demuynck (2020), *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*](https://arxiv.org/abs/2005.07143)

Conceptually:

```text
audio waveform
     ↓
ECAPA-TDNN
     ↓
speaker embedding vector
```

Two recordings could then be compared using cosine similarity.

The generated turn was compared against three useful targets:

```text
1. intended reference speaker
2. competing role reference
3. first accepted generated turn of the same role (anchor)
```

This produced quantities such as:

```text
sim_intended
sim_other
margin = sim_intended - sim_other
sim_anchor
```

This was much better than having no speaker verification.

#### 10.1A.3 Why one raw cosine threshold became a problem

The first QC versions used manually selected thresholds such as approximately:

```text
clean reference similarity      >= 0.45
noisy Vessel similarity         >= 0.30
same-role anchor similarity     >= 0.55
later stricter anchor considered >= 0.70
```

These numbers were useful during debugging, but they were not calibrated probabilities.

The problem was that clean and noisy/VHF recordings produced different score ranges.

For example:

```text
noisy reference + reasonable generated speaker
→ cosine around 0.33

clean reference + same-speaker generation
→ cosine around 0.70 or higher
```

If the same threshold of `0.45` were used everywhere:

```text
some reasonable noisy-reference generations
→ false reject
```

If the noisy threshold were reduced to `0.30`:

```text
weaker or drifting voices
→ could now pass
```

So the question:

> "Why exactly is 0.45 correct?"

did not have a strong scientific answer. The value came from observed pilot behavior rather than a calibrated same/different speaker model.

#### 10.1A.4 Why the old threshold values should be treated as legacy heuristics

Values such as:

```text
0.30
0.45
0.55
0.70
```

should be documented as **development heuristics**, not as probabilities or universal speaker-verification boundaries.

They helped inspect the data, but the score distribution depends on:

```text
speaker
reference quality
microphone/channel
VHF/HF filtering
background noise
compression
duration
generated speech style
```

This is why the current speaker backend tries to learn or normalize the decision rather than rely on one manually chosen raw score.

#### 10.1A.5 Why the intended-vs-other margin was useful but insufficient

The old pipeline also used:

```text
margin =
similarity to intended reference
-
similarity to competing-role reference
```

Example:

```text
generated turn ↔ Vessel reference = 0.35
generated turn ↔ Coast Guard ref  = 0.02

margin = 0.33
```

The margin is large, but `0.35` is not necessarily a strong absolute speaker match.

The large margin can happen simply because the Coast Guard comparison is extremely poor.

Therefore the margin answers:

```text
"Does this sound more like the intended Vessel than this particular Coast Guard?"
```

It does **not** fully answer:

```text
"Is this a strong match to the intended Vessel speaker?"
```

Margin remains useful diagnostic information, but it should not be the only identity decision.

#### 10.1A.6 Why the same-role anchor was useful but also limited

The next improvement used the first accepted role turn as an anchor:

```text
Vessel Turn 1
     ↓ accepted
     ↓
Vessel anchor

Turn 3 ↔ Turn 1
Turn 5 ↔ Turn 1
Turn 7 ↔ Turn 1
```

This is useful for detecting **within-dialogue drift**.

However, suppose Turn 1 is already an imperfect clone:

```text
true Vessel reference
       ↓
generated Turn 1 = slightly wrong
       ↓
Turn 1 becomes anchor
```

Later turns may then pass because they are consistent with the imperfect Turn 1.

The anchor check can therefore become:

```text
"Stay consistent with the first generated approximation."
```

instead of:

```text
"Stay faithful to the original physical speaker."
```

The anchor is valuable as an additional check, but should not replace direct verification against the intended reference.

#### 10.1A.7 Practical false-pass evidence

The reason for changing the QC was not only theoretical.

During noisy-reference Fish-Speech experiments, later Vessel turns could obtain approximately:

```text
anchor similarity ≈ 0.57–0.60
```

and pass an anchor threshold around:

```text
0.55
```

while manual listening still identified an obvious change in Vessel voice.

More stable generations were often much higher, for example around:

```text
0.77–0.84
```

The important observation was therefore:

```text
automatic QC → PASS
human listening → obvious speaker drift
```

This showed that a permissive single-threshold system could produce false passes.

#### 10.1A.8 Why noisy/VHF recordings make speaker verification harder

Noisy maritime reference recordings and generated TTS output can have very different acoustic conditions.

Example:

```text
REFERENCE
radio/VHF/HF channel
static
band-limited speech
compression
fading/interference

GENERATED TURN
cleaner synthetic speech
different spectral balance
different noise level
```

ECAPA is designed to represent speaker information, but its embedding can still be affected by acoustic mismatch.

That means a lower cosine score can be caused by:

```text
speaker difference
and/or
channel difference
and/or
noise difference
```

This is one reason why clean and noisy references should not automatically be expected to share the same raw-score distribution.

The importance of normalization under mismatched conditions is also discussed in speaker-recognition literature:

- [Matějka et al. (2017), *Analysis of Score Normalization in Multilingual Speaker Recognition*](https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html)

#### 10.1A.9 Another limitation: the old system still had one basic scoring view

The old QC had multiple comparisons:

```text
generated ↔ intended reference
generated ↔ other-role reference
generated ↔ same-role anchor
```

but all of them were still fundamentally:

```text
ECAPA embedding
+
raw cosine similarity
```

So it was not truly three independent scoring methods.

That motivated the move to `speaker_backend_3way.py`.

#### 10.1A.10 Current design: three scoring views from one ECAPA representation

The current backend keeps ECAPA as the shared embedding extractor:

```text
                     ECAPA embedding
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
      calibrated cosine   PLDA   cohort-normalized cosine
              │            │            │
              └────────────┼────────────┘
                           ▼
                     2-of-3 vote
                           ▼
                    SAME / DIFFERENT
```

The three branches are not three separate neural speaker models. They are three different interpretations of the same underlying ECAPA speaker representation.

That is still stronger than making the final decision from one raw cosine threshold.

#### 10.1A.11 Method 1 — calibrated cosine

Raw cosine similarity is still useful.

Instead of declaring:

```text
cosine >= 0.45 → SAME
```

the current idea is to collect development examples labelled:

```text
SAME speaker
DIFFERENT speaker
```

and train a logistic-regression calibrator.

Conceptually:

```text
raw cosine
    ↓
logistic calibration
    ↓
calibrated evidence for SAME vs DIFFERENT
```

The important improvement is that the decision relationship is estimated from development data rather than defined only by a hand-written raw threshold.

Calibration/evaluation background:

- [Brümmer & de Villiers (2013), *The BOSARIS Toolkit: Theory, Algorithms and Code for Surviving the New DCF*](https://arxiv.org/abs/1304.2865)

BOSARIS also provides useful background for reporting detection cost, equal-error rate and calibration behavior.

#### 10.1A.12 Method 2 — PLDA

**Probabilistic Linear Discriminant Analysis (PLDA)** models the distribution of embeddings in terms of:

```text
within-speaker variation
versus
between-speaker variation
```

Very simply, cosine asks:

```text
"How close are these two vectors?"
```

PLDA asks something closer to:

```text
"Under the learned speaker-embedding distribution,
is it more plausible that these two embeddings came
from the same speaker or from different speakers?"
```

Foundational reference:

- [Ioffe (2006), *Probabilistic Linear Discriminant Analysis*](https://doi.org/10.1007/11744085_41)

PLDA therefore provides a second scoring view rather than relying only on geometric cosine distance.

#### 10.1A.13 Method 3 — cohort-normalized cosine

A raw similarity score is difficult to interpret in isolation.

Suppose:

```text
generated turn ↔ intended reference = 0.48
```

Case A:

```text
similarity to unrelated speakers:
0.10, 0.12, 0.05, 0.08, 0.14
```

Then `0.48` is unusually high relative to the background population.

Case B:

```text
similarity to unrelated speakers:
0.45, 0.47, 0.44, 0.46
```

Then `0.48` is not very distinctive.

Cohort normalization asks whether the target score is unusual relative to non-target speakers.

Classic and modern references:

- [Rosenberg et al. (1992), *The use of cohort normalized scores for speaker verification*](https://www.isca-archive.org/icslp_1992/rosenberg92_icslp.html)
- [Shum et al. (2010), *Unsupervised Speaker Adaptation based on the Cosine Similarity for Text-Independent Speaker Verification*](https://www.isca-archive.org/odyssey_2010/shum10_odyssey.html) — includes symmetric score normalization (S-norm).
- [Matějka et al. (2017), *Analysis of Score Normalization in Multilingual Speaker Recognition*](https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html)

This is especially relevant to this thesis because clean, VHF, HF and noisy recordings create strong channel mismatch.

#### 10.1A.14 Why use a 2-of-3 decision?

Each scoring branch produces a SAME or DIFFERENT decision.

Example:

```text
calibrated cosine → SAME
PLDA              → SAME
cohort normalized → DIFFERENT

2 SAME votes
→ final SAME
```

Another example:

```text
calibrated cosine → SAME
PLDA              → DIFFERENT
cohort normalized → DIFFERENT

1 SAME vote
→ final DIFFERENT
```

The current rule is therefore:

```python
same_votes >= 2
```

The purpose is not to claim that the three branches are fully independent. The purpose is to avoid allowing one raw similarity threshold to control the complete identity decision.

#### 10.1A.15 Simple explanation of why the project moved to the 3-way backend

The shortest explanation is:

> The original ECAPA approach produced useful speaker embeddings, but the final decision depended too much on manually selected raw-cosine thresholds. Clean and noisy/VHF references produced different score ranges, margins could look strong even when the absolute match was weak, and a generated anchor could itself be imperfect. Some audibly drifting speakers therefore still passed. The current backend keeps ECAPA but evaluates its embeddings using calibrated cosine, PLDA and cohort-normalized cosine, then requires at least two of the three methods to agree on SAME speaker.

#### 10.1A.16 Important limitations that still remain

The 3-way backend improves the decision, but it does not solve every problem.

**1. All three branches still share ECAPA embeddings**

```text
same ECAPA representation
        ↓
three different scoring views
```

It is not equivalent to three completely independent speaker-recognition models.

**2. The current development data can be recording-limited**

If the backend treats every reference WAV as one speaker and obtains same-speaker examples by splitting one recording into chunks, then many SAME examples may be:

```text
different chunks
from the same recording
```

instead of the scientifically stronger situation:

```text
same human speaker
different sessions
different microphones
different channel conditions
different recordings
```

This is an important limitation to state in the thesis.

**3. Calibration needs held-out evaluation**

Where possible:

```text
training/development trials
≠
validation/evaluation trials
```

Calibration, PLDA training and final evaluation should not all reuse exactly the same pairs because this can make results look too optimistic.

**4. A speaker PASS does not mean the words are correct**

Speaker verification cannot detect:

```text
wrong MMSI
wrong position
wrong vessel name
missing sentence
repeated sentence
missing "Over"
duplicated "Over"
hallucinated speech
```

Those require ASR/content comparison.

**5. A speaker PASS does not mean the waveform is good**

The intended speaker can still produce:

```text
clipping
excessive static
long noise-only sections
silence
truncation
runaway duration
```

Those require signal/noise QC.

#### 10.1A.17 Complete progression to remember

```text
STEP 1 — Generate turns separately
↓
Problem:
same role can drift to another-sounding voice

STEP 2 — ECAPA embedding + raw cosine
↓
Checks:
intended reference
other-role reference
same-role anchor
↓
Problems:
manual thresholds
clean/noisy score mismatch
margin not sufficient
anchor can be imperfect
false speaker passes observed

STEP 3 — speaker_backend_3way.py
↓
shared ECAPA embedding
+
calibrated cosine
+
PLDA
+
cohort-normalized cosine
↓
2-of-3 SAME / DIFFERENT decision

STEP 4 — Keep independent QC layers
↓
speaker identity
signal/clipping
ASR/content
noise/channel
manual listening
```

#### 10.1A.18 Local implementation links

Keep the implementation files linked from the README so every methodological claim can be traced to code:

- [Speaker-role classification — `workspace/docs/tools/add_speaker_labels.py`](workspace/docs/tools/add_speaker_labels.py)
- [Generation-plan creation — `workspace/docs/tools/create_pilot_generation_plan.py`](workspace/docs/tools/create_pilot_generation_plan.py)
- [3-way speaker backend — `workspace/docs/tools/speaker_backend_3way.py`](workspace/docs/tools/speaker_backend_3way.py)
- [Current canonical generator — `workspace/docs/tools/run_pilot_generation_dialogue.py`](workspace/docs/tools/run_pilot_generation_dialogue.py)
- Historical/legacy filename used in older snapshots: `run_generation_dialogue.py` (not present in the latest Windows tools tree)

The canonical filename is now `run_pilot_generation_dialogue.py`; new commands should use only that name.

---
<a id="10-2-correct-scientific-interpretation"></a>
### 10.2 Correct scientific interpretation

The backend should be described as:

> A more robust speaker-identity QC method than the previous raw-cosine threshold approach, using three scoring views over a shared ECAPA speaker representation and a 2-of-3 decision.

It should not be described as:

```text
perfect speaker identification
```

or:

```text
three fully independent speaker-recognition systems
```

<a id="10-3-recommended-future-improvements"></a>
### 10.3 Recommended future improvements

```text
multiple independent recordings per physical speaker
held-out speaker/session validation
more diverse development cohorts
explicit channel-robust speaker embeddings
separate clean and noisy evaluation subsets
ROC / DET / EER / minDCF / C_llr reporting
backend disagreement analysis
manual listening validation
```

Useful references:

- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
- [PLDA](https://doi.org/10.1007/11744085_41)
- [BOSARIS calibration/evaluation](https://arxiv.org/abs/1304.2865)
- [Cohort-normalized scores](https://www.isca-archive.org/icslp_1992/rosenberg92_icslp.html)
- [S-norm](https://www.isca-archive.org/odyssey_2010/shum10_odyssey.html)
- [Score normalization under mismatch](https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html)

---

<a id="10-4-remaining-limitations-of-the-3-way-speaker-backend"></a>
### 10.4 Remaining limitations of the 3-way speaker backend

The current backend is stronger than raw cosine thresholding, but it still has important limitations:

1. **Shared embedding model:** calibrated cosine, PLDA, and cohort normalization all start from ECAPA embeddings.
2. **Reference-WAV identity assumption:** if one reference WAV contains more than one physical speaker, no scoring backend can make that reference scientifically clean.
3. **Same-recording chunks:** using multiple chunks from one WAV as same-speaker development examples is weaker than using multiple sessions/recordings of the same human speaker.
4. **Calibration validation:** calibration and PLDA should be evaluated on held-out trials/speakers where possible; training and evaluation on the same trials can give optimistic results.
5. **Channel mismatch:** strong VHF/HF noise, narrow bandwidth, compression, or interference can still influence the ECAPA embedding.
6. **Speaker QC is not content QC:** a turn can have the correct voice but say the wrong MMSI, position, vessel name, or number of `Over` tokens.
7. **Speaker QC is not signal QC:** the correct speaker can still be clipped, noisy, truncated, or excessively silent.

<a id="10-5-general-qc-architecture"></a>
### 10.5 General QC architecture

The current recommendation is to treat quality control as four independent gates.

| GateMain questionExample checks |                                                                 |                                                                                   |
| ------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Speaker identity                | Is this the intended physical speaker?                          | 3-way ECAPA backend, intended-vs-other comparison, within-role consistency        |
| Signal quality                  | Is the waveform technically usable?                             | clipping, peak, RMS, duration, silence, DC offset                                 |
| Content fidelity                | Did the model say the requested message?                        | ASR, WER/CER, vessel name, MMSI, position, missing/repeated `Over`                |
| Noise/channel quality           | Is speech still intelligible under the generated channel/noise? | SNR estimate, speech/background ratio, excessive static, long noise-only sections |

A turn should be accepted only when all required gates pass.

<a id="10-6-reference-qc-before-tts-generation"></a>
### 10.6 Reference QC before TTS generation

Reference quality is now treated as a first-class input requirement because recent controlled tests showed that clean references are generally more stable, while some noisy references fail across both TTS models.
Before a reference is allowed into the generation pool, check:

```
one physical speaker only
no overlapping speech
no announcer/commentary mixed with the target speaker
sufficient continuous speech
reasonable speech-to-noise ratio
a stable channel condition
no large noise-only region
exact transcript matches the trimmed reference WAV
speaker embedding reasonably stable across chunks
```

If a raw recording contains multiple people or commentary, trim a single-speaker segment and update the transcript to match only the trimmed WAV.
For the noisy-reference experiment, do **not** silently replace every noisy reference with a clean reference because that would remove the experimental variable. A useful separation is:

```
raw noisy reference
→ TTS conditioning for noisy-reference experiment

optional cleaned/voice-isolated copy
→ analysis / speaker-verification support only
```

<a id="10-7-current-general-failure-meanings-and-solutions"></a>
### 10.7 Current general failure meanings and solutions

| Failure / symptomMeaningGeneral action        |                                                                |                                                                                       |
| --------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `EXCESSIVE_CLIPPING`                          | waveform spends too much time at/near full scale               | peak-normalize before QC; do not simply loosen the threshold                          |
| `INTENDED_REFERENCE_NOT_SAME_3WAY`            | generated voice does not reliably match the intended reference | try another deterministic seed; if repeated, trim/reject the reference                |
| obvious same-role voice change                | speaker drift                                                  | retry turn; verify reference is single-speaker; keep stable role style/seed           |
| excessive Fish-Speech radio/static noise      | model is strongly copying channel/noise characteristics        | reject turn; improve/trim reference; add a separate noise-quality gate                |
| missing sentence / wrong name / wrong number  | content failure                                                | reject and regenerate using ASR/text QC                                               |
| missing or duplicated `Over`                  | content or source-text mismatch                                | first check the CSV target text; regenerate only if source contains the correct count |
| repeated sentence / extra speech after `Over` | TTS repetition/hallucination                                   | reject turn and regenerate with a different seed/lower randomness                     |
| very short output                             | possible truncation                                            | increase token allowance if needed; reject/regenerate                                 |
| very long output with little speech           | noise, repetition, or long pause failure                       | duration/noise gate; reject/regenerate                                                |
| reference contains >1 speaker                 | invalid speaker-conditioning source                            | trim or reject before TTS                                                             |
| reference transcript does not match WAV       | Fish-Speech conditioning mismatch                              | correct transcript or trim WAV and transcript together                                |

<a id="10-8-retry-policy"></a>
### 10.8 Retry policy

Retries should be bounded. A practical diagnostic policy is:

```
Attempt 1
  ↓ fail
Attempt 2 with next seed
  ↓ fail
Attempt 3
  ↓ fail
Attempt 4
  ↓ fail
reject the turn/reference and report the repeated failure reason
```

Repeated failure across all attempts is useful information:

```
4 × speaker mismatch
→ reference / speaker-conditioning problem

4 × excessive noise
→ noisy-reference / model interaction problem

4 × content mismatch
→ text-generation problem

4 × clipping
→ signal-processing problem
```

The goal is not to generate indefinitely until something passes.

<a id="10-9-current-experimental-interpretation"></a>
### 10.9 Current experimental interpretation

Recent controlled pilot results show a consistent pattern:

```
clean references
→ generally stable for both Chatterbox and Fish-Speech

noisy references
→ much more variable
→ some references fail across both models
```

This indicates that **reference quality is a major independent variable**, not merely a model-specific weakness.
At the same time, the models fail differently:

```
Chatterbox
→ usually clearer perceptually
→ can fail speaker-match checks under difficult noisy references
→ can produce clipping unless output level is controlled

Fish-Speech
→ can preserve speaker identity in some noisy cases
→ can copy too much radio/static/channel noise
→ therefore needs explicit noise and content QC in addition to speaker QC
```

A speaker-verification PASS must therefore never be interpreted as a complete audio-quality PASS.

---

---

<a id="11-planned-evaluation-metrics"></a>
## 11. Planned Evaluation Metrics

```

WER

CER

keyword accuracy

number accuracy

speaker-role classification accuracy

speaker-role precision / recall / F1

automatic speaker-label coverage

REVIEW_REQUIRED rate

SNR

high-frequency energy ratio (\`R_high\`)

STOI

PESQ

subjective listening score

VHF realism score

clipping ratio

silence ratio

```

Speaker-role metrics should be calculated against manually annotated ground truth rather than treating rule scores as probabilities.

<a id="11-1-speaker-verification-metrics-to-report-separately"></a>
### 11.1 Speaker-verification metrics to report separately

For the final speaker-verification study, report both **discrimination** and **calibration** behavior where data size allows. Useful measures include:

```
same-speaker / different-speaker trial counts
ROC / DET curves
EER
minDCF or task-specific detection cost
C_llr for calibrated likelihood-ratio quality
false accept rate
false reject rate
2-of-3 backend agreement/disagreement rate
failure rate by clean vs noisy reference condition
failure rate by reference recording
```

Do not present raw cosine similarity as a probability.

---

---

<a id="13-design-principle"></a>
## 13. Design Principle

The active dialogue pipeline separates three responsibilities:

```

add_speaker_labels.py

→ identifies the communication role from text

create_pilot_generation_plan.py

→ prepares model/reference/output configuration

run_pilot_generation_dialogue.py

→ synthesizes and concatenates the audio

```

Fish-Speech and Chatterbox are **not responsible for deciding whether a turn belongs to the Vessel or Coast Guard**. The TTS stage receives the already resolved role label and uses it only to select the correct reference voice.

<a id="13-1-four-different-questions-must-stay-separate"></a>
### 13.1 Four different questions must stay separate

The current project intentionally separates four questions that can otherwise be confused:

```
1. ROLE:
   Is this text Vessel or Coast Guard?
   → add_speaker_labels.py

2. GENERATION CONFIGURATION:
   Which model, reference WAV, reference transcript, seed, and output path should be used?
   → create_pilot_generation_plan.py

3. SYNTHESIS:
   Generate one turn at a time and concatenate accepted turns.
   → run_pilot_generation_dialogue.py

4. QUALITY / IDENTITY:
   Is the generated audio actually usable and still the intended speaker?
   → signal QC + ASR/content QC + speaker_backend_3way.py + manual listening
```

A failure in one layer should not be hidden by another layer. For example:

```
correct role label
≠ correct speaker identity

correct speaker identity
≠ correct spoken content

correct spoken content
≠ acceptable signal/noise quality
```

This separation is the main design principle for the current thesis pipeli

<a id="14-completed-controlled-pilot-15-dialogues-4-conditions"></a>
## 14. Completed Controlled Pilot — 15 Dialogues × 4 Conditions

A controlled pilot used **15 source dialogues**, each expanded into the four experimental conditions:

```text
15 dialogues × 4 conditions = 60 generation rows
```

Final outcome:

| Condition | Passed | Total | Final pass rate |
|---|---:|---:|---:|
| Chatterbox + clean reference | 15 | 15 | 100% |
| Fish-Speech + clean reference | 15 | 15 | 100% |
| Chatterbox + noisy reference | 9 | 15 | 60.0% |
| Fish-Speech + noisy reference | 11 | 15 | 73.3% |
| **Overall** | **50** | **60** | **83.3%** |

Important interpretation:

```text
all 30 clean-reference runs passed
all 10 dialogue failures occurred in noisy-reference conditions
20 / 30 noisy-reference runs still passed
```

Therefore the finding is **not** “noise always breaks TTS.” The stronger pilot observation is that performance becomes more reference-dependent under noisy/native-VHF conditioning.

Fish-Speech showed greater final-pass robustness than Chatterbox under the noisy-reference condition in this small pilot (`11/15` vs `9/15`), but this should not be generalized to global model superiority because the sample is small and generation stability/retry behavior is a separate question.

<a id="14-1-five-unique-problematic-vessel-references"></a>
### 14.1 Five unique problematic Vessel references

The ten failed dialogue rows reduce to five unique Vessel/source references:

| Reference | Failed dialogue rows observed in the pilot | Main pattern |
|---|---:|---|
| Yacht mayday — “We are sinking” | 4 | Failed across both models and two dialogue texts; strongest reference-specific signal |
| Saginaw Bay capsized vessel | 2 | Failed in both models; later diagnostics showed chunk sensitivity |
| WhatsApp_audio_1 | 2 | Failed in both models repeatedly |
| Boaters mayday | 1 | Chatterbox failed while Fish-Speech passed: model × reference interaction |
| Iran tanker / Hormuz | 1 | Chatterbox failed while Fish-Speech passed; reference A/B control itself was strong |

The pilot therefore suggests that **specific reference quality and model-reference interaction** are more informative than a simple clean/noisy label alone.

---

<a id="15-failed-sample-diagnostic-study"></a>
## 15. Failed-Sample Diagnostic Study

The generalized diagnostic is implemented by:

- [workspace/docs/tools/diagnose_failed_samples.py](workspace/docs/tools/diagnose_failed_samples.py)

Inputs:

```text
pilot_generation_plan.csv
failed per-turn generation/QC records
five problematic original Vessel references
workspace/models/speaker_backend_3way.pkl
```

Outputs:

- [workspace/data/diagnostics/failed_reference_speaker_tests/reference_test1_summary.csv](workspace/data/diagnostics/failed_reference_speaker_tests/reference_test1_summary.csv)
- [workspace/data/diagnostics/failed_reference_speaker_tests/failed_attempt_tests_2_3.csv](workspace/data/diagnostics/failed_reference_speaker_tests/failed_attempt_tests_2_3.csv)
- [A/B diagnostic reference chunks](workspace/data/diagnostics/failed_reference_speaker_tests/references/)

<a id="15-1-test-1-can-the-backend-recognize-two-real-chunks-from-the-same-original-recording"></a>
### 15.1 Test 1 — can the backend recognize two real chunks from the same original recording?

For each problematic original reference:

```text
original reference
      ↓
split/select non-overlapping speech chunks A and B
      ↓
ECAPA embeddings
      ↓
3-way backend
      ↓
expected: SAME
```

The user manually listened to all five A/B pairs and confirmed that A and B contain the **same physical human speaker** for:

```text
Yacht
Boaters
Saginaw Bay
WhatsApp_audio_1
Hormuz
```

The 3-way final decision also classified all five A/B controls as SAME. This establishes an important local sanity control before interpreting the failed generated outputs.

<a id="15-2-tests-2-3-compare-failed-generated-attempts-against-a-b-full-reference"></a>
### 15.2 Tests 2/3 — compare failed generated attempts against A/B/full reference

Across the failed dialogues, the diagnostic found **45 failed generated attempt WAVs**.

Summary:

```text
40 / 45  → GENERATED_SPEAKER_MISMATCH_LIKELY
 5 / 45  → BORDERLINE_OR_CHUNK_DEPENDENT
 0 / 45  → clear generated match to both A and B
```

Signal-QC cross-check:

```text
44 / 45 failed generated attempts still had signal_pass = True
1 / 45 also had a signal failure
```

This is important because it separates two questions:

```text
waveform technically usable?
        ≠
intended physical speaker preserved?
```

Reference-level breakdown:

| Reference | Failed attempts | Clear mismatch | Borderline/chunk-dependent |
|---|---:|---:|---:|
| Yacht | 17 | 17 | 0 |
| Saginaw Bay | 12 | 8 | 4 |
| WhatsApp_audio_1 | 8 | 8 | 0 |
| Boaters | 4 | 4 | 0 |
| Hormuz | 4 | 3 | 1 |
| **Total** | **45** | **40** | **5** |

<a id="15-3-important-detailed-observations"></a>
### 15.3 Important detailed observations

- **Yacht:** known-SAME A/B control succeeds by cosine + cohort while PLDA rejects; generated attempts are consistently poor across both models.
- **Boaters:** real A/B control is SAME, yet Chatterbox failed while Fish-Speech passed in the pilot, suggesting a model-reference interaction rather than a universally bad reference.
- **Saginaw Bay:** generated decisions can change depending on whether chunk A, chunk B or the full reference is used. This is direct evidence of verifier/channel/chunk sensitivity.
- **WhatsApp_audio_1:** A/B control is SAME but repeated generated attempts from both models fail speaker identity.
- **Hormuz:** strongest real A/B control; all three backend branches can agree SAME on the reference itself, while Chatterbox generated attempts still mostly fail and Fish-Speech passed in the pilot. This strongly suggests a TTS/model-reference interaction for that case.
- Across the 45 failed generated comparisons, **PLDA was extremely strict** and rejected every generated A/B/full-reference comparison, which motivates separate PLDA analysis rather than relaxing the ensemble rule just to obtain passes.

---

<a id="16-real-human-speaker-backend-validation"></a>
## 16. Real-Human Speaker-Backend Validation

The next sanity check asks whether the 3-way backend can distinguish real SAME and DIFFERENT controls independently of TTS generation.

Implementation:

- [workspace/docs/tools/validate_speaker_backend_real_trials.py](workspace/docs/tools/validate_speaker_backend_real_trials.py)

Input controls:

```text
5 known-SAME trials
→ A vs B for each manually verified problematic reference

10 known-DIFFERENT trials
→ each Vessel A chunk vs two Coast Guard references
```

Total:

```text
15 real-audio trials
```

Results from [method_summary.csv](workspace/data/diagnostics/backend_real_validation/method_summary.csv):

| Method | SAME correct | SAME total | SAME accuracy | DIFFERENT correct | DIFFERENT total | DIFFERENT accuracy | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Calibrated cosine | 5 | 5 | 100% | 9 | 10 | 90% | 14/15 = 93.3% |
| PLDA | 1 | 5 | 20% | 10 | 10 | 100% | 11/15 = 73.3% |
| Cohort-normalized cosine | 5 | 5 | 100% | 10 | 10 | 100% | 15/15 |
| **Final 2-of-3** | **5** | **5** | **100%** | **10** | **10** | **100%** | **15/15** |

<a id="16-1-what-this-means"></a>
### 16.1 What this means

The final 2-of-3 backend is **not obviously broken** on this small real-audio control set. The ensemble corrects individual method errors:

```text
Known SAME example:
cosine → SAME
PLDA   → DIFFERENT
cohort → SAME
2 votes SAME → final SAME
```

```text
Known DIFFERENT example:
cosine → SAME   # one false accept
PLDA   → DIFFERENT
cohort → DIFFERENT
2 votes DIFFERENT → final DIFFERENT
```

Therefore the current evidence does **not** justify lowering the final `same_votes >= 2` requirement merely to make failed TTS outputs pass.

<a id="16-2-scientific-limitation"></a>
### 16.2 Scientific limitation

This is a **diagnostic sanity check, not a publication-quality independent benchmark**:

```text
only 5 SAME + 10 DIFFERENT trials
same-speaker pairs come from different chunks of the same recording
independent sessions/microphones are not available for these speakers
possible development/backend data overlap must be considered
```

Thesis-safe wording:

> In this diagnostic control set, the cohort-normalized branch and final 2-of-3 decision correctly classified all 15 real-audio trials. The sample is small and same-speaker controls are same-recording chunks, so these results are treated as a sanity check rather than a general speaker-verification accuracy claim.

<a id="16-3-what-is-actually-compared"></a>
### 16.3 What is actually compared

The scoring system compares **numerical ECAPA speaker embeddings**, not speaker IDs:

```text
WAV 1 → ECAPA → embedding 1 ┐
                            ├→ backend.score_pair(...)
WAV 2 → ECAPA → embedding 2 ┘
                            ↓
                 cosine / PLDA / cohort
                            ↓
                       2-of-3 vote
```

`speaker_id` values are metadata used to exclude the actual target speakers from the cohort-normalization impostor pool. They do **not** tell the backend whether a pair is SAME or DIFFERENT. Known ground truth is applied only after scoring to calculate correctness.

---

<a id="17-current-repository-storage-and-cleanup-policy"></a>
## 17. Current Repository Storage and Cleanup Policy

<a id="17-1-keep-as-active-current"></a>
### 17.1 Keep as active/current

```text
workspace/docs/metadata/maritime_distress_formatted (version 1).xlsx
workspace/data/pilot/metadata/maritime_distress_normalized.xlsx
workspace/data/pilot/metadata/metadata/maritime_distress_speaker_labeled.xlsx
workspace/data/pilot/metadata/metadata/dialogue_text_review.csv
workspace/docs/sorted_pipeline_data/classification_summary.csv
workspace/data/diagnostics/**
workspace/docs/tools/**
```

The nested `metadata/metadata/` directory is functional but should eventually be flattened to one `metadata/` directory after updating all script paths.

<a id="17-2-generation-plan-versioning"></a>
### 17.2 Generation-plan versioning

Do not overwrite every historical plan with the generic name `pilot_generation_plan.csv`. Preserve experiment identity.

Important known plans/stages:

```text
60-row controlled pilot
= 15 dialogues × 4 conditions
→ must be retained because current pilot results depend on it

older 400-row plan
= 100 dialogues × 4 conditions
→ scenario_type was blank in the uploaded copy
→ legacy, not the current five-scenario full plan

current full normalized corpus
= 500 dialogues × 4 conditions
→ 2000 rows if all 100 dialogues per scenario are selected
```

Recommended archive naming:

```text
pilot_generation_plan_15texts_60runs.csv
pilot_generation_plan_legacy_100texts_400runs.csv
pilot_generation_plan_5scenarios_2000runs.csv
```

<a id="17-3-earlier-single-output-evaluation-workbooks"></a>
### 17.3 Earlier single-output evaluation workbooks

The earlier metadata/evaluation workbooks remain useful as history but are not the active integrated dialogue pipeline:

```text
tts_dataset_metadata_step4_filled.xlsx
→ complete earlier evaluation-stage workbook

tts_dataset_metadata_with_metrics.xlsx
→ earlier finalized WER/CER/keyword/number evaluation workbook
```

Recommended location:

```text
workspace/data/archive/legacy_single_output_evaluation/
```

An older incomplete `tts_dataset_metadata_step4_filled (version 1).xlsx` should not be retained alongside the complete version.

<a id="17-4-earlier-reference-partition-threshold-experiments"></a>
### 17.4 Earlier reference-partition threshold experiments

`noisy_threshold_test_results.csv` documents the older SNR/noise-dBFS threshold experiment and should be labelled **legacy methodology**, not current classification.

Current active partitioning is:

```text
R_high >= 0.005
AND
SNR_dynamic >= 22 dB
→ clean

otherwise
→ noisy/native-VHF
```

The currently visible `noisy_reference_inventory.csv.xlsx` is not the canonical active partition result; the active result is `sorted_pipeline_data/classification_summary.csv`. If the workbook is retained, label it as historical exploratory material.

<a id="17-5-what-must-stay-data-station-only"></a>
### 17.5 What must stay data-station-only

Do not put the following into Git:

```text
/data/kavya/conda_envs/chatterbox/bin/
/data/kavya/conda_envs/chatterbox/lib/
/data/kavya/conda_envs/chatterbox/include/
/data/kavya/conda_envs/chatterbox/share/
/data/kavya/conda_envs/chatterbox/conda-meta/
Python package caches
__pycache__/
full Hugging Face cache
all generated attempt WAVs unless specifically selected as evidence
large temporary files
```

The ECAPA cache under `workspace/models/spkrec-ecapa-voxceleb/` is runtime/model-cache material. The important reproducibility information is the model identity (`speechbrain/spkrec-ecapa-voxceleb`) and the project backend model used on top of it.

---

<a id="18-next-experiment-channel-mismatch-diagnostic"></a>
## 18. Next Experiment — Channel-Mismatch Diagnostic

Current evidence supports two competing explanations for many failed generated-speaker comparisons:

```text
A. real TTS speaker-conditioning failure
or
B. verifier sensitivity to reference/output channel mismatch
```

Typical mismatch:

```text
REFERENCE
→ real VHF/HF/radio bandwidth
→ static/compression/interference
→ narrowband spectrum

GENERATED TTS
→ cleaner waveform
→ different bandwidth/spectral balance
→ different noise floor
```

Next diagnostic design:

```text
failed generated WAV
        ↓
create a diagnostic channel-matched copy
        ↓
make generated channel more comparable to its reference
        ↓
rerun ECAPA + 3-way backend
        ↓
compare BEFORE vs AFTER
```

Interpretation:

```text
BEFORE DIFFERENT → AFTER SAME
→ channel/verifier mismatch contributed substantially

BEFORE DIFFERENT → AFTER DIFFERENT
→ channel mismatch alone does not explain failure;
  true TTS speaker-conditioning/reference interaction becomes more likely

mixed results
→ reference/chunk/model-specific sensitivity
```

Do **not** modify the original generated WAVs and do **not** lower the 2-of-3 decision rule before this diagnostic.

---

<a id="19-current-thesis-interpretation-and-real-life-use"></a>
## 19. Current Thesis Interpretation and Real-Life Use

The most accurate current framing is:

> The primary application of the proposed system is the automated generation and multi-dimensional quality evaluation of multi-speaker maritime VHF radio dialogues using TTS. Real publicly available maritime VHF recordings, together with cleaner demonstration recordings, are used as reference audio. The system investigates whether TTS models can generate realistic, domain-specific maritime VHF communication while preserving the intended speaker identity, message content, and relevant radio-channel characteristics under clean and noisy reference conditions.

The main technical contribution is therefore not simply “generate speech.” It is:

```text
TTS generation
+
systematic automatic quality control
+
traceable failure analysis
```

ASR belongs to the **content-fidelity QC layer**:

```text
target maritime text
        ↓
TTS-generated speech
        ↓
ASR transcription
        ↓
compare with target
        ↓
WER / CER / keyword / number correctness
```

This can later support downstream maritime speech-system research, data augmentation or simulation, but those downstream systems are not themselves the core implementation of this thesis.

---



<a id="20-references-literature-standards-and-citation-map"></a>
## 20. References, Literature, Standards, and Citation Map


<a id="local-literature-and-standards-stored-in-this-repository"></a>
### Local literature and standards stored in this repository

The following source files are retained locally so methodology statements can be traced to the material used during development:

| Local file | Role in the project |
|---|---|
| [lit/1904.02882v1.pdf](lit/1904.02882v1.pdf) | Local copy of the LibriTTS paper used as general TTS/dataset background |
| [lit/T-REC-P.80-199303-S__PDF-E.pdf](lit/T-REC-P.80-199303-S__PDF-E.pdf) | Historical ITU subjective speech-quality recommendation retained as background; current README cites ITU-T P.800 for modern subjective evaluation framing |
| [lit/VAD-review.pdf](lit/VAD-review.pdf) | Local VAD review/background material |
| [workspace/docs/snr_outputs/Noise_estimation_techniques_for_robust_speech_recognition.pdf](workspace/docs/snr_outputs/Noise_estimation_techniques_for_robust_speech_recognition.pdf) | Noise-estimation background |
| [workspace/docs/snr_outputs/Noise_power_spectral_density_estimation_based_on_optimal_smoothing_and_minimum_statistics.pdf](workspace/docs/snr_outputs/Noise_power_spectral_density_estimation_based_on_optimal_smoothing_and_minimum_statistics.pdf) | Minimum-statistics noise-PSD background |
| [workspace/docs/snr_outputs/On_Reference_Signal_Estimation_from_Noisy_Speech_Using_Deep_Learning_for_Intelligibility_Estimation.pdf](workspace/docs/snr_outputs/On_Reference_Signal_Estimation_from_Noisy_Speech_Using_Deep_Learning_for_Intelligibility_Estimation.pdf) | Noisy/reference-signal estimation background |
| [workspace/docs/snr_outputs/Peeters_2003_cuidadoaudiofeatures.pdf](workspace/docs/snr_outputs/Peeters_2003_cuidadoaudiofeatures.pdf) | Spectral/audio-feature background |
| [workspace/docs/snr_outputs/R-REC-M.493-15-201901-S!!PDF-E.pdf](workspace/docs/snr_outputs/R-REC-M.493-15-201901-S!!PDF-E.pdf) | ITU-R maritime-mobile/DSC communications context |
| [workspace/docs/snr_outputs/T-REC-G.712-199611-S!!PDF-E.pdf](workspace/docs/snr_outputs/T-REC-G.712-199611-S!!PDF-E.pdf) | Telephony/transmission-performance and bandwidth context |
| [workspace/docs/snr_outputs/reference_audio_partitioning_pipeline.md](workspace/docs/snr_outputs/reference_audio_partitioning_pipeline.md) | Project-specific documentation of the current reference partitioning rule |

`Thesis_proposal.docx` and `report.docx` under [lit/](lit/) are project documents rather than external research papers and are therefore not treated as bibliographic sources.


<a id="maritime-communication"></a>
### Maritime communication

- [IMO Standard Marine Communication Phrases (SMCP)](https://www.imo.org/en/ourwork/safety/pages/standardmarinecommunicationphrases.aspx) — standardized maritime safety-language reference used to motivate communication-role cues.

<a id="speech-quality"></a>
### Speech quality

- [ITU-T P.800 — Methods for subjective determination of transmission quality](https://www.itu.int/rec/t-rec-p.800) — subjective speech/transmission-quality evaluation reference. The older P.80 recommendation was revised/renumbered as P.800.

<a id="tts-models"></a>
### TTS models

- [Fish-Speech official repository](https://github.com/fishaudio/fish-speech)

- [Fish-Speech paper — ](https://arxiv.org/abs/2411.01156)[*\*Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis\**](https://arxiv.org/abs/2411.01156)

- [Chatterbox official repository — Resemble AI](https://github.com/resemble-ai/chatterbox)

<a id="dataset-background"></a>
### Dataset background

- [LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech](https://arxiv.org/abs/1904.02882)

- Local copy: [lit/1904.02882v1.pdf](lit/1904.02882v1.pdf)

<a id="speaker-verification-calibration-plda-and-score-normalization"></a>
### Speaker verification, calibration, PLDA, and score normalization

- [Desplanques, B., Thienpondt, J. & Demuynck, K. (2020), ](https://arxiv.org/abs/2005.07143)[*ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification*](https://arxiv.org/abs/2005.07143) — primary architecture reference for ECAPA-TDNN speaker embeddings.

- [Ioffe, S. (2006), ](https://doi.org/10.1007/11744085_41)[*Probabilistic Linear Discriminant Analysis*](https://doi.org/10.1007/11744085_41) — foundational PLDA reference; PLDA models within-class and between-class variation and supports same/different-class inference.

- [Brümmer, N. & de Villiers, E. (2013), ](https://arxiv.org/abs/1304.2865)[*The BOSARIS Toolkit: Theory, Algorithms and Code for Surviving the New DCF*](https://arxiv.org/abs/1304.2865) — speaker-verification score calibration/evaluation background, including logistic-regression-based calibration and detection-cost analysis.

- [Rosenberg, A. E., DeLong, J., Lee, C.-H., Juang, B.-H. & Soong, F. K. (1992), ](https://doi.org/10.21437/ICSLP.1992-176)[*The use of cohort normalized scores for speaker verification*](https://doi.org/10.21437/ICSLP.1992-176) — classic cohort-normalization reference.

- [Shum, S., Dehak, N., Dehak, R. & Glass, J. (2010), ](https://www.isca-archive.org/odyssey_2010/shum10_odyssey.html)[*Unsupervised Speaker Adaptation based on the Cosine Similarity for Text-Independent Speaker Verification*](https://www.isca-archive.org/odyssey_2010/shum10_odyssey.html) — includes symmetric score normalization (S-norm) for speaker verification.

- [Matějka, P., Novotný, O., Plchot, O., Burget, L., Diez Sánchez, M. & Černocký, J. (2017), ](https://doi.org/10.21437/Interspeech.2017-803)[*Analysis of Score Normalization in Multilingual Speaker Recognition*](https://doi.org/10.21437/Interspeech.2017-803) — modern score-normalization analysis under language/channel mismatch and useful background for cohort selection.

- Recommended local implementation link: [workspace/docs/tools/speaker\_backend\_3way.py](workspace/docs/tools/speaker_backend_3way.py)


<a id="speaker-qc-reference-map-what-each-source-supports"></a>
### Speaker-QC Reference Map — What Each Source Supports

The following references are the main sources for the speaker-verification explanation above.

| Topic used in this project | Reference | Why it is relevant |
|---|---|---|
| ECAPA speaker embeddings | [Desplanques et al. (2020), ECAPA-TDNN](https://arxiv.org/abs/2005.07143) | Architecture used to obtain speaker-discriminative embeddings. |
| PLDA | [Ioffe (2006), Probabilistic Linear Discriminant Analysis](https://doi.org/10.1007/11744085_41) | Background for modelling within-class and between-class variation. |
| Score calibration and detection evaluation | [Brümmer & de Villiers (2013), BOSARIS Toolkit](https://arxiv.org/abs/1304.2865) | Logistic score calibration, DCF/minDCF, EER and calibration analysis. |
| Cohort normalization | [Rosenberg et al. (1992), cohort normalized scores](https://www.isca-archive.org/icslp_1992/rosenberg92_icslp.html) | Classic cohort-normalized speaker-verification scoring. |
| S-norm | [Shum et al. (2010)](https://www.isca-archive.org/odyssey_2010/shum10_odyssey.html) | Symmetric score normalization for speaker verification. |
| Score normalization under mismatch | [Matějka et al. (2017)](https://www.isca-archive.org/interspeech_2017/matejka17_interspeech.html) | Shows the importance of normalization under language/channel mismatch. |
| Maritime communication language | [IMO Standard Marine Communication Phrases](https://www.imo.org/en/ourwork/safety/pages/standardmarinecommunicationphrases.aspx) | Motivation for structured Vessel/Coast-Guard communication-role cues. |
| Subjective speech quality | [ITU-T P.800](https://www.itu.int/rec/t-rec-p.800) | Standard subjective speech/transmission-quality methodology. |
| Fish-Speech implementation | [Fish-Speech official GitHub repository](https://github.com/fishaudio/fish-speech) | Official implementation/documentation for Fish-Speech. |
| Chatterbox implementation | [Resemble AI Chatterbox official GitHub repository](https://github.com/resemble-ai/chatterbox) | Official implementation/documentation for Chatterbox. |

These references support the **methodology**. Project-specific thresholds, retry counts, reference selections and failure observations remain experimental choices that must be reported as such.

---

<a id="reference-audio-partitioning-and-noise-estimation"></a>
### Reference-audio partitioning and noise estimation

- [Martin, R. (2001), ](https://doi.org/10.1109/89.928915)[*\*Noise power spectral density estimation based on optimal smoothing and minimum statistics\**](https://doi.org/10.1109/89.928915) — background for minimum-statistics-style noise-floor estimation.

- [Hirsch, H.-G. & Ehrlicher, C. (1995), ](https://doi.org/10.1109/ICASSP.1995.479532)[*\*Noise estimation techniques for robust speech recognition\**](https://doi.org/10.1109/ICASSP.1995.479532) — supporting reference for statistical noise estimation and speech/noise separation.

- [Peeters, G., ](https://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf)[*\*A large set of audio features for sound description (similarity and classification) in the CUIDADO project\**](https://recherche.ircam.fr/anasyn/peeters/ARTICLES/Peeters_2003_cuidadoaudiofeatures.pdf) — supporting background for spectral/audio-feature analysis.

- [Takahashi, H. & Kondo, K. (2018), ](https://doi.org/10.1109/GCCE.2018.8574768)[*\*On Reference Signal Estimation from Noisy Speech Using Deep Learning for Intelligibility Estimation\**](https://doi.org/10.1109/GCCE.2018.8574768) — supporting reference for noisy-speech/reference estimation.

- [ITU-R M.493-15](https://www.itu.int/rec/R-REC-M.493) — maritime mobile-service communications context.

- [ITU-T G.712](https://www.itu.int/rec/T-REC-G.712) — transmission-performance and telephony-channel context used in the bandwidth discussion.

- [Rabiner, L. R. & Schafer, R. W., ](https://books.google.com/books?id=YTa7QgAACAAJ)[*\*Theory and Applications of Digital Speech Processing\**](https://books.google.com/books?id=YTa7QgAACAAJ) — general digital speech-processing background.

- Local methodology: [workspace/docs/snr\_outputs/reference\_audio\_partitioning\_pipeline.md](workspace/docs/snr_outputs/reference_audio_partitioning_pipeline.md)

---

