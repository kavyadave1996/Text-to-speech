# experiments


# Chronological Experiment Overview

**Project:** Maritime Text-to-Speech (TTS) Generation and Quality Assessment  
**Period covered:** April–October 2026  
**Status:** Working historical summary. Execution status is based on available thesis discussions and should be verified against scripts, logs, and experiment records before final publication.

| Period | Experiment or research stage | Purpose | Evidence of execution |
|---|---|---|---|
| April | TTS model investigation | Identify a suitable TTS architecture for maritime distress communication. | Research and planning confirmed. |
| April–May | Baseline TTS and adaptation strategy | Compare zero-shot/few-shot generation against model adaptation. | Some commands discussed; execution not fully verified. |
| May–June | Maritime dialogue corpus preparation | Use existing emergency maritime text for TTS generation. | Corpus selection confirmed. |
| June–July | Generation-condition design | Compare plain text, prosody-formatted text, and reference-conditioned TTS. | Experimental design confirmed. |
| July–August | Real reference-audio preparation | Prepare clean and VHF-like speaker references. | Processing commands recovered. |
| July–August | Fish-Speech prosody-control experiments | Generate speech from prosody-formatted maritime dialogue. | Script execution and troubleshooting confirmed. |
| July–August | VHF simulation and fine-tuning design | Develop a path toward direct VHF-like synthesis. | Selected methodology; training execution unverified. |
| August–September | Speaker reference and backend experiments | Verify speaker identity and distinguish two dialogue roles. | Implementation and diagnostics confirmed. |
| September | Pilot dialogue generation | Test both models under clean/noisy reference conditions. | Execution confirmed. |
| September–October | Turn-level QC development | Identify incorrect speakers, content errors, and audio defects. | Execution and refinement confirmed. |
| October | Final dialogue QC | Validate concatenated maritime conversations. | Execution confirmed. |
| October | Human listening evaluation | Compare automatic acceptance with manual judgment. | Review records confirmed. |
| October | Chatterbox text normalization | Improve pronunciation of maritime abbreviations and identifiers. | Script modification confirmed; results need verification. |

## Experiment 01 — Neural TTS model investigation

*April–May 2026 · Model selection*

**What:** You investigated existing TTS technologies for generating maritime speech, including FastSpeech2, VALL-E-related approaches, Voxtral, MOSS-TTSD, Fish-Speech and Chatterbox. XTTS-v2 and Piper also appeared in baseline discussions.

**Why:** You needed a model capable of producing intelligible maritime distress messages, handling maritime vocabulary and speaker characteristics, and potentially supporting expressive speech or fine-tuning.

**How:** You evaluated model capabilities, adaptation methods, speaker conditioning and the practicality of running inference or fine-tuning. You established that your thesis would adapt existing TTS models rather than develop a new speech-synthesis architecture.

**Outcome:** Fish-Speech and Chatterbox became the two main experimental engines. MOSS-TTSD was also part of the earlier comparison plan, but I cannot verify a completed MOSS generation experiment from the retrieved history.

**GitHub documentation:** `docs/model-selection.md`

## Experiment 02 — Maritime text dataset preparation

*May–August 2026 · Corpus preparation*

**What:** You worked with existing maritime communication text, rather than treating text generation as the primary research task.

**Why:** Your objective was to convert established maritime dialogue content into synthetic speech, preserve critical information and eventually support domain adaptation.

**How:**

- An existing maritime corpus included the `10_reporting_distress_situations` section.
- You worked with formatted text variants including `text`, `clarity_formatted_text` and `natural_prosody_text`.
- An earlier workbook contained 100 rows with those fields but did not initially include speaker metadata.
- Later work introduced speaker-labelled dialogue data and an expanded dialogue corpus, with a 500-dialogue dataset referenced in the September discussions.

The evolution of the datasets matters: the earlier 400-row plan, the expanded 500-dialogue dataset and the 60-row pilot should not be presented as interchangeable.

**GitHub documentation:** `docs/dataset-preparation.md`

## Experiment 03 — Reference-audio preparation

*July–August 2026 · Audio preprocessing*

**What:** You prepared reference audio from maritime VHF recordings and cleaner speech demonstrations, to support voice-conditioned synthesis.

**Why:** Real VHF recordings contain channel noise and filtering effects. You wanted to investigate whether reference-audio condition influenced generated speaker identity, clarity and maritime speech characteristics.

**How:** You used audio preprocessing including mono conversion, resampling, filtering and loudness normalization.

One exact command recovered from the earlier discussions is:

```bash
ffmpeg -y \
  -i raw_vhf_audio/msc_aries_original.wav \
  -ac 1 -ar 24000 \
  -af "highpass=f=120,lowpass=f=6500,loudnorm=I=-18:TP=-2:LRA=7" \
  reference_audio/msc_aries_clean_reference.wav
```

The target naming convention used the term `clean_reference`, although filtering and loudness normalization do not guarantee complete noise removal or reconstruction of the original clean speech.

You also examined VHF-specific signal processing, including band-limited speech, compression, gain adjustments and limiting.

**Outcome:** The clean-reference and noisy-reference conditions became central experimental variables.

**GitHub documentation:** `docs/reference-audio-preparation.md`

## Experiment 04 — Text and prosody conditioning

*June–August 2026 · Speech-control experiment*

**What:** You investigated different ways of controlling TTS using dialogue text, pronunciation formatting, prosody instructions and reference voices.

**Why:** Maritime communication requires understandable call signs, numbers, location information and distress messages. Ordinary TTS may mispronounce these or generate unsuitable speaking styles.

**How:** Your earlier experimental design included three conditions:

| Condition              | Input strategy                                                      | Research purpose                                      |
| ---------------------- | ------------------------------------------------------------------- | ----------------------------------------------------- |
| Baseline               | Plain maritime text                                                 | Measure default model performance                     |
| Text/prompt adaptation | Formatted text, punctuation and possible emotion markers            | Assess controllability without changing model weights |
| Reference conditioning | Formatted text with selected speaker or expressive audio references | Assess speaker and speaking-style adaptation          |

You also attempted to use your prosody-plan workbook to drive Fish-Speech generation.

A confirmed command from this stage was:

```bash
python tools/generate_fish_from_prosody_plan.py
```

The execution produced a missing-reference-text error involving a path under `data/reference_voices`. You subsequently investigated why the synthesized speech appeared to follow the reference transcript instead of the desired workbook dialogue.

This is a valuable documented experiment because it exposes the distinction between reference text used for conditioning and target text used for synthesis.

**GitHub documentation:** `docs/prosody-conditioning.md`

## Experiment 05 — VHF simulation and direct VHF speech adaptation

*July–August 2026 · Domain adaptation design*

**What:** You selected an approach where clean synthesized speech would be processed to resemble maritime VHF radio communication.

**Why:** You wanted to investigate whether model adaptation could eventually produce VHF-like audio directly, instead of always applying a channel simulation after synthesis.

**How:** You selected an approach described in our earlier conversations as Option B.

1. Generate clean maritime speech

2. Apply VHF channel simulation

3. Pair VHF-degraded audio with transcripts

4. Prepare training/validation/test splits

5. Fine-tune an eligible TTS model

6. Evaluate directly generated VHF-like speech

**Important distinction:** This is a confirmed research strategy, but I have not recovered proof of completed LoRA fine-tuning, training checkpoints or final VHF-adapted model evaluation. It must therefore remain separate from your confirmed pilot generation and QC experiments.

**GitHub documentation:** `docs/vhf-adaptation.md`

## Experiment 06 — Speaker-reference consistency

*August–September 2026 · Speaker verification*

**What:** You investigated whether speaker-reference audio chunks represented the same identity consistently and whether VESSEL and COAST_GUARD references were sufficiently distinguishable.

**Why:** A TTS model may produce speech with the wrong speaker identity or drift toward another voice. You needed a verification mechanism before accepting generated turns.

**How:** You developed tests comparing reference chunks and synthesized audio.

| Test   | Comparison                                  | What it checks                                                  |
| ------ | ------------------------------------------- | --------------------------------------------------------------- |
| Test 1 | Reference chunk A ↔ reference chunk B       | Whether two chunks from the intended speaker are consistent     |
| Test 2 | Reference chunk A ↔ generated speaker audio | Whether the synthesized voice resembles the reference           |
| Test 3 | Reference chunk B ↔ generated speaker audio | Whether the same conclusion holds using another reference chunk |

The speaker-verification investigation evolved through several approaches, including raw cosine similarity and a more advanced three-way backend:

- Speaker embeddings, including ECAPA-TDNN-based embeddings in earlier stages.
- Cosine-similarity scoring.
- PLDA scoring.
- Cohort-normalized and calibrated scores.
- A two-out-of-three voting mechanism for speaker decisions.

Known implementation files from this stage include:

```text
speaker_backend_3way.py
validate_speaker_backend_real_trials.py
diagnose_failed_samples.py
speaker_backend_3way.pkl
```

I also recovered references to 15 real control trials, five verified A/B controls and 45 failed-attempt diagnostics.

**Scientific purpose:** Determine whether failed TTS generation resulted from reference inconsistency, cross-speaker confusion, or instability in synthesized voices.

The exact validation commands and numerical results for all trials need to be recovered from your local logs before documenting the final thresholds.

**GitHub documentation:** `docs/speaker-verification-experiments.md`

## Experiment 07 — Pilot generation using Fish-Speech and Chatterbox

*September–October 2026 · Multi-speaker dialogue synthesis*

This is one of your most important confirmed implementation experiments.

**What:** You generated maritime dialogues under four TTS/reference configurations.

| TTS engine | Reference condition | Configuration ID |
| --- | --- | --- |
| Fish-Speech | Clean | `fishspeech_clean_ref` |
| Fish-Speech | Noisy | `fishspeech_noisy_ref` |
| Chatterbox | Clean | `chatterbox_clean_ref` |
| Chatterbox | Noisy | `chatterbox_noisy_ref` |

**Why:** You wanted to compare model behavior and investigate whether clean/noisy reference conditioning affected generation success, intelligibility, speaker consistency and QC rejection patterns.

**How:**

1. Load the pilot generation plan.
2. Parse maritime dialogue into VESSEL and COAST_GUARD turns.
3. Select the model and reference condition.
4. Generate each turn separately.
5. Run turn-level quality checks.
6. Retry rejected turns, with up to four attempts in the documented configuration.
7. Concatenate the accepted turns into one dialogue.
8. Run final-dialogue evaluation and save results.

### Confirmed generation command

```bash
python run_pilot_generation_dialogue.py \
  --plan workspace/data/pilot/metadata/pilot_generation_plan.csv \
  --fish_url http://127.0.0.1:8080/v1/tts \
  --rebuild_speaker_backend \
  --overwrite
```

The exact command is recovered from our earlier discussion, although the path must match your current working directory.

You also ran the script in resume mode. Your logs showed `Skipping finalized row ...` for completed configurations, and you investigated why processing appeared to restart from a particular sample index.

The pilot plan was described as containing 60 rows across four conditions. Later displayed logs covered row indices 0–44. I would verify the exact completed coverage before publishing a definitive count.

**GitHub documentation:** `experiments/pilot/README.md`

## Experiment 08 — Speaker drift and generation retry policy

*September 2026 · Speaker QC refinement*

**What:** You investigated cases where synthesized turns initially resembled the correct speaker but later turns sounded less consistent.

**Why:** Comparing every turn only against a reference may miss speaker drift within the same dialogue.

**How:** The earlier speaker QC approach compared a generated turn with:

- The intended speaker reference.
- The competing speaker reference.
- An accepted turn from the same speaker role, used as an anchor.

For example, a previously discussed Fish-Speech noisy-reference trial had:

| Metric                                     | Recorded value |
| ------------------------------------------ | -------------- |
| Turn 1 similarity to VESSEL reference      | 0.328840       |
| Turn 1 similarity to COAST_GUARD reference | 0.021512       |
| Difference between those similarities      | 0.307328       |
| Later turn's anchor similarity             | 0.599013       |
| Anchor threshold in that test              | 0.70           |

The later turn failed the anchor-consistency requirement despite having a relatively clearer relationship to the intended reference.

**Outcome:** The experiment illustrated why reference similarity and within-dialogue speaker consistency should be evaluated separately.

These thresholds are historical test values, not necessarily the thresholds used by the latest three-way backend.

**GitHub documentation:** `docs/speaker-drift-analysis.md`

## Experiment 09 — Automated turn-level quality control

*September–October 2026 · Quality-validation experiments*

**What:** You built and refined an automated QC framework to determine whether individual generated turns should be accepted, rejected, or held for review.

**Why:** Successful WAV generation does not guarantee an accurate maritime message. The synthesized audio may contain an incorrect call sign, missing content, an extra distress phrase, speaker inconsistency or signal defects.

**How:** Your system evolved toward the following six stages:

| Stage                | Evaluation                                                                              | Main purpose                       |
| -------------------- | --------------------------------------------------------------------------------------- | ---------------------------------- |
| 1. Signal QC         | WAV validity, duration, gaps/dropouts and other signal features                         | Check technical audio integrity    |
| 2. Speaker QC        | Embeddings, intended/other speaker comparison, cohort/PLDA scoring                      | Verify speaker identity            |
| 3. Message integrity | ASR similarity, missing/extra words, critical tokens, repetitions, OVER/OUT, truncation | Verify spoken maritime content     |
| 4. Prosody QC        | Global/local speech rate and reference-rate comparison                                  | Detect abnormal speaking rates     |
| 5. Naturalness QC    | Digital dropouts and additional audio defects                                           | Assess artifacts                   |
| 6. Final dialogue QC | Concatenated-audio ASR and dialogue-level consistency                                   | Validate the complete conversation |

Much of the ASR/prosody/message-integrity logic existed within `prosody_qc.py`, including the `MaritimeASRProsodyQC` class.

You investigated and refined specific rejection mechanisms, such as:

- `TRANSCRIPT_MISMATCH`
- `CRITICAL_TOKEN_MISSING`
- `UNEXPECTED_INSERTION`
- `EXPECTED_CONTENT_MISSING`
- `MESSAGE_UNEXPECTED_INSERTION`
- `LOCAL_SPEECH_RATE_DRIFT`
- `DIGITAL_DROPOUT`

You also discovered that some checks were too strict. For example, a dialogue with ASR similarity approximately `0.9989` was rejected by exact message-integrity rules, prompting discussion of minor-ASR mismatch tolerance.

Other modifications included normalizing numbers spoken digit by digit, case/spacing differences, `wind farm` versus `windfarm`, OVER/OUT counts, and truncation detection.

**GitHub documentation:** `docs/qc-framework.md` and `docs/message-integrity-experiments.md`

## Experiment 10 — Automatic QC versus human listening

*October 2026 · Human evaluation*

**What:** You manually listened to synthesized maritime dialogues and individual attempts, then recorded whether the automatic QC decisions matched human judgment.

**Why:** ASR can misrecognize correct audio, and objective checks may reject usable speech or accept perceptually incorrect messages. Human review provides an external reference for assessing the QC framework.

**How:** You used a manual review CSV containing:

```csv
sample_id,scope,turn_id,attempt,decision,review_reason
```

Examples previously recorded include:

| Sample                     | Human decision | Interpretation                      |
| -------------------------- | -------------- | ----------------------------------- |
| `chatterbox_clean_ref_003` | ACCEPT         | Full dialogue manually verified     |
| `chatterbox_clean_ref_005` | REJECT         | Full dialogue rejected by listening |
| `chatterbox_noisy_ref_001` | ACCEPT         | Full dialogue manually verified     |
| `chatterbox_noisy_ref_002` | REJECT         | Full dialogue rejected by listening |
| `fishspeech_clean_ref_004` | ACCEPT         | Full dialogue manually verified     |
| `fishspeech_noisy_ref_004` | ACCEPT         | Full dialogue manually verified     |

You also reviewed particular failed turns and multiple generation attempts, including cases where a model repeated MAYDAY too many times or inserted unexpected identifiers.

The examples above are individual annotations, not the full human-evaluation dataset.

**Scientific value:** This supports eventual measurements of false rejection, false acceptance, agreement between automated and human decisions, and error-category reliability.

**GitHub documentation:** `docs/human-evaluation.md`

## Experiment 11 — Chatterbox pronunciation normalization

*October 2026 · Text-normalization refinement*

**What:** You investigated whether Chatterbox could generate more accurate maritime terminology when abbreviations and selected identifiers were normalized before synthesis.

**Why:** The original runner passed `turn_text` directly to Chatterbox. Maritime abbreviations and identifiers may be pronounced incorrectly, affecting ASR scores and message intelligibility.

**How:** A modified script was discussed under the filename:

`run_pilot_generation_dialogue_chatterbox_normalized.py`

The change introduced `normalize_chatterbox_text()` and an additional logging field, `chatterbox_tts_text`.

Normalization rules considered included VHF, MMSI, GPS, AIS, Channel 16, VISION II and selected maritime abbreviations or number forms, while retaining the original dialogue text for QC.

The Fish-Speech API was separately reported to use `normalize=True`.

**Current status:** The script modification is documented in our history. I do not yet have verified before/after pronunciation or acceptance-rate results.
## Notes

- **Research/planning confirmed** does not imply an implemented or completed experiment.
- **Execution confirmed** indicates that earlier discussions included evidence of an actual run, but not necessarily a fully reproducible record or final result.
- For the final thesis repository, link each row to its corresponding experimental protocol, configuration, command, code revision, and outputs when available.
