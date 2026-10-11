#!/usr/bin/env python3
"""
Three-way speaker verification backend for the maritime TTS QC pipeline.

Methods
-------
1) Calibrated cosine likelihood evidence:
   ECAPA cosine score -> logistic calibration -> calibrated log-odds.

2) PLDA:
   ECAPA embeddings -> SpeechBrain Gaussian PLDA score -> logistic calibration.

3) Cohort-normalized cosine:
   ECAPA cosine score is S-normalized against unrelated speaker centroids,
   then logistic-calibrated.

The final SAME/DIFFERENT decision uses a fixed 2-of-3 vote. No hand-picked
cosine threshold is used for speaker identity decisions.

IMPORTANT
---------
Speaker references are imported directly from
create_pilot_generation_plan.py through SPEAKER_BACKEND_REFS.

Each reference used for backend training should contain one speaker only.
"""

from __future__ import annotations

import itertools
import math
import pickle
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as NNF
import torchaudio as ta
import torchaudio.functional as AF
from sklearn.linear_model import LogisticRegression
from create_pilot_generation_plan import SPEAKER_BACKEND_REFS

try:
    from speechbrain.processing.PLDA_LDA import (
        Ndx,
        PLDA,
        StatObject_SB,
        fast_PLDA_scoring,
    )
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "Three-way speaker backend requires SpeechBrain PLDA support.\n"
        "Install/upgrade SpeechBrain in the same environment:\n"
        "    pip install -U speechbrain\n"
        f"Original import error: {exc}"
    )


EPS = 1e-8


def _l2_normalize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(x)
    if norm <= EPS:
        return x
    return x / norm


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize_np(a)
    b = _l2_normalize_np(b)
    return float(np.dot(a, b))


def _make_stat_object(vectors, model_ids, segment_ids):
    vectors = np.asarray(vectors, dtype=np.float64)
    n = vectors.shape[0]
    return StatObject_SB(
        modelset=np.asarray(model_ids, dtype="|O"),
        segset=np.asarray(segment_ids, dtype="|O"),
        start=np.asarray([None] * n),
        stop=np.asarray([None] * n),
        stat0=np.asarray([[1.0]] * n, dtype=np.float64),
        stat1=vectors,
    )


def _score_plda_pair(embedding_a, embedding_b, plda_state) -> float:
    """
    Return one SpeechBrain PLDA score for one pair of embeddings.
    """

    a = _l2_normalize_np(embedding_a)
    b = _l2_normalize_np(embedding_b)

    # IMPORTANT:
    # modelset/segset IDs must match the IDs supplied to Ndx.
    enroll = _make_stat_object(
        [a],
        ["enroll"],
        ["enroll"],
    )

    test = _make_stat_object(
        [b],
        ["test"],
        ["test"],
    )

    ndx = Ndx(
        models=np.asarray(
            ["enroll"],
            dtype="|O",
        ),
        testsegs=np.asarray(
            ["test"],
            dtype="|O",
        ),
    )

    scores = fast_PLDA_scoring(
        enroll,
        test,
        ndx,
        plda_state["mean"],
        plda_state["F"],
        plda_state["Sigma"],
    )

    return float(
        scores.scoremat[0, 0]
    )


def _decision_function(calibrator, score: float) -> float:
    """
    Logistic-regression decision_function gives calibrated log-odds.

    Training trials are balanced 50/50 SAME vs DIFFERENT, so zero is the
    equal-prior boundary. Positive means evidence favors SAME; negative means
    evidence favors DIFFERENT.
    """
    return float(
        calibrator.decision_function(
            np.asarray([[float(score)]], dtype=np.float64)
        )[0]
    )


class ThreeWaySpeakerBackend:
    def __init__(
        self,
        verifier,
        model_path="workspace/models/speaker_backend_3way.pkl",
        rebuild=False,
        chunk_sec=3.0,
        hop_sec=1.5,
        min_chunk_sec=1.5,
        min_speakers=5,
        random_seed=20260914,
    ):
        self.verifier = verifier
        self.model_path = Path(model_path)
        self.chunk_sec = float(chunk_sec)
        self.hop_sec = float(hop_sec)
        self.min_chunk_sec = float(min_chunk_sec)
        self.min_speakers = int(min_speakers)
        self.random_seed = int(random_seed)

        self.path_to_speaker_id = {}
        self.cohort_centroids = {}
        self.plda_state = None
        self.cosine_calibrator = None
        self.plda_calibrator = None
        self.cohort_calibrator = None
        self.training_summary = {}

        if rebuild or not self.model_path.exists():
            self.fit_and_save()
        else:
            self.load()

    # ------------------------------------------------------------------
    # Manifest / audio preparation
    # ------------------------------------------------------------------
    def _build_reference_table(self):
        """
        Build the speaker-backend reference table directly from
        SPEAKER_BACKEND_REFS defined in create_pilot_generation_plan.py.

        For now, each reference WAV is treated as one separate speaker.
        """

        rows = []
        seen_paths = set()

        for audio_path in SPEAKER_BACKEND_REFS:
            audio_path = str(audio_path).strip()

            if not audio_path:
                continue

            resolved = str(Path(audio_path).resolve())

            # Avoid accidentally adding the same WAV twice.
            if resolved in seen_paths:
                continue

            seen_paths.add(resolved)

            if not Path(audio_path).exists():
                raise FileNotFoundError(
                    f"Speaker-backend reference not found: {audio_path}"
                )

            speaker_id = Path(audio_path).stem

            rows.append(
                {
                    "speaker_id": speaker_id,
                    "audio_path": audio_path,
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            raise ValueError(
                "SPEAKER_BACKEND_REFS is empty."
            )

        speaker_count = df["speaker_id"].nunique()

        if speaker_count < self.min_speakers:
            raise ValueError(
                f"Need at least {self.min_speakers} distinct speakers "
                f"for the 3-way backend; currently found {speaker_count}."
            )

        self.path_to_speaker_id = {
            str(Path(row.audio_path).resolve()): str(row.speaker_id)
            for row in df.itertuples(index=False)
        }

        return df

    @staticmethod
    def _load_mono_16k(path):
        wav, sr = ta.load(str(path))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = AF.resample(wav, sr, 16000)
        return wav

    def _chunk_audio(self, path):
        wav = self._load_mono_16k(path)
        sr = 16000
        chunk_n = int(self.chunk_sec * sr)
        hop_n = int(self.hop_sec * sr)
        min_n = int(self.min_chunk_sec * sr)

        chunks = []
        start = 0
        index = 0
        while start < wav.shape[1]:
            chunk = wav[:, start:start + chunk_n]
            if chunk.shape[1] < min_n:
                break

            # Skip nearly silent chunks.
            rms = torch.sqrt(torch.mean(chunk * chunk) + 1e-12).item()
            dbfs = 20.0 * math.log10(max(rms, 1e-12))
            if dbfs >= -45.0:
                index += 1
                chunks.append((index, chunk))

            start += hop_n

        return chunks

    def _extract_training_embeddings(self, df):
        records = []

        temp_root = self.model_path.parent / ".speaker_backend_chunks"
        temp_root.mkdir(parents=True, exist_ok=True)

        try:
            for row_idx, row in enumerate(df.itertuples(index=False), start=1):
                speaker_id = str(row.speaker_id)
                audio_path = Path(row.audio_path)
                chunks = self._chunk_audio(audio_path)

                print(
                    f"\n[{row_idx}/{len(df)}] Reference: {audio_path.name}"
                )
                print(
                    f"    speaker_id: {speaker_id}"
                )
                print(
                    f"    usable chunks: {len(chunks)}"
                )

                if not chunks:
                    raise ValueError(
                        f"No usable >= {self.min_chunk_sec:.1f}s speech chunks "
                        f"found in: {audio_path}"
                    )

                for chunk_idx, chunk in chunks:
                    print(
                        f"    ECAPA embedding chunk "
                        f"{chunk_idx}/{len(chunks)}",
                        flush=True,
                    )
                    temp_path = temp_root / (
                        f"spk_{speaker_id}_file_{row_idx}_chunk_{chunk_idx}.wav"
                    )
                    ta.save(str(temp_path), chunk, 16000)
                    emb = self.verifier.embedding(temp_path)
                    emb_np = _l2_normalize_np(emb.detach().cpu().numpy())
                    records.append(
                        {
                            "speaker_id": speaker_id,
                            "segment_id": (
                                f"{speaker_id}__{row_idx:03d}__{chunk_idx:03d}"
                            ),
                            "embedding": emb_np,
                        }
                    )
        finally:
            if temp_root.exists():
                for item in temp_root.glob("*.wav"):
                    try:
                        item.unlink()
                    except Exception:
                        pass

        counts = {}
        for rec in records:
            counts[rec["speaker_id"]] = counts.get(rec["speaker_id"], 0) + 1

        bad = {spk: n for spk, n in counts.items() if n < 2}
        if bad:
            raise ValueError(
                "PLDA/calibration needs at least 2 usable chunks per speaker. "
                f"Too few chunks for: {bad}"
            )

        return records

    # ------------------------------------------------------------------
    # Cohort normalization
    # ------------------------------------------------------------------
    def _build_cohort_centroids(self, records):
        grouped = {}
        for rec in records:
            grouped.setdefault(rec["speaker_id"], []).append(rec["embedding"])

        centroids = {}
        for speaker_id, vectors in grouped.items():
            centroid = np.mean(np.stack(vectors, axis=0), axis=0)
            centroids[speaker_id] = _l2_normalize_np(centroid)
        return centroids

    def cohort_snorm_score(
        self,
        embedding_a,
        embedding_b,
        exclude_speaker_ids=None,
    ) -> float:
        """
        Symmetric cohort-normalized cosine (S-norm style).

        It asks whether the A<->B target cosine is unusually high relative to
        each embeddings similarities to many unrelated cohort speakers.
        """
        exclude_speaker_ids = set(
            x for x in (exclude_speaker_ids or []) if x
        )
        a = _l2_normalize_np(embedding_a)
        b = _l2_normalize_np(embedding_b)
        target = _cosine_np(a, b)

        cohort = [
            c
            for speaker_id, c in self.cohort_centroids.items()
            if speaker_id not in exclude_speaker_ids
        ]
        if len(cohort) < 3:
            cohort = list(self.cohort_centroids.values())

        cohort_mat = np.stack(cohort, axis=0)
        scores_a = cohort_mat @ a
        scores_b = cohort_mat @ b

        mean_a = float(np.mean(scores_a))
        std_a = float(np.std(scores_a) + EPS)
        mean_b = float(np.mean(scores_b))
        std_b = float(np.std(scores_b) + EPS)

        z_a = (target - mean_a) / std_a
        z_b = (target - mean_b) / std_b
        return float(0.5 * (z_a + z_b))

    # ------------------------------------------------------------------
    # PLDA
    # ------------------------------------------------------------------
    def _train_plda(self, records):
        vectors = np.stack([rec["embedding"] for rec in records], axis=0)
        model_ids = [rec["speaker_id"] for rec in records]
        segment_ids = [rec["segment_id"] for rec in records]
        stat = _make_stat_object(vectors, model_ids, segment_ids)

        n_speakers = len(set(model_ids))
        emb_dim = vectors.shape[1]
        rank_f = max(1, min(20, n_speakers - 1, emb_dim - 1))

        plda = PLDA(
            rank_f=rank_f,
            nb_iter=10,
        )
        plda.plda(stat)

        return {
            "mean": plda.mean,
            "F": plda.F,
            "Sigma": plda.Sigma,
            "rank_f": rank_f,
        }

    # ------------------------------------------------------------------
    # Balanced SAME/DIFFERENT calibration trials
    # ------------------------------------------------------------------
    def _make_calibration_trials(self, records):
        by_speaker = {}
        for rec in records:
            by_speaker.setdefault(rec["speaker_id"], []).append(rec)

        same_pairs = []
        for speaker_id, items in by_speaker.items():
            for a, b in itertools.combinations(items, 2):
                same_pairs.append((a, b, 1))

        different_pairs = []
        speaker_ids = sorted(by_speaker)
        for spk_a, spk_b in itertools.combinations(speaker_ids, 2):
            for a in by_speaker[spk_a]:
                for b in by_speaker[spk_b]:
                    different_pairs.append((a, b, 0))

        if not same_pairs or not different_pairs:
            raise ValueError(
                "Could not create both SAME and DIFFERENT calibration pairs."
            )

        rng = random.Random(self.random_seed)
        rng.shuffle(same_pairs)
        rng.shuffle(different_pairs)

        n = min(len(same_pairs), len(different_pairs), 5000)
        same_pairs = same_pairs[:n]
        different_pairs = different_pairs[:n]

        trials = same_pairs + different_pairs
        rng.shuffle(trials)
        return trials

    def _train_calibrators(self, trials):
        cosine_scores = []
        plda_scores = []
        cohort_scores = []
        labels = []

        for a, b, label in trials:
            emb_a = a["embedding"]
            emb_b = b["embedding"]

            cosine_scores.append(_cosine_np(emb_a, emb_b))
            plda_scores.append(
                _score_plda_pair(emb_a, emb_b, self.plda_state)
            )
            cohort_scores.append(
                self.cohort_snorm_score(
                    emb_a,
                    emb_b,
                    exclude_speaker_ids=[
                        a["speaker_id"],
                        b["speaker_id"],
                    ],
                )
            )
            labels.append(label)

        y = np.asarray(labels, dtype=np.int64)

        def fit_one(scores):
            x = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
            model = LogisticRegression(
                C=1e6,
                solver="lbfgs",
                max_iter=2000,
                random_state=self.random_seed,
            )
            model.fit(x, y)
            return model

        self.cosine_calibrator = fit_one(cosine_scores)
        self.plda_calibrator = fit_one(plda_scores)
        self.cohort_calibrator = fit_one(cohort_scores)

        return {
            "same_trials": int(np.sum(y == 1)),
            "different_trials": int(np.sum(y == 0)),
            "cosine_same_mean": float(np.mean(np.asarray(cosine_scores)[y == 1])),
            "cosine_diff_mean": float(np.mean(np.asarray(cosine_scores)[y == 0])),
            "plda_same_mean": float(np.mean(np.asarray(plda_scores)[y == 1])),
            "plda_diff_mean": float(np.mean(np.asarray(plda_scores)[y == 0])),
            "cohort_same_mean": float(np.mean(np.asarray(cohort_scores)[y == 1])),
            "cohort_diff_mean": float(np.mean(np.asarray(cohort_scores)[y == 0])),
        }

    # ------------------------------------------------------------------
    # Public training / persistence
    # ------------------------------------------------------------------
    def fit_and_save(self):
        print("\nTraining 3-way speaker backend...")
        print(
            f"  references loaded from "
            f"create_pilot_generation_plan.py: "
            f"{len(SPEAKER_BACKEND_REFS)}"
        )

        df = self._build_reference_table()
        records = self._extract_training_embeddings(df)

        self.cohort_centroids = self._build_cohort_centroids(records)
        self.plda_state = self._train_plda(records)
        trials = self._make_calibration_trials(records)
        calibration_summary = self._train_calibrators(trials)

        self.training_summary = {
            "speaker_count": int(df["speaker_id"].nunique()),
            "audio_file_count": int(len(df)),
            "embedding_chunk_count": int(len(records)),
            "plda_rank_f": int(self.plda_state["rank_f"]),
            **calibration_summary,
        }

        payload = {
            "path_to_speaker_id": self.path_to_speaker_id,
            "cohort_centroids": self.cohort_centroids,
            "plda_state": self.plda_state,
            "cosine_calibrator": self.cosine_calibrator,
            "plda_calibrator": self.plda_calibrator,
            "cohort_calibrator": self.cohort_calibrator,
            "training_summary": self.training_summary,
        }

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as handle:
            pickle.dump(payload, handle)

        print(f"  saved backend: {self.model_path}")
        print(f"  speakers: {self.training_summary['speaker_count']}")
        print(f"  chunks: {self.training_summary['embedding_chunk_count']}")
        print(
            "  balanced calibration trials: "
            f"{self.training_summary['same_trials']} SAME + "
            f"{self.training_summary['different_trials']} DIFFERENT"
        )

    def load(self):
        with open(self.model_path, "rb") as handle:
            payload = pickle.load(handle)

        self.path_to_speaker_id = payload["path_to_speaker_id"]
        self.cohort_centroids = payload["cohort_centroids"]
        self.plda_state = payload["plda_state"]
        self.cosine_calibrator = payload["cosine_calibrator"]
        self.plda_calibrator = payload["plda_calibrator"]
        self.cohort_calibrator = payload["cohort_calibrator"]
        self.training_summary = payload.get("training_summary", {})

        print(f"Loaded 3-way speaker backend: {self.model_path}")

    def speaker_id_for_path(self, path):
        try:
            return self.path_to_speaker_id.get(str(Path(path).resolve()), "")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Runtime scoring
    # ------------------------------------------------------------------
    def score_pair(
        self,
        embedding_a,
        embedding_b,
        exclude_speaker_ids=None,
    ):
        if torch.is_tensor(embedding_a):
            embedding_a = embedding_a.detach().cpu().numpy()
        if torch.is_tensor(embedding_b):
            embedding_b = embedding_b.detach().cpu().numpy()

        a = _l2_normalize_np(embedding_a)
        b = _l2_normalize_np(embedding_b)

        raw_cosine = _cosine_np(a, b)
        raw_plda = _score_plda_pair(a, b, self.plda_state)
        raw_cohort = self.cohort_snorm_score(
            a,
            b,
            exclude_speaker_ids=exclude_speaker_ids,
        )

        cosine_llr = _decision_function(self.cosine_calibrator, raw_cosine)
        plda_llr = _decision_function(self.plda_calibrator, raw_plda)
        cohort_llr = _decision_function(self.cohort_calibrator, raw_cohort)

        votes = {
            "calibrated_cosine_same": cosine_llr > 0.0,
            "plda_same": plda_llr > 0.0,
            "cohort_same": cohort_llr > 0.0,
        }
        same_votes = int(sum(bool(v) for v in votes.values()))
        decision_same = same_votes >= 2

        return {
            "raw_cosine": round(raw_cosine, 6),
            "raw_plda_score": round(raw_plda, 6),
            "raw_cohort_snorm": round(raw_cohort, 6),
            "calibrated_cosine_llr": round(cosine_llr, 6),
            "calibrated_plda_llr": round(plda_llr, 6),
            "calibrated_cohort_llr": round(cohort_llr, 6),
            "calibrated_cosine_same": bool(votes["calibrated_cosine_same"]),
            "plda_same": bool(votes["plda_same"]),
            "cohort_same": bool(votes["cohort_same"]),
            "same_votes": same_votes,
            "decision_same": bool(decision_same),
        }
