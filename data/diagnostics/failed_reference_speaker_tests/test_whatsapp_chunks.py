from pathlib import Path

from run_pilot_generation_dialogue import SpeakerEmbeddingVerifier
from speaker_backend_3way import ThreeWaySpeakerBackend

"""
# =========================================================
# 1. FILE PATHS
# =========================================================
'''
chunk_a = Path(
    "workspace/data/diagnostics/whatsapp_057/"
    "whatsapp_chunk_A.wav"
)

chunk_b = Path(
    "workspace/data/diagnostics/whatsapp_057/"
    "whatsapp_chunk_B.wav"
)

original_reference = Path(
    "workspace/docs/Noisy_dataset/reference_voices/"
    "WhatsApp_audio_1.wav"
)

chatterbox_attempt = Path(
    "workspace/data/pilot/chatterbox_noisy_ref/turns/"
    "chatterbox_noisy_ref_057/attempts/"
    "001_vessel_attempt_2.wav"
)

fishspeech_attempt = Path(
    "workspace/data/pilot/fishspeech_noisy_ref/turns/"
    "fishspeech_noisy_ref_057/attempts/"
    "001_vessel_attempt_2.wav"
)
"""

# =========================================================
# 2. CHECK FILES
# =========================================================

for path in [
    chunk_a,
    chunk_b,
    original_reference,
    chatterbox_attempt,
    fishspeech_attempt,
]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


# =========================================================
# 3. FIND EXISTING TRAINED BACKEND
# =========================================================

model_candidates = list(
    Path(".").rglob("speaker_backend_3way.pkl")
)

if not model_candidates:
    raise FileNotFoundError(
        "Could not find speaker_backend_3way.pkl"
    )

if len(model_candidates) > 1:
    print("Found multiple backend models:")
    for path in model_candidates:
        print(" ", path)

    raise RuntimeError(
        "More than one speaker_backend_3way.pkl was found."
    )

model_path = model_candidates[0]

print(f"Using backend model: {model_path}")


# =========================================================
# 4. CREATE VERIFIER + BACKEND
# =========================================================

speaker_verifier = SpeakerEmbeddingVerifier()

speaker_backend = ThreeWaySpeakerBackend(
    verifier=speaker_verifier,
    model_path=str(model_path),
    rebuild=False,
)


# =========================================================
# 5. CREATE ALL EMBEDDINGS
# =========================================================

print("\nCreating Chunk A embedding...")
embedding_a = speaker_verifier.embedding(chunk_a)

print("Creating Chunk B embedding...")
embedding_b = speaker_verifier.embedding(chunk_b)

print("Creating Chatterbox embedding...")
embedding_chatterbox = speaker_verifier.embedding(
    chatterbox_attempt
)

print("Creating Fish-Speech embedding...")
embedding_fishspeech = speaker_verifier.embedding(
    fishspeech_attempt
)


# =========================================================
# 6. FIND ORIGINAL SPEAKER ID
# =========================================================

speaker_id = speaker_backend.speaker_id_for_path(
    original_reference
)

print(f"\nOriginal reference speaker ID: {speaker_id}")

exclude_ids = []

if speaker_id is not None:
    exclude_ids = [speaker_id]


# =========================================================
# TEST 1 — CHUNK A vs CHUNK B
# Known SAME real speaker
# =========================================================

result_real = speaker_backend.score_pair(
    embedding_a,
    embedding_b,
    exclude_speaker_ids=exclude_ids,
)

print("\n" + "=" * 60)
print("TEST 1: CHUNK A vs CHUNK B")
print("KNOWN REAL SAME-SPEAKER TEST")
print("=" * 60)

for key, value in result_real.items():
    print(f"{key}: {value}")


# =========================================================
# TEST 2A — CHUNK A vs CHATTERBOX
# =========================================================

result_a_chatterbox = speaker_backend.score_pair(
    embedding_a,
    embedding_chatterbox,
    exclude_speaker_ids=exclude_ids,
)

print("\n" + "=" * 60)
print("TEST 2A: CHUNK A vs CHATTERBOX ATTEMPT 2")
print("=" * 60)

for key, value in result_a_chatterbox.items():
    print(f"{key}: {value}")


# =========================================================
# TEST 2B — CHUNK A vs FISH-SPEECH
# =========================================================

result_a_fish = speaker_backend.score_pair(
    embedding_a,
    embedding_fishspeech,
    exclude_speaker_ids=exclude_ids,
)

print("\n" + "=" * 60)
print("TEST 2B: CHUNK A vs FISH-SPEECH ATTEMPT 2")
print("=" * 60)

for key, value in result_a_fish.items():
    print(f"{key}: {value}")


# =========================================================
# TEST 3A — CHUNK B vs CHATTERBOX
# =========================================================

result_b_chatterbox = speaker_backend.score_pair(
    embedding_b,
    embedding_chatterbox,
    exclude_speaker_ids=exclude_ids,
)

print("\n" + "=" * 60)
print("TEST 3A: CHUNK B vs CHATTERBOX ATTEMPT 2")
print("=" * 60)

for key, value in result_b_chatterbox.items():
    print(f"{key}: {value}")


# =========================================================
# TEST 3B — CHUNK B vs FISH-SPEECH
# =========================================================

result_b_fish = speaker_backend.score_pair(
    embedding_b,
    embedding_fishspeech,
    exclude_speaker_ids=exclude_ids,
)

print("\n" + "=" * 60)
print("TEST 3B: CHUNK B vs FISH-SPEECH ATTEMPT 2")
print("=" * 60)

for key, value in result_b_fish.items():
    print(f"{key}: {value}")


# =========================================================
# SIMPLE SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    f"Real A vs Real B: "
    f"{result_real['same_votes']}/3 "
    f"SAME -> {result_real['decision_same']}"
)

print(
    f"A vs Chatterbox: "
    f"{result_a_chatterbox['same_votes']}/3 "
    f"SAME -> {result_a_chatterbox['decision_same']}"
)

print(
    f"B vs Chatterbox: "
    f"{result_b_chatterbox['same_votes']}/3 "
    f"SAME -> {result_b_chatterbox['decision_same']}"
)

print(
    f"A vs Fish-Speech: "
    f"{result_a_fish['same_votes']}/3 "
    f"SAME -> {result_a_fish['decision_same']}"
)

print(
    f"B vs Fish-Speech: "
    f"{result_b_fish['same_votes']}/3 "
    f"SAME -> {result_b_fish['decision_same']}"
)