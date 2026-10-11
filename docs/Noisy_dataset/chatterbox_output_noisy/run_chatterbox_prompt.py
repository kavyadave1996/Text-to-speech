import os
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# -----------------------------
# 1. Device
# -----------------------------
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# -----------------------------
# 2. Load Chatterbox model
# -----------------------------
model = ChatterboxTTS.from_pretrained(device=device)

# -----------------------------
# 3. Output folder
# -----------------------------
output_dir = "/data/kavya/conda_envs/chatterbox/prompt_outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 4. Prompt-adapted formatted text
# -----------------------------
text = (
     "MAYDAY! MAYDAY! MAYDAY! "
    "This is motor vessel PACIFIC GUARDIAN! "
    "M M S I: Three One Six Zero One Six One Three One! "
    "We are under attack by armed assailants! "
    "Position: Six Nine degrees North, Seven Three degrees East! "
    "We have been targeted by armed men who have boarded our vessel! "
    "They are armed with firearms! "
)

# -----------------------------
# 5. Generate prompt-adapted audio
# -----------------------------
wav = model.generate(
    text,
    exaggeration=0.7,
    cfg_weight=0.3
)

output_path = os.path.join(output_dir, "chatterbox_prompt_d001.wav")
ta.save(output_path, wav, model.sr)

print(f"Saved prompt-adapted audio: {output_path}")