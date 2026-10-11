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
# 2. Load Chatterbox
# -----------------------------
model = ChatterboxTTS.from_pretrained(device=device)

# -----------------------------
# 3. Output folder
# -----------------------------
output_dir = "/data/kavya/conda_envs/chatterbox/baseline_outputs"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 4. Plain baseline text
# -----------------------------
text = (
    "Mayday, Mayday, Mayday. This is three one six zero one six one three one motor vessel PACIFIC GUARDIAN. We are under attack by armed assailants. Position is six nine degrees North, seven three degrees East. We have been targeted by armed men who have boarded our vessel. They are armed with firearms."
)

# -----------------------------
# 5. Generate baseline audio
# -----------------------------
wav = model.generate(text)

output_path = os.path.join(output_dir, "chatterbox_baseline_d001.wav")
ta.save(output_path, wav, model.sr)

print(f"Saved baseline audio: {output_path}")