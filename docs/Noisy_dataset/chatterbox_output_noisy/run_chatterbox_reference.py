import os
import torch
import torchaudio as ta
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

# Device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load model
model = ChatterboxTTS.from_pretrained(device=device)

# Output folder
output_dir = "/data/kavya/conda_envs/chatterbox/reference_outputs"
os.makedirs(output_dir, exist_ok=True)

# Emotional distress reference audio
AUDIO_PROMPT_PATH = "/data/kavya/conda_envs/chatterbox/WhatsApp_audio_2.wav"

if not Path(AUDIO_PROMPT_PATH).exists():
    raise FileNotFoundError(f"Reference audio not found: {AUDIO_PROMPT_PATH}")

# Target text: formatted distress text
text = (
   
    "MAYDAY! MAYDAY! MAYDAY! "
    "This is motor vessel F R B Zero Niner! "
    "M M S I: Two One Nine Zero One Six Seven Five Eight! "
    "We are under attack by armed men! "
    "Position: Seven Seven degrees North, Nine Nine degrees East! "
    "We are defending the vessel! "
    "We request immediate military assistance! "
    "OVER!")
# Generate with emotional reference audio
wav = model.generate(
    text,
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.7,
    cfg_weight=0.3
)

output_path = os.path.join(output_dir, "chatterbox_reference_WhatsApp_audio_2.wav")
ta.save(output_path, wav, model.sr)

print(f"Saved reference-adapted audio: {output_path}")