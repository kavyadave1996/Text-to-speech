# TTS Model Evaluation

This document contains evaluation examples for the TTS models in this repository.
All audio links are relative to this document, which is located in
`workspace/docs/`.

---

## Clean Dataset

| Test | Reference Voice | Chatterbox Output | FishSpeech Output |
|---|---|---|---|
| Northern Belle | [Reference](<clean_dataset/reference_voices/(41) Northern Belle_ A hero's last call_clean_audio.wav>) | — | [FishSpeech](<clean_dataset/fishspeech_clean_output/audio_fishspeech_clean_audio_northenbell.wav>) |
| Hormuz | [Reference](<clean_dataset/reference_voices/(51) Audio captures captain’s distress call clean audio Hormuz.wav>) | — | [FishSpeech](<clean_dataset/fishspeech_clean_output/audio_clean_audio_fishspeech_hormuz.wav>) |
| Boaters Mayday | [Reference](<clean_dataset/reference_voices/(54) Boaters mayday call_clean_Audio.wav>) | — | [FishSpeech](<clean_dataset/fishspeech_clean_output/audio_fishspeech_clean_audio_boaters_call.wav>) |
| WhatsApp Clean Audio | [Reference](<clean_dataset/reference_voices/WhatsApp_clean_audio_1.wav>) | — | [FishSpeech](<clean_dataset/fishspeech_clean_output/whatsapp_clean_audio_fishspeech_1.wav>) |

---

## Noisy Dataset

| Test | Reference Voice | Chatterbox Output | FishSpeech Output |
|---|---|---|---|
| Northern Belle | [Reference](<Noisy_dataset/reference_voices/Northern_Belle.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_northenbell.wav>) | — |
| Hormuz | [Reference](<Noisy_dataset/reference_voices/(51) Audio captures captain’s distress call after Iran fires on Indian tanker in Hormuz.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_hormuz.wav>) | [FishSpeech](<Noisy_dataset/fishspeech_output_noisy/audio_no_clean_audio_fishspeech_hormuz.wav>) |
| Boaters Mayday | [Reference](<Noisy_dataset/reference_voices/(54) Boaters mayday call.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_(54) Boaters mayday call.wav>) | [FishSpeech](<Noisy_dataset/fishspeech_output_noisy/audio_no_clean_audio_fishspeech_boaters_mayday.wav>) |
| New Hampshire Fishing Vessel | [Reference](<Noisy_dataset/reference_voices/(51) Mayday Call from Fishing Vessel Off New Hampshire.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_Hampshire.wav>) | [FishSpeech](<Noisy_dataset/fishspeech_output_noisy/audio_no_clean_audio_fishspeech_hampshire.wav>) |
| Yacht Sinking | [Reference](<Noisy_dataset/reference_voices/(51) Yacht's mayday call_ 'We are sinking'.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_(51) Yacht's.wav>) | — |
| Jet Blue Medical Emergency | [Reference](<Noisy_dataset/reference_voices/(51)Mayday call jet blue airline landing medical emergency but atc stop him %23airplane %23atcmemes.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_medical emergency.wav>) | [FishSpeech](<Noisy_dataset/fishspeech_output_noisy/audio_no_clean_audio_fishspeech_jet_blue.wav>) |
| Example Mayday Message | [Reference](<Noisy_dataset/reference_voices/(54) Example Mayday Message.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_(54) Example Mayday Message.wav>) | [FishSpeech](<Noisy_dataset/fishspeech_output_noisy/audio_no_clean_audio_fishspeech_Mayday_Message.wav>) |
| Saginaw Bay Capsized Vessel | [Reference](<Noisy_dataset/reference_voices/(54) Mayday call from capsized vessel on Saginaw Bay.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_(54) Mayday call from capsized vessel on Saginaw Bay.wav>) | — |
| French Coastguard | [Reference](<Noisy_dataset/reference_voices/French coastguard's mayday call after boat capsized – audio.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_French coastguard's mayday call after boat capsized_audio.wav>) | — |
| Motorboat / Bluefin | [Reference](<Noisy_dataset/reference_voices/(41) Practical Boating MBM How to do a mayday DSC Motor Boats Monthly MBM.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_motorboat_bluefin.wav>) | [FishSpeech](<Noisy_dataset/fishspeech_output_noisy/audio_no_clean_audio_fishspeech_motorbluefin.wav>) |
| WhatsApp Audio 1 | [Reference](<Noisy_dataset/reference_voices/WhatsApp_audio_1.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_WhatsApp_audio_1.wav>) | — |
| WhatsApp Audio 2 | [Reference](<Noisy_dataset/reference_voices/WhatsApp Audio 2026-08-01 at 08.13.34.wav>) | [Chatterbox](<Noisy_dataset/chatterbox_output_noisy/chatterbox_reference_WhatsApp_audio_2.wav>) | — |