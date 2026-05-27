# Voice Reference Files

Place your voice cloning reference recordings here.

## Required files

| File | Who | Notes |
|---|---|---|
| `nova_reference.wav` | Nova's voice | 10–30s, clean speech, no background noise |
| `simona_reference.wav` | Simona's voice | 10–30s, clean speech, no background noise |

## Recording tips

- **Duration**: 10–30 seconds. More is better up to ~30s.
- **Format**: WAV, any sample rate (XTTS will resample). Mono or stereo.
- **Content**: Read any text naturally — a paragraph from a book works well.
- **Environment**: Quiet room. No reverb. No music.
- **Mic**: Any microphone including the laptop mic. Quality helps but isn't critical.

## Language

XTTS v2 clones the voice AND the language from the reference.
If you want Nova to speak Bulgarian, record `nova_reference.wav` in Bulgarian.

Supported languages include: `en` `bg` `de` `fr` `es` `it` `pt` `ru` `tr` `pl` `nl` `cs` `ar` `zh-cn` `ja` `hu` `ko`

## First run

After placing the files, trigger the model download once:

```bash
python tts_engine.py --download
```

Then test voice cloning:

```bash
python tts_engine.py --test
```
