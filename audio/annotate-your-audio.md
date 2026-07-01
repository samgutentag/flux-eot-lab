# Bring your own audio (annotation assist)

The synthetic clips this lab ships with have exact ground truth: `generate_audio.py`
places every sample, so it knows precisely where each turn's speech ends. Your own
recordings do not come with that, so `annotate_real.py` drafts it for you using
Deepgram's own pre-recorded speech-to-text (not Flux), and you review the draft.

This is a scaffold, a starting point. It gets you most of the way to a sidecar so
the review is a few nudges, not hours in an audio editor.

## 0. The honest caveat (read first)

A word-end timestamp is close to, but not exactly, the true end of speech (there is
trailing breath and filler). So the draft is a proposal, and **you are the final
authority on `true_end_ms`.** You do not need sample-perfect labels: the +/-200ms
tolerance window applied at analysis time absorbs small errors. You do need honest
ones, so listen and correct anything clearly off.

## 1. Convert your recording to 16 kHz mono linear16 WAV

The harness only accepts 16 kHz mono `linear16` WAV. Convert with ffmpeg:

```bash
ffmpeg -i your-recording.m4a -ac 1 -ar 16000 -sample_fmt s16 your-clip.wav
```

(`-ac 1` is mono, `-ar 16000` is 16 kHz, `-sample_fmt s16` is 16-bit linear16.)

## 2. Draft the sidecar

```bash
python audio/annotate_real.py your-clip.wav --class noisy_single
# crosstalk (two voices)? the louder/longer speaker is primary by default:
python audio/annotate_real.py call.wav --class crosstalk --primary-speaker 0
```

This writes a DRAFT sidecar to `audio/clips/your-clip.json`. Every turn carries a
`_draft_review` marker so you cannot forget to check it.

## 3. Review the draft (the part only you can do)

Open the sidecar, play the clip, and for each turn:

- Confirm `true_end_ms` lands where the speaker actually stops (trim trailing breath
  or "um" if the word-end ran long).
- Confirm the turn boundaries (merge or split if the segmentation grouped wrong).
- For `crosstalk`, confirm the `distractor_spans` line up with the second voice.

Remove the `_draft_review` markers once each turn is confirmed.

## 4. Register and run

- Put the WAV at `audio/your-clip.wav` (or point `audio_file` at its path).
- Add the sidecar to `audio/manifest.json`.
- Run `flux-bench --input audio/manifest.json --output results/` to sweep and get
  the per-class recommendation.

## Which scenarios should you test?

Do not test all four classes. Pick the one or two that match where you actually
deploy, record 5 to 10 representative clips of your own audio for those, and tune.
The whole point is to measure your conditions, not someone else's.

| Class | Real-world deployment |
| --- | --- |
| `clean_short` | Voice assistant, command interface, IVR in a quiet space |
| `clean_long` | Dictation, voice notes, "thinking out loud" assistants |
| `noisy_single` | Call center, drive-thru, mobile, in-car, field |
| `crosstalk` | Multi-party calls, open office, anything with a second voice |

**The strongest fit is the call center.** Real call audio is phone-band, noisy, and
two-party, which is exactly the noisy and crosstalk conditions where the defaults are
wrong and tuning pays off the most. Run your real calls through it, find your
threshold, and you have measured numbers for your own conditions instead of a guess.

One reminder on scope: this tool tunes Flux end-of-turn **detection** for your audio.
Your downstream large-language-model and text-to-speech pipeline is your own choice
and is out of scope here, so the answer you get is the detection setting, not your
end-to-end latency.
