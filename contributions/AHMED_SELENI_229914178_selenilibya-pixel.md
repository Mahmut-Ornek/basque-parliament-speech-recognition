# Contribution: Ahmed Seleni (229914178, @selinilibya-pixel)

Role: licence and citation files, timing, runtime table, two benchmark models, figures.

## What I did

**Licence and citation files**
- `LICENSE`: MIT for our code, plus a paragraph saying the data in `data/` is the Basque Parliament Speech Corpus 1.0 (CC0).
- `CITATION.md`: the corpus and its paper (doi:10.3390/app14051951), and every model in the results table with its link and licence, including the CC-BY-NC-SA licence of `Ansu/mHubert-basque-ASR`. I checked each licence on the model's Hugging Face page.

**Timing**
- `src/timing.py`: the timing pass. It uses the same 100 dev clips for every model, with 5 warm-up clips, batch size 1, on a Colab T4. The clock runs from "audio in memory" to "text returned", and model loading is excluded. It records RTFx, real-time factor, median and 95th-percentile time per clip, the GPU name and the precision.
- `src/runtime_table.py`: reads the timing and training-log JSON files and writes `results/runtime_table.md`. Nobody edits that table by hand.
- Timing files I ran on Colab T4: <LIST: e.g. whisper-large-v3, mhubert-basque, ...>
- Timing files run by others: <LIST: e.g. NeMo models by Tuğra>

**Benchmark runs**
- `whisper-large-v3` and `mhubert-basque` on dev and test with `src/transcribe.py`: <DONE / NOT DONE>, run on <HARDWARE>.
- `qwen3-asr`: <DONE by me / run by NAME / not done>.

**Figures**
- <`results/figures/fig3_speed_accuracy.png` and `src/plot_figures.py`: DONE / NOT DONE>. The benchmark and cleaning figures are Shahem's `wer_ci.png`.

## How I worked
- I committed my own files from my own account.
- I used an AI assistant (Claude) to draft `LICENSE`, `CITATION.md`, `timing.py` and `runtime_table.py`. I <reviewed / tested / changed> them: <WHAT YOU CHANGED OR CHECKED>.
- Nothing was chosen using test scores. The timing clips come from dev only.

## What did not get done
- <ANYTHING MISSING, OR "Nothing.">
