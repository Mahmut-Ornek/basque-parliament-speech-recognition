# Citation and licences

Licences below were read from each model's Hugging Face page on 22 November 2026.
If a page changes later, the page is the authority, not this file.

## Corpus

**Basque Parliament Speech Corpus 1.0** (`gttsehu/basque_parliament_1`, Basque part).
Licence: **CC0 1.0** (public domain dedication).

- Dataset page: https://huggingface.co/datasets/gttsehu/basque_parliament_1
- Dataset commit we used: `46208b3001c500aa53ddb5f4a9da81493bbb8067`
- Paper: https://doi.org/10.3390/app14051951

```bibtex
@article{varona2024bilingual,
  author  = {Varona, Amparo and Penagarikano, Mikel and Bordel, Germ{\'a}n and Rodriguez-Fuentes, Luis Javier},
  title   = {A Bilingual Basque--Spanish Dataset of Parliamentary Sessions for the Development and Evaluation of Speech Technology},
  journal = {Applied Sciences},
  volume  = {14},
  number  = {5},
  pages   = {1951},
  year    = {2024},
  doi     = {10.3390/app14051951}
}
```

## Models

One row per run name in `results/results_table.*`.

| Run name | Model | Link | Licence |
|---|---|---|---|
| `whisper-large-v3` | OpenAI Whisper large-v3 | https://huggingface.co/openai/whisper-large-v3 | Apache-2.0 |
| `whisper-small-zeroshot` and all `ft-*` runs | OpenAI Whisper small (the `ft-*` runs are our fine-tunes of it) | https://huggingface.co/openai/whisper-small | Apache-2.0 |
| `qwen3-asr` | Qwen3-ASR 1.7B | https://huggingface.co/Qwen/Qwen3-ASR-1.7B | Apache-2.0 |
| `parakeet-v3` | NVIDIA Parakeet TDT 0.6B v3 | https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 | CC-BY-4.0 |
| `canary-v2` | NVIDIA Canary 1B v2 | https://huggingface.co/nvidia/canary-1b-v2 | CC-BY-4.0 |
| `hitz-ctc` | HiTZ/Aholab Conformer-CTC large (Basque) | https://huggingface.co/HiTZ/stt_eu_conformer_ctc_large | Apache-2.0 |
| `hitz-transducer` | HiTZ/Aholab Conformer-Transducer large (Basque) | https://huggingface.co/HiTZ/stt_eu_conformer_transducer_large | Apache-2.0 |
| `wav2vec2-bp1` | wav2vec 2.0 XLS-R 300M, fine-tuned on the Basque Parliament corpus (Spanish + Basque) | https://huggingface.co/gttsehu/wav2vec2-xls-r-300m-bp1-es_eu | Apache-2.0 |
| `mhubert-basque` | mHuBERT-147 fine-tuned for Basque CTC ASR (Ansu) | https://huggingface.co/Ansu/mHubert-basque-ASR | **CC-BY-NC-SA-4.0** |

### Notes on the licences

- **`Ansu/mHubert-basque-ASR` is non-commercial and share-alike (CC-BY-NC-SA-4.0).**
  We use it only to produce benchmark predictions for this coursework. We do not
  redistribute its weights (`models/` is in `.gitignore`). Its predictions in
  `results/predictions/` are model output for an academic comparison; anyone
  reusing them should check the NC and SA terms for their own use. The model
  card lists `utter-project/mHuBERT-147` as its base model and
  `asierhv/composite_corpus_eu_v2.1` as its training data.
- **NVIDIA Parakeet and Canary (CC-BY-4.0)** require attribution, which this file
  gives.
- **Both HiTZ models and `wav2vec2-bp1` were trained on this corpus's training
  data** (the HiTZ cards list `gttsehu/basque_parliament_1`; the HiTZ transducer
  card names its `train_clean` split). They are in-domain baselines.
- The HiTZ cards show Apache-2.0 today. An older revision of the transducer card
  said CC-BY-4.0, so check the revision hash recorded in each run's JSON file if
  the difference matters.
- Our own fine-tuned Whisper-small weights are derivatives of an Apache-2.0
  model. They are not published.

## Papers behind the models

- Whisper: Radford et al., *Robust Speech Recognition via Large-Scale Weak
  Supervision*, arXiv:2212.04356.
- Qwen3-ASR: Qwen Team, *Qwen3-ASR Technical Report*, arXiv:2601.21337.
- Canary-1B-v2 and Parakeet-TDT-0.6B-v3: Sekoyan, Koluguri, Tadevosyan et al.,
  *Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models
  for Multilingual ASR and AST*, arXiv:2509.14128.
- Conformer (the HiTZ architecture): Gulati et al., *Conformer:
  Convolution-augmented Transformer for Speech Recognition*, arXiv:2005.08100.

## Software

Evaluation uses `jiwer`; models run through Hugging Face `transformers`, NVIDIA
NeMo and the `qwen-asr` package. Exact versions are pinned in `requirements.txt`,
`requirements-nemo.txt` and `requirements-qwen.txt`.
