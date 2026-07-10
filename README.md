# rl-agent-arcade

> A Proximal Policy Optimization (PPO) agent **implemented from scratch** that
> learns to play Snake — trained, evaluated, and served as a live demo.

🚧 **Work in progress.** The custom Snake environment and its test suite are done
and passing; the from-scratch PPO trainer, evaluation, GIFs and demo are being
built next. See the development plan below.

## Why this project

It is meant to demonstrate three things:

1. **The maths behind RL** — PPO is written by hand (GAE, clipped surrogate
   objective, entropy bonus), not called from a library.
2. **A full ML pipeline** — custom environment, training, monitoring,
   evaluation, and deployment.
3. **Communication** — clear docs, reproducible commands, and metrics.

## Status

| Component | State |
| --- | --- |
| Custom Snake environment (Gymnasium API) | ✅ done |
| Unit tests | ✅ 12 passing |
| PPO from scratch | ⏳ in progress |
| Stable-Baselines3 baseline | ⏳ |
| Training + TensorBoard monitoring | ⏳ |
| Progression GIFs (random → mid → expert) | ⏳ |
| Live demo (Gradio / HF Spaces) | ⏳ |

## Quickstart

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt
pytest -q                                           # run the env test suite
```

## Repository layout

```
rl-agent-arcade/
├── envs/           # custom Snake environment (Gymnasium API)
├── agents/         # PPO from scratch, networks, SB3 baseline
├── training/       # training loop + YAML configs
├── evaluation/     # quantitative eval + GIF recording
├── demo/           # Gradio app
├── tests/          # unit tests for the environment
└── media/          # progression GIFs
```

## License

MIT
