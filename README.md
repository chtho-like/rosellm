# RoseLLM

RoseLLM is both an implementation lab and a rigorous, source-cited learning
repository for large language models.

The codebase is organized around three complementary goals:

1. **Learn the stack from first principles.** The documentation starts with the
   required mathematics and builds toward modern pretraining, post-training,
   agentic reinforcement learning, distributed training, and inference systems.
2. **Read executable implementations.** Minimal PyTorch, Triton, and CUDA
   implementations expose the mechanics that production frameworks often hide.
3. **Connect papers to real systems.** Research notes identify exactly what a
   paper or model developer disclosed, what can be reproduced, what is a
   defensible inference, and what remains unknown.

## Start here

- [LLM learning roadmap](docs/learning-roadmap.md)
- [Agentic RL: zero-to-researcher curriculum](docs/agentic-rl/index.md)
- [Frontier-lab and open-industry evidence matrix](docs/agentic-rl/case-studies/index.md)
- [Research and citation standard](docs/research-method.md)
- [Training implementation](rosellm/rosetrainer/)
- [Inference implementation](rosellm/roseinfer/)
- [Kernel notebooks](notebooks/cuda/)

The first long-form curriculum is **Agentic Reinforcement Learning**. It covers
the formal POMDP view, policy-gradient mathematics, data and environment
construction, verifiable and learned rewards, multi-turn credit assignment,
rollout/training infrastructure, evaluation, safety, and evidence-backed case
studies of frontier model families.

## Documentation

The repository language is English. Technical claims should cite primary
sources whenever a primary source exists.

```bash
python -m pip install -e '.[dev]'
mkdocs serve
```

The documentation site is configured with MkDocs Material. Every guide is also
readable directly on GitHub.

## Package status

RoseLLM is an experimental learning and research project. Its APIs may change
while the training and inference stacks are being developed.

## License

[MIT](LICENSE)
