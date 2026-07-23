# Model Evaluation, Factuality, and Reliability

This section treats reliability as an end-to-end system property rather than a
single benchmark score. A fluent response can be false, unsupported by its
sources, overconfident, inconsistent with a tool result, structurally invalid,
or correct for the wrong task. Each failure needs a different reference,
detector, and control.

Use this dependency-ordered reading path:

1. [LLM hallucination and mitigation map](hallucination.md) defines the failure
   surfaces, causes, and complete control stack.
2. [Disclosed vendor practices](hallucination-vendors.md) separates public
   evidence about real models and products from generic architecture advice and
   undisclosed production details.
3. [Evaluation and production operations](hallucination-operations.md) turns
   the method map into a claim-level architecture, test suite, telemetry schema,
   and release gate.
4. [Instruction following and steerability](instruction-following.md) covers
   constraint retention, role hierarchy, tool-policy compliance, and
   multi-turn reliability. Those failures can coexist with perfect factuality.
5. [Instruction-following improvement methods](instruction-following-methods.md)
   spans data construction, SFT, preference/RL, hierarchy training, tool
   trajectories, constrained decoding, verification, memory, and product
   controls.
6. [Disclosed instruction-following vendor
   practices](instruction-following-vendors.md) audits named models and
   products without inferring proprietary training from an API feature.
7. [Instruction-following production
   operations](instruction-following-operations.md) defines a versioned
   contract, rule-level evaluation, release gates, telemetry, and incident
   response.

The repository-wide [research standard](../research-method.md) defines the
evidence labels used throughout:

- **D — disclosed:** a primary source states the practice;
- **C — confirmed artifact:** released code, configuration, data, or weights
  directly establish it;
- **R — reproduced:** RoseLLM retained a reproducible run and artifacts;
- **I — inferred:** a conclusion follows from named evidence and assumptions;
- **U — unknown:** public evidence is insufficient.

## Reliability surfaces that must remain separate

| Surface | Reference used to judge it | Representative failure |
|---|---|---|
| **Factuality** | the best available external evidence about the world | a real person's birth date is invented |
| **Faithfulness / grounding** | the supplied document, image, database result, or tool observation | a summary adds a claim absent from the source |
| **Attribution** | the cited source span and its provenance | a real link is attached to a claim it does not support |
| **Calibration** | empirical correctness at each confidence or abstention level | a wrong answer is stated with unjustified certainty |
| **Reasoning correctness** | mathematical, logical, executable, or domain rules | the cited premises are correct but the conclusion does not follow |
| **Tool integrity** | the actual call, result, version, and environment state | the model narrates a search result or test run that never occurred |
| **Instruction following** | the authorized instruction hierarchy and task contract | an answer is true but violates a required format or prohibition |
| **Safety and authorization** | policy, permission, and state-transition rules | a factually correct action is unauthorized or harmful |
| **Usefulness and coverage** | the user's decision need and required claim set | every sentence is true but the decisive caveat is omitted |

“Hallucination” is therefore useful only after the failed reference is named.
The detailed chapters use more precise terms such as unsupported claim,
contradicted claim, fabricated citation, retrieval failure, tool-result
fabrication, visual grounding error, and uncalibrated guess.

## The minimum evidence bundle for a reliability claim

A claim that a model or system “reduces hallucinations” is incomplete unless it
identifies:

1. the model checkpoint, product surface, tools, retriever, and date;
2. the target failure surface and the authoritative reference;
3. answerable and unanswerable examples, including false premises;
4. the sampling, reasoning, context, and tool budget;
5. whether abstentions count as failures, partial credit, or a separate outcome;
6. claim-level correctness, citation support, source quality, and coverage;
7. repeated-run variation and confidence intervals;
8. the judge or verifier, its calibration, and independent human audit; and
9. the cost, latency, coverage, and failure-severity trade-off.

This prevents a lower error rate obtained by refusing every question, a higher
accuracy obtained by guessing more often, or a citation score obtained by
attaching irrelevant links from being mistaken for a generally more reliable
system.
