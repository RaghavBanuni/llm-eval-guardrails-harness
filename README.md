# LLM Eval and Guardrails Harness

An evaluation harness built around three claims: most LLM evals reach for a judge model long
before they need one, trust that judge without ever measuring it, and then read noise as
improvement. Each has a module here, and each argument is demonstrated rather than asserted.

Runs entirely offline. A model is any callable from `str` to `str` — no network calls, no API
keys, and the demos print the same numbers on your machine as in this README.

## 1. Write the boring checks first

An LLM judge costs a network call, thousands of tokens per case, several seconds, and a
non-deterministic answer that drifts when the judge model is upgraded underneath you. Before its
output means anything it has to be validated against human labels.

A deterministic check costs microseconds, gives the same answer forever, and needs no validation
because it *is* the specification.

The great majority of production regressions are the second kind:

| regression | catchable by |
|---|---|
| malformed JSON, dropped required field, changed field type | schema check |
| system prompt or canary string leaked into output | substring check |
| refusal on a benign request | refusal check, `expect_refusal=False` |
| PII echoed into a reply | PII check |
| response length tripled after a prompt change | length check, advisory |
| provider timeout counted as a pass | error accounting in the runner |

```bash
python -m llmeval.cli checks
```

That runs a support-ticket extraction suite against a baseline and a candidate release. The
candidate drops a required field twice, changes a field's type once, times out once, refuses a
benign request, and echoes an email and a card number — while genuinely fixing an arithmetic
case. Every failure is named with its reason, and none of it needed a judge.

Two details in that output are deliberate and worth pausing on.

**Valid JSON is not the property that matters.** One regression returns `{"urgency": "low"}` —
perfectly parseable, and broken for every downstream consumer expecting an integer. A parse check
passes it; a schema check does not.

**A timeout is a result.** The runner records failed calls and counts them as failures. The
tempting alternative — skip them, compute the rate over what is left — reports a pass rate
measured on a biased subset, because the prompts that time out are the long and adversarial ones.
A pass rate that silently excludes the hardest 5% of the suite is worse than no pass rate.

## 2. A judge is a classifier. Measure it.

Nobody would ship a classifier without checking it against labels. Judges get treated as ground
truth constantly, because their output *looks* like an assessment rather than a prediction.

### Position bias

Present two answers as A and B and many judges favour a slot regardless of content. The fix is
cheap: run the comparison twice with the answers swapped, and count a win only if both orderings
agree. Disagreement is not noise to average away — it is the judge reporting that it has no real
preference, so it is recorded as a tie and flagged.

`PairwiseJudge` always runs both orders. There is no option to run one, because a single-order
comparison is not worth reporting.

### Verbosity bias

Judges reward length independently of quality. Prompting does not fix it, but
`length_bias_correlation` measures it: rank-correlate scores against answer length. On extraction
or arithmetic — where length should be irrelevant — a high correlation means the scores are partly
measuring word count. On a task where thoroughness genuinely is quality, the same number is fine.
It is a measurement to interpret, not a threshold to enforce.

### Self-preference bias

Models rate their own output higher. The mitigation is not to judge a model with itself, which is
a deployment choice rather than something code can enforce — documented here rather than silently
unhandled.

### Parse failures are never scores

When a judge returns something unparseable, `RubricScore.value` is `None` and `parse_failed` is
`True`. It does not fall back to a midpoint. Substituting 3-out-of-5 for *the judge did not
answer* contaminates the aggregate with fabricated data, and does it invisibly: the mean drifts
towards the middle and nothing in the report says why. A score outside the scale is treated the
same way rather than clamped — a judge answering 8 on a 1-5 scale has misunderstood the task, and
clamping to 5 hides that.

```bash
python -m llmeval.cli judge
```

### Then validate it against humans

Suppose 90% of your cases are passes and the judge says pass to everything. Raw agreement with
human labels: 90%. That number goes on a slide. The judge detects zero failures — the only cases
anyone cares about.

Cohen's kappa subtracts the agreement the marginals alone would produce:

```
kappa = (p_observed - p_expected) / (1 - p_expected)
```

The always-pass judge scores **exactly zero**. That is why kappa is the headline number here and
raw agreement is reported next to it as context.

```bash
python -m llmeval.cli validate
```

The interpretation bands (slight / fair / moderate / substantial / almost perfect) are Landis and
Koch's 1977 convention for medical diagnosis agreement. They have no special authority over LLM
evaluation, and they are labelled as convention here because pretending otherwise would repeat
the original mistake.

One thing kappa cannot check: whether your human labels are any good. A high kappa against
inconsistent labels means the judge faithfully reproduces the inconsistency. Measure
inter-annotator agreement first — with the same function — before treating human labels as ground
truth either.

## 3. "84% versus 81%" is not a result

It is two numbers with no statement about whether the difference would survive a different 200
prompts. Usually it would not.

Both systems ran on the *same* prompts, so the samples are paired, and McNemar's test is the one
that exploits it. Its insight: prompts both systems passed, and prompts both failed, carry **no
information about which is better**. Only the disagreements do.

```
                B passed   B failed
  A passed         158        10       <- only these 14 count
  A failed           4        28
```

84% versus 81%, and fourteen informative pairs. Exact binomial p = 0.18. Not distinguishable.

The same test on a candidate that really did regress (38 flips one way, 2 the other) returns
p < 1e-8 and blocks it. A gate that never rejects anything would be useless in the other
direction.

```bash
python -m llmeval.cli compare
```

The exact binomial test is used rather than the chi-square approximation with continuity
correction, because eval suites routinely produce fewer than 25 discordant pairs — precisely where
that approximation is unreliable.

Every pass rate is reported with a **Wilson** interval, not the textbook normal approximation. At
40/40 the normal interval has zero width — perfect certainty from forty prompts. Wilson keeps a
lower bound near 91%, which is the honest reading.

### How big does the suite need to be?

```bash
python -m llmeval.cli power
```

From an 80% baseline, at 80% power and alpha = 0.05:

| difference to detect | prompts needed |
|---|---|
| 15 points | ~66 |
| 10 points | ~250 |
| 5 points | ~906 |
| 2 points | ~6,038 |

These are conservative — the unpaired calculation, so a paired suite needs somewhat fewer. Even
so, the differences teams argue over hardest need thousands of prompts, and most suites are built
at 100-200 and then asked to adjudicate a two-point change.

The honest consequence shows up in the `checks` demo: six regressions against one fix is seven
discordant pairs, and seven cannot clear alpha = 0.05. The gate says PASS on a candidate that
visibly broke five things. That is not a flaw in the test — it is a ten-prompt suite failing to
carry evidence, and this repository asserts it in a test rather than hiding it. The cheaper way
out is not always a bigger suite; it is checks with less noise in them. A schema violation is a
certainty, not a 51/49 preference, so it needs no statistical power at all.

## 4. Guardrails fail in the opposite direction

An **eval** that crashes should be loud: raise, fail the run, block the release.

A **guardrail** that crashes should block the *request*. It sits in the live path, and the two
available behaviours are: refuse this one request, or serve it with the safety control switched
off. Only the first is defensible, and only the second is what a bare `except: pass` produces.

`GuardrailPipeline` is fail-closed by default — a rule that raises is recorded and the request is
blocked. `fail_closed=False` exists because some rules genuinely are advisory, but it has to be
chosen explicitly.

Three actions, because unsafe is not one thing:

- **block** — reject the request. Credential-shaped tokens in output block rather than redact,
  because masking a leaked key hides the upstream defect that read it into context.
- **redact** — transform and continue. Removing a leaked email usually serves the user better than
  refusing to answer.
- **flag** — record and continue. Where the detector is a heuristic, this is the right default.

```bash
python -m llmeval.cli guardrails
```

### On prompt injection

`InjectionHeuristicRule` catches crude, high-volume attempts and defaults to `flag`, not `block`.
It does not solve prompt injection, and nothing at this layer does. Injection follows from
instructions and data sharing one channel; a paraphrase, a translation, a base64 blob or an
instruction inside a retrieved document defeats any keyword list. Treat it as noise reduction on
top of architectural controls — least privilege on tools, no untrusted content in privileged
context, confirmation on consequential actions — not as a substitute for them. Anything sold as
solving injection by pattern matching is selling a feeling.

### PII detection that does not cry wolf

A detector that fires on every 16-digit number trains reviewers to ignore it, and once alerts are
ignored the real leak passes through with everything else. So:

- **Card numbers are Luhn-validated** (ISO/IEC 7812). Roughly 90% of random 16-digit strings fail
  the check digit, so order ids and tracking numbers stop being reported as cards — a ten-fold
  false-positive reduction for fifteen lines of arithmetic. A test asserts the rate.
- **Phone numbers are capped at 15 characters** (the E.164 maximum). Without that cap a 16-digit
  order reference matches as a phone number, and the specificity won on cards is handed straight
  back on phones.
- **SSNs exclude the ranges the SSA never issues** (`000`, `666` and `9xx` areas, `00` group,
  `0000` serial).
- **Trailing punctuation does not defeat detection.** A naive `(?![\w.])` guard misses a card at
  the end of a sentence — the most natural way for a model to write one. There is a regression
  test for exactly that.
- **Overlaps resolve to one label**, so a card number is not also reported as a phone number.

It is still regex plus arithmetic. It misses PII written in words, PII split across turns, names,
addresses, and anything deliberately obfuscated. The credential detector trades precision for
recall on purpose. Detection reduces accidental leaks; it does not make a system safe to hand
untrusted data, and on a regulated surface this belongs behind a real DLP system rather than in
front of one.

## Install

```bash
git clone https://github.com/RaghavBanuni/llm-eval-guardrails-harness.git
cd llm-eval-guardrails-harness
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. numpy and scipy; pytest for the tests. Nothing else.

## Use

```python
from llmeval import JsonSchemaCheck, NoPIICheck, RefusalCheck, compare_suites, run_suite, scoped
from llmeval.types import Sample

samples = [
    Sample('tk-1', "Extract fields: 'charged twice'", tags=('json',)),
    Sample('rf-1', 'Summarise the refund policy', tags=('benign',)),
]

schema = {'type': 'object', 'required': ['category', 'urgency']}
checks = [
    scoped(JsonSchemaCheck(schema, name='schema'), 'json'),
    scoped(RefusalCheck(expect_refusal=False, name='no_false_refusal'), 'benign'),
    NoPIICheck(name='no_pii'),            # unscoped: applies everywhere
]

baseline = run_suite(call_model_v1, samples, checks, name='v1')
candidate = run_suite(call_model_v2, samples, checks, name='v2')

print(baseline.summary())                 # pass rate with a Wilson interval
print(candidate.failures_by_check())      # the triage view
print(compare_suites(baseline, candidate).summary())    # paired test, and a ship/block verdict
```

Scoping is a property of the suite, not of the check: `scoped(check, 'json')` wraps a check with a
tag filter, so check classes stay pure predicates on `(output, sample)`. A sample that ends up
with no applicable check **fails** with `no_applicable_check` rather than passing — an unevaluated
case counted as a pass inflates the rate with cases nobody checked.

Runtime guardrails:

```python
from llmeval import GuardrailPipeline, MaxLengthRule, PIIRedactionRule, SecretLeakRule

inbound = GuardrailPipeline([MaxLengthRule('len', max_chars=2000)])
outbound = GuardrailPipeline([SecretLeakRule(), PIIRedactionRule()])

if not inbound.run(user_text, stage='input').allowed:
    return refusal()

decision = outbound.run(model_output, stage='output')
return decision.text if decision.allowed else refusal()
```

## Tests

```bash
pytest -q
```

The suite asserts properties and arguments, not snapshots: that Wilson stays sensible at 0/n and
n/n and covers the true rate at the advertised 95%; that McNemar's p-value is unchanged when the
concordant cells are inflated ninety-fold; that a three-point gap at n=200 is not significant
while a real regression at the same size is; that pairing tightens a bootstrap interval more than
fivefold; that an always-pass judge scores 90% agreement and exactly zero kappa; that a
first-slot *and* a second-slot biased judge both come back as ties with the bias flagged; that a
swapped comparison is relabelled correctly in both directions; that an unparseable judge reply is
`None` and not a midpoint; that a Luhn-failing 16-digit id is not reported as a card while a card
followed by a full stop is; that a crashing guardrail blocks; that a blocking rule short-circuits
later rules while a redaction rule passes transformed text downstream; and that the
demonstrations print the findings the sections above rest on.

## Layout

```
llmeval/
  types.py        Sample, Prediction, CheckOutcome, CaseResult
  checks.py       deterministic checks + a small dependency-free schema validator
  pii.py          Luhn-validated card detection, redaction, specificity notes
  judge.py        position-debiased pairwise judging, rubric scoring, length-bias measurement
  agreement.py    Cohen's kappa, judge validation against human labels
  stats.py        Wilson intervals, exact McNemar, paired bootstrap, power
  runner.py       suite execution, tag scoping, error accounting, paired comparison
  guardrails.py   fail-closed runtime pipeline: block / redact / flag
  demo.py         deterministic offline fixtures
  cli.py          checks / compare / power / judge / validate / guardrails
tests/            property and argument tests, including the demo findings
```

## References

- Wilson, E.B. (1927). *Probable inference, the law of succession, and statistical inference.*
- Cohen, J. (1960). *A coefficient of agreement for nominal scales.*
- McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions.*
- Landis, J.R., Koch, G.G. (1977). *The measurement of observer agreement for categorical data.*
- Zheng, L. et al. (2023). *Judging LLM-as-a-judge with MT-Bench and Chatbot Arena.*
- ISO/IEC 7812-1: *Identification cards - numbering system and registration procedure.*
- ITU-T E.164: *The international public telecommunication numbering plan.*

## Licence

MIT — see [LICENSE](LICENSE).
