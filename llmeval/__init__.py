"""An evaluation harness built on the premise that most LLM evals do not measure what they
claim to.

Three failures recur, and each has a module here:

1. **Reaching for a judge model too early.** Most regressions are catchable by a deterministic
   assertion that costs nothing and never drifts. :mod:`llmeval.checks` and :mod:`llmeval.pii`
   run first for that reason.
2. **Treating an LLM judge as ground truth.** A judge is an unvalidated classifier with
   documented biases. :mod:`llmeval.judge` debiases position and measures length bias;
   :mod:`llmeval.agreement` scores the judge against human labels before anybody trusts it.
3. **Reading noise as improvement.** "84% versus 81% on 200 prompts" is not a result.
   :mod:`llmeval.stats` supplies the paired test that question actually requires.

Plus :mod:`llmeval.guardrails` for the runtime path, which is a different problem from
evaluation and fails in the opposite direction: an eval that errors should be loud, a guardrail
that errors should block.

The harness is transport-agnostic. A model is any callable from ``str`` to ``str``, so nothing
here makes a network call and the whole suite runs offline.
"""

from .agreement import (
    JudgeValidation,
    cohen_kappa,
    confusion_matrix,
    interpret_kappa,
    percent_agreement,
    validate_judge,
)
from .checks import (
    Check,
    ContainsCheck,
    ExactMatchCheck,
    JsonSchemaCheck,
    MaxLengthCheck,
    NoPIICheck,
    NotContainsCheck,
    NumericToleranceCheck,
    RefusalCheck,
    RegexCheck,
    validate_against_schema,
)
from .guardrails import (
    DenyPatternRule,
    GuardrailDecision,
    GuardrailPipeline,
    GuardrailRule,
    InjectionHeuristicRule,
    MaxLengthRule,
    PIIRedactionRule,
    RuleOutcome,
    SecretLeakRule,
)
from .judge import (
    PairwiseJudge,
    PairwiseVerdict,
    RubricJudge,
    RubricScore,
    length_bias_correlation,
)
from .pii import PII_KINDS, PIIMatch, detect_pii, luhn_valid, redact_pii
from .runner import (
    ComparisonResult,
    ScopedCheck,
    SuiteResult,
    compare_suites,
    run_suite,
    scoped,
)
from .stats import (
    McNemarResult,
    mcnemar,
    paired_bootstrap_ci,
    required_samples_for_difference,
    wilson_interval,
)
from .types import CaseResult, CheckOutcome, Prediction, Sample

__version__ = "1.0.0"

__all__ = [
    "PII_KINDS",
    "CaseResult",
    "Check",
    "CheckOutcome",
    "ComparisonResult",
    "ContainsCheck",
    "DenyPatternRule",
    "ExactMatchCheck",
    "GuardrailDecision",
    "GuardrailPipeline",
    "GuardrailRule",
    "InjectionHeuristicRule",
    "JsonSchemaCheck",
    "JudgeValidation",
    "MaxLengthCheck",
    "MaxLengthRule",
    "McNemarResult",
    "NoPIICheck",
    "NotContainsCheck",
    "NumericToleranceCheck",
    "PIIMatch",
    "PIIRedactionRule",
    "PairwiseJudge",
    "PairwiseVerdict",
    "Prediction",
    "RefusalCheck",
    "RegexCheck",
    "RubricJudge",
    "RubricScore",
    "RuleOutcome",
    "Sample",
    "ScopedCheck",
    "SecretLeakRule",
    "SuiteResult",
    "cohen_kappa",
    "compare_suites",
    "confusion_matrix",
    "detect_pii",
    "interpret_kappa",
    "length_bias_correlation",
    "luhn_valid",
    "mcnemar",
    "paired_bootstrap_ci",
    "percent_agreement",
    "redact_pii",
    "required_samples_for_difference",
    "run_suite",
    "scoped",
    "validate_against_schema",
    "validate_judge",
    "wilson_interval",
]
