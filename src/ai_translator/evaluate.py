"""
BLEU evaluation — extracted from the original app.py.

Both app.py and the notebook import from here.
"""

from typing import Tuple
import sacrebleu


def calculate_bleu(
    reference: str,
    hypothesis: str,
    smooth_method: str = "exp",
) -> Tuple[float, str]:
    """
    Compute sentence-level BLEU between a reference and a hypothesis.

    Args:
        reference:     Ground-truth translation.
        hypothesis:    Model-generated translation.
        smooth_method: sacrebleu smoothing ("exp", "floor", "add-k").

    Returns:
        (score, report) — score is 0–100, report is a human-readable string
        with precision breakdown and brevity penalty.

    Example:
        >>> score, report = calculate_bleu("Le chat", "Le chat")
        >>> score
        100.0
    """
    if not reference or not hypothesis:
        return 0.0, "Both reference and hypothesis must be provided"

    try:
        bleu  = sacrebleu.sentence_bleu(hypothesis, [reference], smooth_method=smooth_method)
        score = bleu.score

        if score >= 60:
            quality = "Excellent"
        elif score >= 40:
            quality = "Good"
        elif score >= 20:
            quality = "Fair"
        else:
            quality = "Poor"

        report = (
            f"\n📊 BLEU Evaluation Results\n"
            f"BLEU Score : {score:.2f} / 100\n"
            f"Quality    : {quality}\n\n"
            f"Reference  : {reference}\n"
            f"Hypothesis : {hypothesis}\n\n"
            f"Precision Scores:\n"
            f"  1-gram : {bleu.precisions[0]:.2f}%\n"
            f"  2-gram : {bleu.precisions[1]:.2f}%\n"
            f"  3-gram : {bleu.precisions[2]:.2f}%\n"
            f"  4-gram : {bleu.precisions[3]:.2f}%\n\n"
            f"Brevity Penalty : {bleu.bp:.3f}\n"
        )
        return score, report

    except Exception as exc:
        return 0.0, f"Error calculating BLEU: {exc}"


def corpus_bleu(references: list[str], hypotheses: list[str]) -> float:
    """
    Compute corpus-level BLEU over parallel sentence lists.

    Args:
        references:  List of ground-truth translations.
        hypotheses:  List of model-generated translations.

    Returns:
        Corpus BLEU score (0–100).
    """
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")

    return sacrebleu.corpus_bleu(hypotheses, [references]).score
