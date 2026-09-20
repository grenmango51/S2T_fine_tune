"""Unit tests for the split/merge-tolerant WER metric."""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from s2t.common import corpus_wer, corpus_wer_splitmerge, _splitmerge_edits, normalize_text


def test_merge_is_forgiven():
    """goodbye vs Good-bye should be forgiven (split/merge only)."""
    ref = ["Goodbye, Pierre, he shouted."]
    hyp = ["Good-bye, Pierre, he shouted."]
    assert corpus_wer(ref, hyp) > 0.0
    assert corpus_wer_splitmerge(ref, hyp) == 0.0


def test_split_is_forgiven():
    """ref 'well-known' normalizes to 'well known'; hyp 'wellknown' is one word."""
    ref = ["He is well-known here."]      # normalizes to 'well known'
    hyp = ["He is wellknown here."]
    assert corpus_wer_splitmerge(ref, hyp) == 0.0


def test_real_error_still_counts():
    """he vs she is a genuine error, not a split/merge."""
    ref = ["Goodbye, Pierre, he shouted."]
    hyp = ["Goodbye, Pierre, she shouted."]
    assert corpus_wer_splitmerge(ref, hyp) == corpus_wer(ref, hyp) > 0.0


def test_near_miss_is_not_forgiven():
    """tapped vs tapered have different letters, not just boundary."""
    ref = ["They tapped the door."]
    hyp = ["They tapered the door."]
    assert corpus_wer_splitmerge(ref, hyp) == corpus_wer(ref, hyp) > 0.0


def test_never_exceeds_standard_wer():
    """The tolerant metric should always be <= standard WER."""
    ref = ["one two three", "well-known man", "goodbye now"]
    hyp = ["one three", "wellknown men", "good bye now please"]
    assert corpus_wer_splitmerge(ref, hyp) <= corpus_wer(ref, hyp)


def test_meaning_changing_boundary_is_still_discounted():
    """'a lone' vs 'alone' is meaning-changing but matches concatenation.
    Document that it IS discounted (a known caveat of the metric)."""
    ref = ["He walked alone."]
    hyp = ["He walked a lone."]
    # After normalization: ref='he walked alone' hyp='he walked a lone'
    nr = normalize_text(ref[0])
    nh = normalize_text(hyp[0])
    std, tol, _ = _splitmerge_edits(nr, nh)
    assert std > 0, "Standard WER should flag this"
    assert tol == 0, "Tolerant metric should discount it (documented caveat)"


def test_repeated_words():
    """Repeated words: 'the the' vs 'thethe' should be forgiven."""
    nr = "the the cat"
    nh = "thethe cat"
    std, tol, n_ref = _splitmerge_edits(nr, nh)
    assert std > 0
    assert tol == 0


def test_splitmerge_adjacent_to_real_error():
    """Split/merge difference adjacent to a real substitution.
    Only the split/merge portion should be forgiven."""
    # 'goodbye' vs 'good bye' is split/merge (forgiven)
    # 'shouted' vs 'whispered' is real error (not forgiven)
    ref = ["Goodbye, Pierre, he shouted."]
    hyp = ["Good-bye, Pierre, he whispered."]
    wer_std = corpus_wer(ref, hyp)
    wer_sm = corpus_wer_splitmerge(ref, hyp)
    assert wer_sm > 0.0, "Real error should still be counted"
    assert wer_sm < wer_std, "Split/merge portion should be forgiven"


def test_empty_strings():
    """Empty/whitespace strings are handled safely."""
    ref = ["Hello world"]
    hyp = [""]
    wer_std = corpus_wer(ref, hyp)
    wer_sm = corpus_wer_splitmerge(ref, hyp)
    # Both should give the same result (no split/merge to discount)
    assert wer_sm == wer_std


def test_identical_strings():
    """Identical strings => 0 WER for both metrics."""
    ref = ["The quick brown fox jumps."]
    hyp = ["The quick brown fox jumps."]
    assert corpus_wer(ref, hyp) == 0.0
    assert corpus_wer_splitmerge(ref, hyp) == 0.0


def test_today_variant():
    """'to-day' normalizes to 'to day'; vs 'today' should be forgiven."""
    ref = ["It happened to-day."]
    hyp = ["It happened today."]
    wer_sm = corpus_wer_splitmerge(ref, hyp)
    # Whether this is forgiven depends on normalization.
    # 'to-day' -> 'to day' (2 words) vs 'today' (1 word)
    # concatenation: 'today' == 'today' => forgiven
    assert wer_sm == 0.0
