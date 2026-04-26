"""Unit tests for the article scoring logic."""
from __future__ import annotations

import math

import pytest

from src.scoring import (
    contains_any, count_bigrams, count_matches, context_score,
    lm_sentiment_score, score_article,
)


class TestContainsAny:
    def test_match_present(self):
        assert contains_any("κρίση χρέους", ("κρίσ",))

    def test_match_absent(self):
        assert not contains_any("ηρεμία", ("κρίσ",))

    def test_case_insensitive(self):
        assert contains_any("ΚΡΙΣΗ", ("κρίσ",)) is False  # Greek case is non-trivial
        assert contains_any("ΚΡΙΣΗ".lower(), ("κρισ",))


class TestCountMatches:
    def test_multiple_occurrences(self):
        assert count_matches("κρίση κρίση κρίση", ("κρίσ",)) == 3

    def test_no_match(self):
        assert count_matches("ηρεμία", ("κρίσ",)) == 0


class TestCountBigrams:
    def test_present(self):
        assert count_bigrams("η οικονομική κρίση είναι βαθιά", ("οικονομική κρίση",)) == 1

    def test_absent(self):
        assert count_bigrams("τίποτα σχετικό", ("οικονομική κρίση",)) == 0


class TestContextScore:
    def test_baseline_one_keyword(self):
        eff, total, neg, dim, amp = context_score(
            "η αβεβαιότητα στην οικονομία αυξάνεται διαρκώς",
        )
        # "αυξάν" is an amplifier within window of "αβεβαι"
        assert total == 1
        assert eff == 2.0
        assert amp == 1

    def test_negation_suppresses(self):
        eff, total, neg, dim, amp = context_score(
            "δεν υπάρχει αβεβαιότητα στις αγορές",
        )
        assert total == 1
        assert neg == 1
        assert eff == 0.0

    def test_diminisher_suppresses(self):
        eff, total, neg, dim, amp = context_score(
            "η κρίση τελείωσε επιτέλους",
        )
        assert total == 1
        assert dim == 1
        assert eff == 0.0

    def test_negated_diminisher_restores(self):
        eff, total, neg, dim, amp = context_score(
            "η κρίση δεν τελείωσε ακόμη",
        )
        assert total == 1
        assert eff == 1.5

    def test_no_uncertainty_keyword(self):
        eff, total, *_ = context_score("όλα ήρεμα στις αγορές")
        assert total == 0
        assert eff == 0.0


class TestLMSentiment:
    def test_purely_positive(self):
        s = lm_sentiment_score("ανάπτυξη ανάπτυξη ανάπτυξη")
        assert s == 1.0

    def test_purely_negative(self):
        s = lm_sentiment_score("καταστροφή καταστροφή")
        assert s == -1.0

    def test_balanced(self):
        s = lm_sentiment_score("ανάπτυξη καταστροφή")
        assert math.isclose(s, 0.0, abs_tol=1e-9)

    def test_no_terms(self):
        assert lm_sentiment_score("ουδέτερο κείμενο") == 0.0


class TestScoreArticle:
    def test_short_content_returns_empty(self):
        s = score_article("τιπ", "τίτλος")
        assert not s.is_epu_content
        assert s.context_score == 0.0

    def test_full_epu_article(self):
        title = "Πολιτική αβεβαιότητα στην ελληνική οικονομία"
        content = (
            "Η κυβέρνηση αντιμετωπίζει αβεβαιότητα στην οικονομία "
            "λόγω της κρίσης χρέους και των επικείμενων εκλογών. "
            "Η αστάθεια στις αγορές κλιμακώνεται." * 3
        )
        s = score_article(content, title)
        assert s.is_epu_content
        assert s.is_epu_title
        assert s.context_score > 0

    def test_missing_one_category_excluded(self):
        # Only economy keywords, no policy or uncertainty
        content = "Η οικονομία και η τράπεζα είχαν καλή χρονιά." * 3
        s = score_article(content, None)
        assert not s.is_epu_content
