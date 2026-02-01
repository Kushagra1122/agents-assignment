"""Unit tests for InterruptionClassifier."""

from __future__ import annotations

import pytest

from livekit.agents.voice.interruption_filter import (
    DEFAULT_IGNORE_WORDS,
    DEFAULT_INTERRUPT_WORDS,
    InterruptionClassifier,
    InterruptionDecision,
    InterruptionFilterConfig,
)


class TestInterruptionClassifier:
    """Tests for the InterruptionClassifier class."""

    @pytest.fixture
    def classifier(self) -> InterruptionClassifier:
        """Create a default classifier instance."""
        return InterruptionClassifier()

    @pytest.fixture
    def custom_classifier(self) -> InterruptionClassifier:
        """Create a classifier with custom word lists."""
        config = InterruptionFilterConfig(
            ignore_words=frozenset({"yeah", "ok", "uh-huh"}),
            interrupt_words=frozenset({"stop", "wait", "no"}),
            grace_period_ms=100,
        )
        return InterruptionClassifier(config)

    # ==================== Agent Speaking + Backchannel Tests ====================

    def test_ignore_backchannel_while_agent_speaking(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Agent speaking + 'yeah' → IGNORE"""
        decision = classifier.classify(transcript="yeah", agent_speaking=True)
        assert decision == InterruptionDecision.IGNORE

    def test_ignore_backchannel_ok_while_agent_speaking(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Agent speaking + 'ok' → IGNORE"""
        decision = classifier.classify(transcript="ok", agent_speaking=True)
        assert decision == InterruptionDecision.IGNORE

    def test_ignore_backchannel_uh_huh_while_agent_speaking(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Agent speaking + 'uh-huh' → IGNORE"""
        decision = classifier.classify(transcript="uh-huh", agent_speaking=True)
        assert decision == InterruptionDecision.IGNORE

    def test_ignore_backchannel_mhm_while_agent_speaking(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Agent speaking + 'mhm' → IGNORE"""
        decision = classifier.classify(transcript="mhm", agent_speaking=True)
        assert decision == InterruptionDecision.IGNORE

    def test_ignore_backchannel_right_while_agent_speaking(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Agent speaking + 'right' → IGNORE"""
        decision = classifier.classify(transcript="right", agent_speaking=True)
        assert decision == InterruptionDecision.IGNORE

    def test_ignore_backchannel_with_mixed_case(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Agent speaking + 'YEAH' (uppercase) → IGNORE"""
        decision = classifier.classify(transcript="YEAH", agent_speaking=True)
        assert decision == InterruptionDecision.IGNORE

    # ==================== Agent Speaking + Interrupt Word Tests ====================

    def test_interrupt_on_stop_word(self, classifier: InterruptionClassifier) -> None:
        """Agent speaking + 'stop' → INTERRUPT"""
        decision = classifier.classify(transcript="stop", agent_speaking=True)
        assert decision == InterruptionDecision.INTERRUPT

    def test_interrupt_on_wait_word(self, classifier: InterruptionClassifier) -> None:
        """Agent speaking + 'wait' → INTERRUPT"""
        decision = classifier.classify(transcript="wait", agent_speaking=True)
        assert decision == InterruptionDecision.INTERRUPT

    def test_interrupt_on_no_word(self, classifier: InterruptionClassifier) -> None:
        """Agent speaking + 'no' → INTERRUPT"""
        decision = classifier.classify(transcript="no", agent_speaking=True)
        assert decision == InterruptionDecision.INTERRUPT

    def test_interrupt_on_hold_on(self, classifier: InterruptionClassifier) -> None:
        """Agent speaking + 'hold on' → INTERRUPT"""
        decision = classifier.classify(transcript="hold on", agent_speaking=True)
        assert decision == InterruptionDecision.INTERRUPT

    def test_interrupt_on_cancel(self, classifier: InterruptionClassifier) -> None:
        """Agent speaking + 'cancel' → INTERRUPT"""
        decision = classifier.classify(transcript="cancel", agent_speaking=True)
        assert decision == InterruptionDecision.INTERRUPT

    # ==================== Agent Silent + Backchannel (Valid Response) ====================

    def test_allow_backchannel_when_agent_silent(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Agent silent + 'yeah' → treated as valid response (INTERRUPT to start turn)"""
        decision = classifier.classify(transcript="yeah", agent_speaking=False)
        # When agent is not speaking, any speech is valid user turn
        assert decision == InterruptionDecision.INTERRUPT

    def test_allow_any_word_when_agent_silent(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Agent silent + 'ok' → treated as valid response"""
        decision = classifier.classify(transcript="ok", agent_speaking=False)
        assert decision == InterruptionDecision.INTERRUPT

    # ==================== Mixed Input Tests ====================

    def test_mixed_input_interrupts(self, classifier: InterruptionClassifier) -> None:
        """Agent speaking + 'yeah wait' → INTERRUPT (interrupt word overrides)"""
        decision = classifier.classify(transcript="yeah wait", agent_speaking=True)
        assert decision == InterruptionDecision.INTERRUPT

    def test_mixed_input_yeah_stop(self, classifier: InterruptionClassifier) -> None:
        """Agent speaking + 'yeah stop' → INTERRUPT"""
        decision = classifier.classify(transcript="yeah stop", agent_speaking=True)
        assert decision == InterruptionDecision.INTERRUPT

    def test_mixed_input_ok_but_wait(self, classifier: InterruptionClassifier) -> None:
        """Agent speaking + 'ok but wait a second' → INTERRUPT"""
        decision = classifier.classify(
            transcript="ok but wait a second", agent_speaking=True
        )
        assert decision == InterruptionDecision.INTERRUPT

    def test_mixed_input_yeah_no(self, classifier: InterruptionClassifier) -> None:
        """Agent speaking + 'yeah no' → INTERRUPT"""
        decision = classifier.classify(transcript="yeah no", agent_speaking=True)
        assert decision == InterruptionDecision.INTERRUPT

    # ==================== Edge Cases ====================

    def test_empty_transcript_returns_pending(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Empty transcript while agent speaking → PENDING (need STT data)"""
        decision = classifier.classify(transcript="", agent_speaking=True)
        assert decision == InterruptionDecision.PENDING

    def test_whitespace_only_returns_pending(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Whitespace-only transcript → PENDING"""
        decision = classifier.classify(transcript="   ", agent_speaking=True)
        assert decision == InterruptionDecision.PENDING

    def test_long_sentence_interrupts(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Long sentence without interrupt words → INTERRUPT (substantive input)"""
        decision = classifier.classify(
            transcript="I have a question about something else",
            agent_speaking=True,
        )
        assert decision == InterruptionDecision.INTERRUPT

    def test_unrecognized_short_word_pending(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Short unrecognized word → PENDING (could be partial)"""
        decision = classifier.classify(transcript="wh", agent_speaking=True)
        assert decision == InterruptionDecision.PENDING

    def test_partial_word_that_could_be_backchannel(
        self, classifier: InterruptionClassifier
    ) -> None:
        """Partial word 'ye' could become 'yeah' or 'yes stop'"""
        decision = classifier.classify(transcript="ye", agent_speaking=True)
        # Should be pending since it's ambiguous
        assert decision == InterruptionDecision.PENDING

    # ==================== classify_final Tests ====================

    def test_classify_final_never_pending(
        self, classifier: InterruptionClassifier
    ) -> None:
        """classify_final should never return 'pending'"""
        # Empty transcript
        decision = classifier.classify_final(transcript="", agent_speaking=True)
        assert decision in ("ignore", "interrupt")

        # Ambiguous short word
        decision = classifier.classify_final(transcript="wh", agent_speaking=True)
        assert decision in ("ignore", "interrupt")

    def test_classify_final_backchannel_ignores(
        self, classifier: InterruptionClassifier
    ) -> None:
        """classify_final with backchannel → ignore"""
        decision = classifier.classify_final(transcript="yeah", agent_speaking=True)
        assert decision == "ignore"

    def test_classify_final_interrupt_word_interrupts(
        self, classifier: InterruptionClassifier
    ) -> None:
        """classify_final with interrupt word → interrupt"""
        decision = classifier.classify_final(transcript="stop", agent_speaking=True)
        assert decision == "interrupt"

    def test_classify_final_empty_ignores(
        self, classifier: InterruptionClassifier
    ) -> None:
        """classify_final with empty transcript → ignore (probably noise)"""
        decision = classifier.classify_final(transcript="", agent_speaking=True)
        assert decision == "ignore"

    # ==================== Configuration Tests ====================

    def test_custom_ignore_words(self) -> None:
        """Custom ignore words should work"""
        config = InterruptionFilterConfig(
            ignore_words=frozenset({"custom-word", "another"}),
            interrupt_words=frozenset({"halt"}),
        )
        classifier = InterruptionClassifier(config)

        # Custom ignore word should be ignored
        decision = classifier.classify(
            transcript="custom-word", agent_speaking=True
        )
        assert decision == InterruptionDecision.IGNORE

        # Default ignore word should NOT be ignored
        decision = classifier.classify(transcript="yeah", agent_speaking=True)
        assert decision == InterruptionDecision.PENDING  # Short but unrecognized

    def test_custom_interrupt_words(self) -> None:
        """Custom interrupt words should work"""
        config = InterruptionFilterConfig(
            ignore_words=frozenset({"yeah"}),
            interrupt_words=frozenset({"halt", "abort"}),
        )
        classifier = InterruptionClassifier(config)

        # Custom interrupt word should interrupt
        decision = classifier.classify(transcript="halt", agent_speaking=True)
        assert decision == InterruptionDecision.INTERRUPT

        # Default interrupt word should NOT interrupt if not in custom list
        decision = classifier.classify(transcript="stop", agent_speaking=True)
        # "stop" is not in custom list, so depends on other logic
        assert decision in (
            InterruptionDecision.PENDING,
            InterruptionDecision.INTERRUPT,
        )

    def test_grace_period_configuration(self) -> None:
        """Grace period should be configurable"""
        config = InterruptionFilterConfig(grace_period_ms=200)
        classifier = InterruptionClassifier(config)
        assert classifier.grace_period_s == 0.2

        config = InterruptionFilterConfig(grace_period_ms=50)
        classifier = InterruptionClassifier(config)
        assert classifier.grace_period_s == 0.05

    # ==================== Default Word Lists Tests ====================

    def test_default_ignore_words_comprehensive(self) -> None:
        """Test that common backchannels are in the default ignore list"""
        expected_ignore = [
            "yeah", "yep", "ok", "okay", "uh-huh", "mhm", "right",
            "sure", "alright", "uh", "um", "hmm", "got it"
        ]
        for word in expected_ignore:
            assert word in DEFAULT_IGNORE_WORDS, f"{word} should be in ignore words"

    def test_default_interrupt_words_comprehensive(self) -> None:
        """Test that common interrupt commands are in the default interrupt list"""
        expected_interrupt = [
            "stop", "wait", "hold on", "no", "cancel", "pause",
            "actually", "but", "however"
        ]
        for word in expected_interrupt:
            assert word in DEFAULT_INTERRUPT_WORDS, f"{word} should be in interrupt words"


# Run with: uv run pytest tests/test_interruption_filter.py -v
