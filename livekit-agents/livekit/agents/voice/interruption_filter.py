"""Smart interruption filtering for LiveKit voice agents.

This module provides an InterruptionClassifier that determines whether user speech
should interrupt the agent or be treated as a backchannel acknowledgment.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Literal


@unique
class InterruptionDecision(str, Enum):
    """Decision from the interruption classifier."""

    IGNORE = "ignore"
    """Backchannel - agent should continue speaking."""

    INTERRUPT = "interrupt"
    """Real interruption - agent should stop immediately."""

    PENDING = "pending"
    """Not enough information yet - wait for more STT data."""


# Default word sets - can be overridden via config or environment
DEFAULT_IGNORE_WORDS: frozenset[str] = frozenset({
    # English backchannels
    "yeah", "yep", "yes", "yup", "ya",
    "ok", "okay", "k",
    "uh-huh", "uh huh", "uhuh", "mhm", "mm-hmm", "mmhmm", "hmm", "hm",
    "right", "sure", "alright", "fine",
    "uh", "um", "ah", "oh",
    "got it", "gotcha", "i see", "see",
    "cool", "nice", "great", "good",
    "go on", "go ahead", "continue",
})

DEFAULT_INTERRUPT_WORDS: frozenset[str] = frozenset({
    # Explicit stop commands
    "stop", "wait", "hold on", "hold",
    "no", "nope", "don't", "dont",
    "cancel", "pause", "quiet", "enough",
    "shut up", "be quiet", "silence",
    "actually", "but", "however", "although",
    "hang on", "one moment", "just a moment", "one sec", "one second",
    "let me", "i want", "i need", "can you", "could you",
    "excuse me", "sorry but", "pardon",
    "that's wrong", "thats wrong", "not right", "incorrect",
})


def _load_words_from_env(env_var: str, default: frozenset[str]) -> frozenset[str]:
    """Load word set from environment variable (comma-separated) or use default."""
    env_value = os.environ.get(env_var)
    if env_value:
        words = {w.strip().lower() for w in env_value.split(",") if w.strip()}
        return frozenset(words)
    return default


@dataclass
class InterruptionFilterConfig:
    """Configuration for the InterruptionClassifier.

    Attributes:
        ignore_words: Words to treat as backchannels when agent is speaking.
        interrupt_words: Words that always trigger an interruption.
        grace_period_ms: Time to wait for STT before making a decision (milliseconds).
        require_transcript_for_interrupt: If True, never interrupt without STT text.
        min_word_match_ratio: Minimum ratio of transcript matching ignore words to ignore.
        rapid_backchannel_threshold: Number of backchannels in window to treat as interruption.
        rapid_backchannel_window_s: Time window for rapid backchannel detection (seconds).
    """

    ignore_words: frozenset[str] = field(
        default_factory=lambda: _load_words_from_env(
            "LIVEKIT_IGNORE_WORDS", DEFAULT_IGNORE_WORDS
        )
    )
    interrupt_words: frozenset[str] = field(
        default_factory=lambda: _load_words_from_env(
            "LIVEKIT_INTERRUPT_WORDS", DEFAULT_INTERRUPT_WORDS
        )
    )
    grace_period_ms: int = field(
        default_factory=lambda: int(
            os.environ.get("LIVEKIT_INTERRUPTION_GRACE_PERIOD_MS", "150")
        )
    )
    require_transcript_for_interrupt: bool = False
    min_word_match_ratio: float = 0.8
    rapid_backchannel_threshold: int = 3
    rapid_backchannel_window_s: float = 2.0


class InterruptionClassifier:
    """Stateless classifier for determining interruption vs backchannel.

    This classifier examines the STT transcript and agent state to decide
    whether user speech should interrupt the agent or be ignored.

    Usage:
        classifier = InterruptionClassifier(config)
        decision = classifier.classify(transcript="yeah", agent_speaking=True)
        if decision == InterruptionDecision.IGNORE:
            # Continue agent speech
            pass
        elif decision == InterruptionDecision.INTERRUPT:
            # Stop agent immediately
            pass
    """

    def __init__(self, config: InterruptionFilterConfig | None = None) -> None:
        self._config = config or InterruptionFilterConfig()
        # Compile patterns for multi-word matching
        self._ignore_patterns = self._compile_patterns(self._config.ignore_words)
        self._interrupt_patterns = self._compile_patterns(self._config.interrupt_words)

    @property
    def config(self) -> InterruptionFilterConfig:
        return self._config

    @property
    def grace_period_s(self) -> float:
        """Grace period in seconds."""
        return self._config.grace_period_ms / 1000.0

    def _compile_patterns(self, words: frozenset[str]) -> list[re.Pattern[str]]:
        """Compile word set into regex patterns for matching."""
        patterns = []
        for word in sorted(words, key=len, reverse=True):  # Match longer phrases first
            # Word boundary matching, case insensitive
            pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
            patterns.append(pattern)
        return patterns

    def _normalize_transcript(self, transcript: str) -> str:
        """Normalize transcript for matching."""
        # Lowercase and strip whitespace
        text = transcript.lower().strip()
        # Remove trailing punctuation (Deepgram often adds periods)
        text = text.rstrip(".,!?;:")
        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text

    def _contains_interrupt_word(self, transcript: str) -> tuple[bool, str | None]:
        """Check if transcript contains any interrupt word.

        Returns:
            Tuple of (found, matched_word)
        """
        normalized = self._normalize_transcript(transcript)
        for pattern in self._interrupt_patterns:
            match = pattern.search(normalized)
            if match:
                return True, match.group(0)
        return False, None

    def _is_pure_backchannel(self, transcript: str) -> tuple[bool, str | None]:
        """Check if transcript is purely a backchannel response.

        Returns:
            Tuple of (is_backchannel, matched_word)
        """
        normalized = self._normalize_transcript(transcript)
        if not normalized:
            return False, None

        # Split into words for analysis
        words = normalized.split()

        # Check if the entire transcript matches an ignore phrase
        for pattern in self._ignore_patterns:
            if pattern.fullmatch(normalized):
                return True, normalized

        # Check if all words are ignore words (for single words or short phrases)
        if len(words) <= 3:
            # Strip punctuation from each word before matching
            def normalize_word(w: str) -> str:
                return w.strip(".,!?;:")
            
            words_in_ignore = sum(
                1 for w in words
                if any(p.fullmatch(normalize_word(w)) for p in self._ignore_patterns)
            )
            if words_in_ignore / len(words) >= self._config.min_word_match_ratio:
                return True, normalized

        return False, None

    def classify(
        self,
        transcript: str,
        agent_speaking: bool,
        *,
        speech_duration_s: float | None = None,
    ) -> InterruptionDecision:
        """Classify whether user speech should interrupt the agent.

        Args:
            transcript: Current STT transcript (may be partial/interim).
            agent_speaking: Whether the agent is currently speaking.
            speech_duration_s: Duration of user speech in seconds (optional).

        Returns:
            InterruptionDecision indicating whether to IGNORE, INTERRUPT, or wait (PENDING).
        """
        # If agent is not speaking, any speech is a valid user turn
        if not agent_speaking:
            # Let normal turn detection handle it
            return InterruptionDecision.INTERRUPT

        transcript = transcript.strip()

        # No transcript yet - decision depends on config
        if not transcript:
            if self._config.require_transcript_for_interrupt:
                return InterruptionDecision.PENDING
            # Without transcript, we can't classify - wait for STT
            return InterruptionDecision.PENDING

        # Check for explicit interrupt words first (highest priority)
        has_interrupt, matched_interrupt = self._contains_interrupt_word(transcript)
        if has_interrupt:
            return InterruptionDecision.INTERRUPT

        # Check if it's a pure backchannel
        is_backchannel, matched_backchannel = self._is_pure_backchannel(transcript)
        if is_backchannel:
            return InterruptionDecision.IGNORE

        # For longer utterances not matching either category, assume interruption
        # (User is trying to say something substantive)
        words = transcript.split()
        if len(words) >= 3:
            return InterruptionDecision.INTERRUPT

        # Short but unrecognized - could be partial, wait for more
        return InterruptionDecision.PENDING

    def classify_final(
        self,
        transcript: str,
        agent_speaking: bool,
    ) -> Literal["ignore", "interrupt"]:
        """Make a final classification decision (no PENDING allowed).

        Used when grace period expires and we must make a decision.

        Args:
            transcript: Final STT transcript.
            agent_speaking: Whether the agent is currently speaking.

        Returns:
            Either "ignore" or "interrupt" (never "pending").
        """
        decision = self.classify(transcript, agent_speaking)

        if decision == InterruptionDecision.PENDING:
            # For PENDING decisions, apply smarter heuristics:
            transcript = transcript.strip()
            if not transcript:
                # No transcript at all - likely just noise, ignore
                return "ignore"
            
            # Check if ALL words look like backchannels (even if not in exact list)
            words = transcript.lower().split()
            
            # Strip punctuation from each word
            def clean_word(w: str) -> str:
                return w.strip(".,!?;:")
            
            cleaned_words = [clean_word(w) for w in words]
            
            # Check if all words match ignore patterns
            all_backchannel = all(
                any(p.fullmatch(w) for p in self._ignore_patterns)
                for w in cleaned_words if w  # Skip empty strings
            )
            
            if all_backchannel:
                return "ignore"
            
            # If short utterance (1-2 words) and no interrupt word found, ignore
            # This handles cases like "uh huh" or "mm hmm" that may not be in list
            if len(cleaned_words) <= 2:
                has_interrupt, _ = self._contains_interrupt_word(transcript)
                if not has_interrupt:
                    return "ignore"
            
            # Default: treat as interruption
            return "interrupt"

        return decision.value  # type: ignore


# Convenience function for simple usage
def create_classifier(
    ignore_words: frozenset[str] | None = None,
    interrupt_words: frozenset[str] | None = None,
    grace_period_ms: int | None = None,
) -> InterruptionClassifier:
    """Create an InterruptionClassifier with optional custom configuration.

    Args:
        ignore_words: Custom ignore words (uses defaults if None).
        interrupt_words: Custom interrupt words (uses defaults if None).
        grace_period_ms: Custom grace period in milliseconds.

    Returns:
        Configured InterruptionClassifier instance.
    """
    config = InterruptionFilterConfig()

    if ignore_words is not None:
        config = InterruptionFilterConfig(
            ignore_words=ignore_words,
            interrupt_words=config.interrupt_words,
            grace_period_ms=config.grace_period_ms,
        )

    if interrupt_words is not None:
        config = InterruptionFilterConfig(
            ignore_words=config.ignore_words,
            interrupt_words=interrupt_words,
            grace_period_ms=config.grace_period_ms,
        )

    if grace_period_ms is not None:
        config = InterruptionFilterConfig(
            ignore_words=config.ignore_words,
            interrupt_words=config.interrupt_words,
            grace_period_ms=grace_period_ms,
        )

    return InterruptionClassifier(config)
