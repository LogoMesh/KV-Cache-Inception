"""
SAGE Library Loader — Battle-time payload selection from pre-computed libraries.

Sits between the PayloadLibrary and the live attacker agent. At the start of
each battle, LibraryLoader loads the library for the detected game type and
serves payloads on demand, tracking which ones have been used to ensure
diversity across rounds.

Integration with agent.py:
    The attacker checks AblationConfig.use_payload_library at battle start.
    If True, it instantiates LibraryLoader and calls .next_payload() instead
    of running the full 4-call generation loop.

Call budget impact:
    Without library: 4 calls/round (expand → generate → refine → self-eval)
    With library:    1 call/round  (deliver payload → score response)
    Savings:         3 calls/round freed for additional rounds or self-eval

Diversity strategy:
    - Primary: serve top-k by composite_score, excluding already-used payloads
    - Fallback 1: if top-k exhausted, serve used payloads (recycling with
                  awareness that they previously failed)
    - Fallback 2: if library empty for this game type, return None so the
                  attacker falls back to live generation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .payload_library import PayloadEntry, PayloadLibrary

if TYPE_CHECKING:
    from .ablation import AblationConfig

logger = logging.getLogger(__name__)


class LibraryLoader:
    """Serves pre-computed payloads for a single battle session.

    Instantiate once per battle (not per round). Tracks usage across
    rounds so the diversity selection works correctly.

    Args:
        library: Loaded PayloadLibrary to draw from.
        game_type: The detected game type for this battle.
        max_h_score: Payloads with h_score above this are skipped in
                     primary selection. Default 0.7 (exclude incoherent).
        min_composite: Minimum composite score for primary selection.

    Example
    -------
    # In attacker agent __init__ or battle setup:
    if config.use_payload_library and config.payload_library_path:
        lib = PayloadLibrary.load(config.payload_library_path)
        loader = LibraryLoader(lib, game_type=detected_game_type)

    # In each round:
    payload = loader.next_payload()
    if payload is None:
        # Fall back to live generation
        payload = await self._generate_payload_live(...)
    else:
        loader.mark_used(payload, battle_result=0.0)  # update after scoring
    """

    def __init__(
        self,
        library: PayloadLibrary,
        game_type: str,
        max_h_score: float = 0.7,
        min_composite: float = 0.0,
    ) -> None:
        self._library = library
        self._game_type = game_type
        self._max_h_score = max_h_score
        self._min_composite = min_composite

        # Track used payload objects by identity (not content) this session
        self._used_this_session: set[int] = set()
        self._round = 0

        available = library.top_k(
            game_type=game_type, k=9999,
            exclude_used=False, max_h_score=max_h_score,
            min_composite=min_composite,
        )
        logger.info(
            "LibraryLoader initialized: game_type=%s, %d/%d payloads eligible",
            game_type, len(available), len(library),
        )

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def next_payload(self) -> PayloadEntry | None:
        """Return the best unused payload for this round.

        Selection order:
        1. Unused payloads (not used in prior battles AND not used this session),
           sorted by composite_score descending.
        2. Session-recycled payloads (used this session but not in prior battles),
           sorted by composite_score.
        3. All payloads including prior-battle-used ones (last resort).
        4. None — library is empty for this game type → caller should fall back
           to live generation.

        Returns:
            PayloadEntry or None.
        """
        self._round += 1

        # Priority 1: fresh payloads not used anywhere
        candidate = self._pick(exclude_battle_used=True, exclude_session_used=True)
        if candidate:
            logger.info(
                "Round %d: serving fresh payload (composite=%.3f, strategy=%s)",
                self._round, candidate.composite_score, candidate.strategy,
            )
            return candidate

        # Priority 2: not used in prior battles but used this session
        candidate = self._pick(exclude_battle_used=True, exclude_session_used=False)
        if candidate:
            logger.info(
                "Round %d: recycling session payload (composite=%.3f)",
                self._round, candidate.composite_score,
            )
            return candidate

        # Priority 3: any payload (including prior-battle failures)
        candidate = self._pick(exclude_battle_used=False, exclude_session_used=False)
        if candidate:
            logger.info(
                "Round %d: serving previously-failed payload (composite=%.3f)",
                self._round, candidate.composite_score,
            )
            return candidate

        logger.warning(
            "Round %d: library exhausted for game_type=%s — caller should fall back to live generation",
            self._round, self._game_type,
        )
        return None

    def mark_used(self, entry: PayloadEntry, battle_result: float | None = None) -> None:
        """Record that a payload was delivered in a live round.

        Args:
            entry: The PayloadEntry returned by next_payload().
            battle_result: Reward received in the round (0.0–1.0), if known.
                           Pass None if result is not yet available.
        """
        self._used_this_session.add(id(entry))
        entry.used_in_battle = True
        if battle_result is not None:
            entry.battle_result = battle_result

    def remaining(self) -> int:
        """Number of payloads not yet used this session."""
        all_eligible = self._library.top_k(
            game_type=self._game_type, k=9999,
            exclude_used=False,
            max_h_score=self._max_h_score,
            min_composite=self._min_composite,
        )
        return sum(1 for e in all_eligible if id(e) not in self._used_this_session)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: "AblationConfig",
        game_type: str,
    ) -> "LibraryLoader | None":
        """Construct from AblationConfig. Returns None if library is disabled/missing.

        Args:
            config: AblationConfig with use_payload_library and payload_library_path.
            game_type: Detected game type for the current battle.

        Returns:
            LibraryLoader if library is enabled and file exists, else None.
        """
        if not config.use_payload_library:
            return None

        if not config.payload_library_path:
            logger.warning(
                "use_payload_library=True but payload_library_path is None. "
                "Set AblationConfig.payload_library_path to a valid JSON file."
            )
            return None

        try:
            library = PayloadLibrary.load(config.payload_library_path)
        except FileNotFoundError as e:
            logger.error("LibraryLoader: %s — falling back to live generation", e)
            return None

        if not library.top_k(game_type=game_type, k=1):
            logger.warning(
                "Library has no payloads for game_type=%s (has: %s). "
                "Falling back to live generation.",
                game_type, library.game_types(),
            )
            return None

        return cls(library=library, game_type=game_type)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pick(
        self,
        exclude_battle_used: bool,
        exclude_session_used: bool,
    ) -> PayloadEntry | None:
        candidates = self._library.top_k(
            game_type=self._game_type,
            k=9999,
            exclude_used=exclude_battle_used,
            max_h_score=self._max_h_score,
            min_composite=self._min_composite,
        )
        # Confirmed wins (battle_result=1.0) rank above all non-wins regardless
        # of composite_score. Within each tier, composite_score breaks ties.
        candidates.sort(
            key=lambda e: (0 if e.battle_result == 1.0 else 1, -e.composite_score)
        )
        for entry in candidates:
            if exclude_session_used and id(entry) in self._used_this_session:
                continue
            return entry
        return None
