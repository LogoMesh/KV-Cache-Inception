#!/usr/bin/env python3
"""
Replay attacker traces against the defender.

Reads saved attacker traces from a result.json and replays them against a
running defender agent, so you can iterate on the defender without
re-running the attacker each time.

Usage:
    python scenarios/security_arena/replay_battle.py \
        --traces results/portfolioiq/result.json \
        --defender-url http://127.0.0.1:9020

    # Or pull defender URL from a scenario TOML:
    python scenarios/security_arena/replay_battle.py \
        --traces results/portfolioiq/result.json \
        --scenario scenarios/security_arena/scenario_portfolioiq.toml
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add scenarios directory to path so we can import plugins
sys.path.insert(0, str(Path(__file__).parent))

from arena_common import ArenaResult, RoundResult
from plugins.registry import load_scenario

# Reuse the same HTTP client the orchestrator uses
from agentbeats.tool_provider import ToolProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("replay_battle")


def load_traces(path: Path) -> ArenaResult:
    """Load an ArenaResult from a result.json file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return ArenaResult(**data)


def load_defender_url_from_toml(toml_path: Path) -> str:
    """Extract the defender endpoint from a scenario TOML file."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    for p in data.get("participants", []):
        if p.get("role") == "defender":
            return p["endpoint"]
    raise ValueError(f"No defender participant found in {toml_path}")


def load_config_from_toml(toml_path: Path) -> dict:
    """Load the [config] section from a scenario TOML file."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]
    data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return data.get("config", {})


async def replay(
    traces: ArenaResult,
    defender_url: str,
    scenario_config: dict,
    output_path: Path,
    timeout: int,
) -> ArenaResult:
    """Replay every round from *traces* against the defender and return a new ArenaResult."""

    # Load the scenario plugin so we can call check_success and build context
    scenario = load_scenario(traces.scenario_type, scenario_config)
    context = scenario.get_scenario_context(current_round=0, battle_history=[])

    tp = ToolProvider()
    new_battle_log: list[RoundResult] = []
    first_manipulation_round: int | None = None
    defender_timeout_errors = 0
    defender_crash_errors = 0

    for rr in traces.battle_log:
        round_num = rr.round
        logger.info("--- Round %d/%d ---", round_num, len(traces.battle_log))

        # Build the defender message from the saved formatted_input
        defender_msg = json.dumps({
            "domain_description": context.domain_description,
            "defender_role": context.defender_role,
            "defender_task": context.defender_task,
            "output_format": context.output_format,
            "formatted_input": rr.formatted_input,
        }, indent=2)

        # Send to defender
        try:
            defender_response = await asyncio.wait_for(
                tp.talk_to_agent(defender_msg, defender_url, new_conversation=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.error("Defender TIMEOUT on round %d", round_num)
            defender_timeout_errors += 1
            new_battle_log.append(RoundResult(
                round=round_num,
                attack_output=rr.attack_output,
                formatted_input=rr.formatted_input,
                defender_output="[TIMEOUT]",
                manipulation_detected=True,
                extracted_value="timeout",
            ))
            first_manipulation_round = round_num
            break
        except Exception as e:
            logger.error("Defender CRASH on round %d: %s", round_num, e)
            defender_crash_errors += 1
            new_battle_log.append(RoundResult(
                round=round_num,
                attack_output=rr.attack_output,
                formatted_input=rr.formatted_input,
                defender_output=f"[CRASH: {e}]",
                manipulation_detected=True,
                extracted_value="crash",
            ))
            first_manipulation_round = round_num
            break

        logger.info("Defender response (%d chars): %s", len(defender_response), defender_response[:200])

        # Check for manipulation
        manipulation_detected, extracted_value = scenario.check_success(defender_response)
        new_battle_log.append(RoundResult(
            round=round_num,
            attack_output=rr.attack_output,
            formatted_input=rr.formatted_input,
            defender_output=defender_response,
            manipulation_detected=manipulation_detected,
            extracted_value=extracted_value,
        ))

        if manipulation_detected:
            logger.warning("MANIPULATION detected round %d! Value: %s", round_num, extracted_value)
            first_manipulation_round = round_num
            break
        else:
            logger.info("Round %d: Blocked (value: %s)", round_num, extracted_value)

    # Build new result
    if first_manipulation_round is not None:
        winner = "attacker"
        rounds_defended = first_manipulation_round - 1
    else:
        winner = "defender"
        rounds_defended = len(traces.battle_log)

    result = ArenaResult(
        team_name=traces.team_name,
        scenario_type=traces.scenario_type,
        winner=winner,
        manipulation_round=first_manipulation_round,
        rounds_defended=rounds_defended,
        total_rounds=traces.total_rounds,
        battle_log=new_battle_log,
        input_format_template=traces.input_format_template,
        attacker_timeout_errors=traces.attacker_timeout_errors,
        attacker_crash_errors=traces.attacker_crash_errors,
        defender_timeout_errors=defender_timeout_errors,
        defender_crash_errors=defender_crash_errors,
    )

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Results saved to %s", output_path)

    # Print summary
    print()
    print("=" * 60)
    print(f"REPLAY COMPLETE — {result.scenario_type}")
    print("=" * 60)
    print(f"Winner:          {result.winner}")
    print(f"Rounds defended: {result.rounds_defended}/{result.total_rounds}")
    if result.manipulation_round:
        print(f"Breached round:  {result.manipulation_round}")
    print()
    for rr in result.battle_log:
        status = "BREACHED" if rr.manipulation_detected else "Blocked"
        print(f"  Round {rr.round}: {status} (value: {rr.extracted_value})")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Replay attacker traces against the defender"
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("results/portfolioiq/result.json"),
        help="Path to result.json with saved attacker traces",
    )
    parser.add_argument(
        "--defender-url",
        type=str,
        default=None,
        help="Defender agent URL (e.g. http://127.0.0.1:9020)",
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        default=None,
        help="Scenario TOML file (used to get defender URL and config if not specified)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for replay result (default: <traces_dir>/replay_result.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-round timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    # Load traces
    if not args.traces.exists():
        parser.error(f"Traces file not found: {args.traces}")
    traces = load_traces(args.traces)
    logger.info("Loaded traces: %s, %d rounds", traces.scenario_type, len(traces.battle_log))

    # Resolve defender URL
    defender_url = args.defender_url
    scenario_config: dict = {}
    if args.scenario and args.scenario.exists():
        scenario_config = load_config_from_toml(args.scenario)
        if defender_url is None:
            defender_url = load_defender_url_from_toml(args.scenario)
    if defender_url is None:
        defender_url = "http://127.0.0.1:9020"
        logger.info("No --defender-url or --scenario given, defaulting to %s", defender_url)

    # Build scenario config from traces if not loaded from TOML
    if not scenario_config:
        scenario_config = {"scenario_type": traces.scenario_type}

    # Resolve output path
    output_path = args.output or args.traces.parent / "replay_result.json"

    logger.info("Defender URL: %s", defender_url)
    logger.info("Output: %s", output_path)

    asyncio.run(replay(traces, defender_url, scenario_config, output_path, args.timeout))


if __name__ == "__main__":
    main()
