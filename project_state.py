"""Per-session Production Plan state for Studio Copilot.

The plan is structured application state (goal, findings, decisions, next
actions, plus the active reference and arrangement). The agent edits it only
through update_project_state; the harness records reference/arrangement/
analysis results from other tools via record_tool_result.

Keys starting with "_" are internal bookkeeping and are never returned to the
model or the browser.
"""

import copy
import json
import math

MAX_FINDINGS = 10
MAX_DECISIONS = 8
MAX_PENDING_ACTIONS = 5
MAX_DONE_ACTIONS = 5
MAX_GOAL_CHARS = 240
MAX_ITEM_CHARS = 220
MAX_ANALYSIS_LOG = 50

# A measured finding's timestamps may extend this far past an analyzed range.
ANALYZED_RANGE_TOLERANCE_SECONDS = 0.5

# Finding ranges may overshoot the decoded duration by this much (rounding).
DURATION_TOLERANCE_SECONDS = 1.0

FINDING_SOURCES = ("measurement", "user")
NEXT_ACTION_STATUSES = ("pending", "done")


def new_project_state() -> dict:
    return {
        "goal": None,
        "findings": [],
        "decisions": [],
        "next_actions": [],
        "reference": None,
        "arrangement": None,
        "revision": 0,
        "_analysis_log": [],
        "_counters": {"F": 0, "D": 0, "N": 0},
    }


def public_project_state(state: dict) -> dict:
    return copy.deepcopy({
        key: value for key, value in state.items() if not key.startswith("_")
    })


def _next_id(state: dict, prefix: str) -> str:
    state["_counters"][prefix] += 1
    return f"{prefix}{state['_counters'][prefix]}"


def _finite_number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _format_seconds(seconds: float) -> str:
    tenths = round(max(0.0, seconds) * 10)
    minutes, rest = divmod(tenths, 600)
    return f"{minutes}:{rest / 10:04.1f}"


def _clean_text(value, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _parse_result(result) -> dict | None:
    if isinstance(result, dict):
        return result
    try:
        parsed = json.loads(result)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


# --- Harness-recorded context ---


def record_tool_result(state: dict, name: str, args: dict, result) -> None:
    """Record durable context from successful tool results.

    Called by the harness after every tool call; never raises.
    """
    try:
        parsed = _parse_result(result)
        if not parsed or "error" in parsed:
            return

        if name == "analyze_audio_track":
            analysis_range = parsed.get("analysis_range") or {}
            start = _finite_number(analysis_range.get("start_seconds"))
            end = _finite_number(analysis_range.get("end_seconds"))
            audio_file_id = parsed.get("audio_file_id") or (args or {}).get("audio_file_id")

            if isinstance(audio_file_id, str) and start is not None and end is not None:
                log = state["_analysis_log"]
                log.append({
                    "audio_file_id": audio_file_id,
                    "start_seconds": start,
                    "end_seconds": end,
                })
                del log[:-MAX_ANALYSIS_LOG]

        elif name == "lookup_reference_track":
            match = parsed.get("selected_match")
            if isinstance(match, dict) and match.get("title"):
                state["reference"] = {
                    key: match.get(key)
                    for key in (
                        "title",
                        "artist",
                        "duration",
                        "duration_seconds",
                        "release",
                        "release_type",
                        "first_release_date",
                        "musicbrainz_url",
                    )
                }
                state["revision"] += 1

        elif name == "build_arrangement":
            sections = parsed.get("arrangement")
            if isinstance(sections, list) and sections:
                state["arrangement"] = {
                    "bpm": parsed.get("bpm"),
                    "time_signature": parsed.get("time_signature"),
                    "seconds_per_bar": parsed.get("seconds_per_bar"),
                    "total_bars": parsed.get("total_bars"),
                    "requested_duration": parsed.get("requested_duration"),
                    "actual_duration": parsed.get("actual_duration"),
                    "actual_duration_seconds": parsed.get("actual_duration_seconds"),
                    "sections": [
                        {
                            key: section.get(key)
                            for key in (
                                "section",
                                "start_bar",
                                "end_bar",
                                "bars",
                                "start_time",
                                "end_time",
                                "start_seconds",
                                "end_seconds",
                            )
                        }
                        for section in sections
                        if isinstance(section, dict)
                    ],
                }
                state["revision"] += 1
    except Exception:
        # Recording context must never break the agent loop.
        return


def set_next_action_status(state: dict, action_id: str, status: str) -> bool:
    """Mark a next action pending/done from the UI. Returns False if not found."""
    if status not in NEXT_ACTION_STATUSES:
        return False

    for action in state["next_actions"]:
        if action["id"] == action_id:
            if action["status"] != status:
                action["status"] = status
                _prune_done_actions(state)
                state["revision"] += 1
            return True

    return False


def _prune_done_actions(state: dict) -> None:
    done_ids = [a["id"] for a in state["next_actions"] if a["status"] == "done"]
    stale = set(done_ids[:-MAX_DONE_ACTIONS]) if len(done_ids) > MAX_DONE_ACTIONS else set()
    if stale:
        state["next_actions"] = [a for a in state["next_actions"] if a["id"] not in stale]


# --- Hidden context for the model ---


def format_project_context(state: dict, current_audio_file_id: str | None) -> str:
    goal = state.get("goal")
    findings = state.get("findings") or []
    decisions = state.get("decisions") or []
    actions = state.get("next_actions") or []
    reference = state.get("reference")
    arrangement = state.get("arrangement")

    if not any([goal, findings, decisions, actions, reference, arrangement]):
        return (
            "[Current Studio Copilot project state: the Production Plan is empty. "
            "No goal, findings, decisions, or next actions have been recorded yet.]"
        )

    lines = [
        "[Current Studio Copilot project state (the user's persistent Production Plan, "
        "shown to the user in the workspace; change it only with update_project_state):",
        f"Goal: {goal}" if goal else "Goal: not set",
    ]

    if findings:
        lines.append("Key findings (grounded in tool measurements or user statements):")
        for finding in findings:
            tags = ["measured" if finding.get("source") == "measurement" else "user-stated"]

            if finding.get("start_seconds") is not None and finding.get("end_seconds") is not None:
                tags.append(
                    f"{_format_seconds(finding['start_seconds'])}-"
                    f"{_format_seconds(finding['end_seconds'])}"
                )

            if finding.get("audio_file_id"):
                if finding["audio_file_id"] == current_audio_file_id:
                    tags.append("current audio")
                else:
                    tags.append("earlier upload; timestamps may not apply to the current audio")

            lines.append(f"- {finding['id']} ({', '.join(tags)}): {finding['text']}")
    else:
        lines.append("Key findings: none recorded")

    if decisions:
        lines.append("Decisions (choices the user made):")
        lines += [f"- {d['id']}: {d['text']}" for d in decisions]
    else:
        lines.append("Decisions: none recorded")

    if actions:
        lines.append("Next actions (suggested steps, not decisions; done only if marked done):")
        lines += [f"- {a['id']} [{a['status']}]: {a['text']}" for a in actions]
    else:
        lines.append("Next actions: none recorded")

    if reference:
        details = [reference.get("duration")]
        if reference.get("release"):
            year = str(reference.get("first_release_date") or "")[:4]
            details.append(f"release {reference['release']}" + (f" ({year})" if year else ""))
        lines.append(
            "Active reference (from lookup_reference_track): "
            f"{reference.get('title')} - {reference.get('artist')}"
            + "".join(f", {d}" for d in details if d)
        )

    if arrangement:
        sections = " | ".join(
            f"{s.get('section')} {s.get('bars')}" for s in arrangement.get("sections", [])
        )
        lines.append(
            "Active arrangement (a build_arrangement plan, not detected from the audio): "
            f"{arrangement.get('bpm')} BPM {arrangement.get('time_signature') or ''}, "
            f"{arrangement.get('actual_duration')}, {arrangement.get('total_bars')} bars: "
            f"{sections}"
        )

    lines[-1] += "]"
    return "\n".join(lines)


# --- The tool ---


def _error(message: str, action: str) -> str:
    return json.dumps({"error": message, "action": action})


def _validate_finding(item, index, state, current_audio_file_id):
    """Return (finding_dict, None) or (None, error_json)."""
    from tools import get_audio_duration, resolve_uploaded_audio

    label = f"add_findings[{index}]"

    if not isinstance(item, dict):
        return None, _error(
            f"{label} must be an object with text and source.",
            'Use {"text": "...", "source": "measurement"} or "user".',
        )

    text = _clean_text(item.get("text"), MAX_ITEM_CHARS)
    if not text:
        return None, _error(f"{label}.text must be a non-empty string.", "Describe the finding in one sentence.")

    source = item.get("source")
    if source not in FINDING_SOURCES:
        return None, _error(
            f"{label}.source must be 'measurement' or 'user'.",
            "Use 'measurement' for analyze_audio_track results and 'user' for facts the user stated.",
        )

    raw_start = item.get("start_seconds")
    raw_end = item.get("end_seconds")
    has_range = raw_start is not None or raw_end is not None
    start = end = None

    if has_range:
        start = _finite_number(raw_start)
        end = _finite_number(raw_end)
        if start is None or end is None:
            return None, _error(
                f"{label} needs both start_seconds and end_seconds as numbers.",
                "Provide both timestamps in seconds, or neither.",
            )
        if start < 0 or end <= start:
            return None, _error(
                f"{label} has an invalid range {start}-{end}.",
                "Use 0 <= start_seconds < end_seconds.",
            )

    audio_file_id = item.get("audio_file_id") or None
    if audio_file_id is None and (has_range or source == "measurement"):
        audio_file_id = current_audio_file_id

    if audio_file_id is not None:
        try:
            file_path = resolve_uploaded_audio(audio_file_id)
        except (ValueError, FileNotFoundError):
            return None, _error(
                f"{label} references an unknown audio_file_id.",
                "Use the exact audio_file_id from the attached audio context.",
            )

        if has_range:
            try:
                duration = get_audio_duration(file_path)
            except Exception:
                duration = None
            if duration is not None and end > duration + DURATION_TOLERANCE_SECONDS:
                return None, _error(
                    f"{label} range ends after the track ({round(duration, 2)} s).",
                    "Use a range inside the track.",
                )
    elif has_range:
        return None, _error(
            f"{label} has timestamps but no audio file is attached.",
            "Only attach timestamps to findings about an uploaded audio file.",
        )

    if source == "measurement":
        if audio_file_id is None:
            return None, _error(
                f"{label} is a measurement finding but no audio file is attached.",
                "Save measured findings only after analyzing an uploaded file.",
            )

        analyses = [a for a in state["_analysis_log"] if a["audio_file_id"] == audio_file_id]
        if not analyses:
            return None, _error(
                f"{label} is marked as a measurement, but this audio has not been analyzed in this session.",
                "Run analyze_audio_track first, then save findings supported by its results.",
            )

        if has_range and not any(
            start >= a["start_seconds"] - ANALYZED_RANGE_TOLERANCE_SECONDS
            and end <= a["end_seconds"] + ANALYZED_RANGE_TOLERANCE_SECONDS
            for a in analyses
        ):
            return None, _error(
                f"{label} range {_format_seconds(start)}-{_format_seconds(end)} is not covered by any analysis.",
                "Run analyze_audio_track on this range before saving it as a measured finding.",
            )

    finding = {"text": text, "source": source}
    if audio_file_id is not None:
        finding["audio_file_id"] = audio_file_id
    if has_range:
        finding["start_seconds"] = round(start, 2)
        finding["end_seconds"] = round(end, 2)

    return finding, None


def _string_list(value, name):
    """Return (cleaned_list, None) or (None, error_json)."""
    if value is None:
        return [], None
    if not isinstance(value, list):
        return None, _error(f"{name} must be a list of strings.", f"Pass {name} as a JSON array.")

    cleaned = []
    for index, item in enumerate(value):
        text = _clean_text(item, MAX_ITEM_CHARS)
        if not text:
            return None, _error(f"{name}[{index}] must be a non-empty string.", "Remove empty entries.")
        cleaned.append(text)
    return cleaned, None


def _id_list(value, name):
    if value is None:
        return [], None
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        return None, _error(f"{name} must be a list of item IDs such as 'N1'.", "Use IDs from the project state.")
    return [v.strip().upper() for v in value], None


def update_project_state(
    state: dict,
    current_audio_file_id: str | None,
    /,
    goal: str | None = None,
    add_findings: list | None = None,
    add_decisions: list | None = None,
    add_next_actions: list | None = None,
    complete_next_actions: list | None = None,
    remove_items: list | None = None,
) -> str:
    """Apply a validated, atomic patch to the session's Production Plan."""
    new_goal = None
    if goal is not None:
        new_goal = _clean_text(goal, MAX_GOAL_CHARS)
        if not new_goal:
            return _error("goal must be a non-empty string.", "State the user's goal in one sentence.")

    if add_findings is not None and not isinstance(add_findings, list):
        return _error("add_findings must be a list of objects.", "Pass add_findings as a JSON array.")

    decisions, error = _string_list(add_decisions, "add_decisions")
    if error:
        return error
    actions, error = _string_list(add_next_actions, "add_next_actions")
    if error:
        return error
    complete_ids, error = _id_list(complete_next_actions, "complete_next_actions")
    if error:
        return error
    remove_ids, error = _id_list(remove_items, "remove_items")
    if error:
        return error

    existing_ids = {
        item["id"]
        for key in ("findings", "decisions", "next_actions")
        for item in state[key]
    }
    action_ids = {a["id"] for a in state["next_actions"]}

    unknown = [i for i in remove_ids if i not in existing_ids]
    if unknown:
        return _error(f"Unknown item IDs in remove_items: {unknown}.", "Use IDs listed in the project state.")

    unknown = [i for i in complete_ids if i not in action_ids or i in remove_ids]
    if unknown:
        return _error(
            f"Unknown next action IDs in complete_next_actions: {unknown}.",
            "Use next action IDs (N#) listed in the project state.",
        )

    findings = []
    for index, item in enumerate(add_findings or []):
        finding, error = _validate_finding(item, index, state, current_audio_file_id)
        if error:
            return error
        findings.append(finding)

    if new_goal is None and not any([findings, decisions, actions, complete_ids, remove_ids]):
        return _error(
            "No changes were provided.",
            "Only call update_project_state when the goal, findings, decisions, or next actions change.",
        )

    def kept(key):
        return [item for item in state[key] if item["id"] not in remove_ids]

    def dedupe(new_items, existing_texts, key=lambda x: x):
        seen = {t.casefold() for t in existing_texts}
        unique, skipped = [], []
        for item in new_items:
            folded = key(item).casefold()
            if folded in seen:
                skipped.append(key(item))
            else:
                seen.add(folded)
                unique.append(item)
        return unique, skipped

    kept_findings = kept("findings")
    kept_decisions = kept("decisions")
    kept_actions = kept("next_actions")

    findings, skipped_findings = dedupe(
        findings, [f["text"] for f in kept_findings], key=lambda f: f["text"]
    )
    decisions, skipped_decisions = dedupe(decisions, [d["text"] for d in kept_decisions])
    actions, skipped_actions = dedupe(actions, [a["text"] for a in kept_actions])

    if len(kept_findings) + len(findings) > MAX_FINDINGS:
        return _error(
            f"The plan can hold at most {MAX_FINDINGS} findings.",
            "Remove outdated findings with remove_items (F#) before adding more.",
        )

    if len(kept_decisions) + len(decisions) > MAX_DECISIONS:
        return _error(
            f"The plan can hold at most {MAX_DECISIONS} decisions.",
            "Remove outdated decisions with remove_items (D#) before adding more.",
        )

    pending_after = sum(
        1 for a in kept_actions if a["status"] == "pending" and a["id"] not in complete_ids
    ) + len(actions)
    if pending_after > MAX_PENDING_ACTIONS:
        return _error(
            f"The plan can hold at most {MAX_PENDING_ACTIONS} pending next actions.",
            "Complete or remove existing next actions (N#) first, and keep the list focused.",
        )

    # --- Apply (all validation passed) ---
    changes = []

    if remove_ids:
        for key in ("findings", "decisions", "next_actions"):
            state[key] = [item for item in state[key] if item["id"] not in remove_ids]
        changes.append(f"removed {', '.join(remove_ids)}")

    for action in state["next_actions"]:
        if action["id"] in complete_ids and action["status"] != "done":
            action["status"] = "done"
            changes.append(f"completed {action['id']}")

    if new_goal is not None and new_goal != state["goal"]:
        changes.append("set goal" if state["goal"] is None else "updated goal")
        state["goal"] = new_goal

    for finding in findings:
        finding = {"id": _next_id(state, "F"), **finding}
        state["findings"].append(finding)
        changes.append(f"added finding {finding['id']}")

    for text in decisions:
        decision = {"id": _next_id(state, "D"), "text": text}
        state["decisions"].append(decision)
        changes.append(f"added decision {decision['id']}")

    for text in actions:
        action = {"id": _next_id(state, "N"), "text": text, "status": "pending"}
        state["next_actions"].append(action)
        changes.append(f"added next action {action['id']}")

    _prune_done_actions(state)

    skipped = skipped_findings + skipped_decisions + skipped_actions
    if changes:
        state["revision"] += 1

    result = {
        "status": "updated" if changes else "unchanged",
        "changes": changes,
        "project_state": public_project_state(state),
    }
    if skipped:
        result["skipped_duplicates"] = skipped

    return json.dumps(result)


UPDATE_PROJECT_STATE_TOOL = {
    "type": "function",
    "function": {
        "name": "update_project_state",
        "description": (
            "Update the user's persistent Production Plan, which is shown in the "
            "Studio Copilot workspace and given back to you as project state on later "
            "turns. This is durable project memory ('remember this'), unlike "
            "focus_audio_region, which only directs visual attention right now. "
            "Call it only when something durable changes: the user states or changes "
            "their production goal; an analysis produced a genuinely useful finding; "
            "the user made an actual decision; concrete next actions were agreed; or an "
            "existing next action was completed or became obsolete. Do NOT call it on "
            "every response and do not save every observation. "
            "Goal: the user's own stated intent, never a guess. "
            "Findings: only facts supported by analyze_audio_track measurements "
            "(source 'measurement') or explicit user statements (source 'user'); no "
            "speculation, no invented numbers, no section labels the user did not give. "
            "Include exact analyzed start_seconds/end_seconds when the finding is about a "
            "specific region. "
            "Decisions: only choices the user actually made; your suggestions are not "
            "decisions. "
            "Next actions: 2-5 concrete, actionable production steps; they are "
            "suggestions until the user does them. "
            "Returns the full resulting project state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": (
                        "Set or replace the production goal, in one sentence, only when "
                        "the user has stated or changed it."
                    ),
                },
                "add_findings": {
                    "type": "array",
                    "description": "Grounded findings to add to Key Findings.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": (
                                    "One concise sentence, citing measured values when "
                                    "relevant, e.g. 'Median RMS drops to -30 dBFS, about "
                                    "13 dB below the track median.'"
                                ),
                            },
                            "source": {
                                "type": "string",
                                "enum": list(FINDING_SOURCES),
                                "description": (
                                    "'measurement' if supported by analyze_audio_track "
                                    "results in this session; 'user' if the user stated it."
                                ),
                            },
                            "start_seconds": {
                                "type": "number",
                                "description": "Optional exact analyzed region start in seconds.",
                            },
                            "end_seconds": {
                                "type": "number",
                                "description": "Optional exact analyzed region end in seconds.",
                            },
                            "audio_file_id": {
                                "type": "string",
                                "description": (
                                    "Optional; defaults to the currently attached audio file."
                                ),
                            },
                        },
                        "required": ["text", "source"],
                    },
                },
                "add_decisions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Choices the user actually made, one short sentence each.",
                },
                "add_next_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concrete production steps to add to Next Moves.",
                },
                "complete_next_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "IDs of next actions (e.g. 'N2') the user has completed.",
                },
                "remove_items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "IDs of findings, decisions, or next actions (F#, D#, N#) that are "
                        "obsolete, superseded, or wrong."
                    ),
                },
            },
        },
    },
}
