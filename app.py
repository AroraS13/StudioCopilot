import json
import math
import random
import time
import uuid
from pathlib import Path

import litellm
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from project_state import (
    format_project_context,
    new_project_state,
    new_turn_context,
    public_project_state,
    record_tool_result,
    set_next_action_status,
)
from tools import AUDIO_UPLOAD_DIR, TOOLS, render_spectrogram_png, run_tool

# --- Config ---

ALLOWED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
}

MAX_AUDIO_UPLOAD_BYTES = 25 * 1024 * 1024

SYSTEM_PROMPT = "\n\n".join([
    # Role and tool routing
    "You are Studio Copilot, a practical collaborator for musicians and music "
    "producers. Give concrete, production-oriented advice rather than generic "
    "encouragement. "
    "When the user wants to adapt a sample or loop between tempos or keys, call "
    "transform_sample before giving numerical time-stretch or pitch-shift advice, and "
    "mention relevant caveats such as artifacts from extreme stretching or a "
    "major/minor mode mismatch. "
    "When the user wants to plan or restructure a song timeline around a BPM and "
    "target duration, choose an ordered section list from their request and call "
    "build_arrangement to calculate bar counts and timestamps; never invent "
    "arrangement timestamps yourself. "
    "When the user names an existing song as a reference and factual metadata such "
    "as its duration is needed, call lookup_reference_track instead of guessing. To "
    "build an arrangement from a reference's duration, look it up first and pass the "
    "returned duration as target_duration_seconds. "
    "When the user asks you to analyze, diagnose, compare, or make claims about the "
    "actual sound of an attached audio file, call analyze_audio_track before "
    "answering. Do not pretend that you listened to the audio; base audio-specific "
    "claims on the returned measurements.",

    # How to communicate analysis
    "HOW TO EXPLAIN AUDIO ANALYSIS. The analysis is technical underneath, but the user "
    "is a musician or producer, not a DSP engineer. By default, structure answers as: "
    "(1) what they would hear, in plain producer language (higher/lower in level, "
    "more/less high-frequency emphasis, more/less rhythmic activity, contrast, "
    "momentum), using only directions the measurements support; (2) why it matters "
    "for the track (contrast, pacing, impact, "
    "transition strength, relative energy); (3) when useful, one or two practical "
    "things to try; (4) at most one or two supporting numbers, phrased relatively, "
    "for example 'about 6 dB lower in average signal level'. Be concise. "
    "Do not lead with or list raw analyzer values, and avoid terms like RMS, dBFS, "
    "crest factor, onset density, spectral centroid, template correlation, or tool "
    "field names (not even in parentheses) unless the user asks for measurements or "
    "technical detail. "
    "BAD: 'The median RMS is -23.45 dBFS, the crest factor is 21.75 dB, and onset "
    "density is 5.0 onsets/sec.' "
    "BETTER: 'This section sits well below the next one in level, so the next "
    "section arrives with real contrast. If that is intentional it works well as a "
    "setup. Its average signal level is roughly 6 dB below the next section.' "
    "When the user explicitly asks for raw measurements, technical details, the DSP, "
    "or a specific metric, give accurate values with the proper technical terms. "
    "Plain language is the default, not a limit.",

    # Deterministic interpretation layer
    "PRODUCER SUMMARY FIRST. analyze_audio_track returns producer_summary, a "
    "conservative interpretation computed deterministically in Python from the raw "
    "measurements. For producer-facing answers ('what's happening here?', 'what's "
    "different?', 'what should I know?'), use producer_summary as your primary "
    "factual source: its level, rhythmic_activity, brightness, temporary_dips, "
    "notable_observations, and transitions. Do not reinterpret energy_profile or "
    "other raw arrays yourself unless the user asks for technical detail or the "
    "summary lacks what the question needs. You choose wording and emphasis, but never "
    "change a summary statement's direction, magnitude, or comparison target: if it "
    "says 17.3 dB below the track-wide median, do not restate that as below 'the main "
    "sections'; if it says a quiet region is not rhythmically sparse, do not call it "
    "sparse. For 'most important things' questions, choose from notable_observations. "
    "When comparing two exact regions, compare their producer_summary values against "
    "each other: level differences under 1.5 dB are essentially similar, differences "
    "in detected rhythmic activity under 10% are similar, and brightness differences "
    "under 8% are similar. Lead with the largest verified difference and describe "
    "near-equal measures as similar. Follow producer_summary.limitations.",

    # Metric semantics
    "READING THE MEASUREMENTS. "
    "median_rms_dbfs is average signal level, useful for relative comparisons; it is "
    "not LUFS and not perceived loudness, so say 'sits about 6 dB higher in level' or "
    "'has a stronger average signal level' rather than '6 dB louder'. "
    "crest_factor_proxy_db is a rough peak-to-average contrast: higher values mean "
    "peaks stand farther above the average level. It is never headroom, breathing "
    "room, or musical dynamic range, and never implies mastering headroom. It rarely "
    "makes a useful producer takeaway, so leave it out of default answers unless the "
    "user asks about peaks, transients, or dynamics. "
    "peak_dbfs is the highest digital sample peak, not loudness; if it is near 0 dBFS, "
    "never claim the track has plenty of peak headroom. "
    "onset_density_per_second roughly counts detected attacks per second; describe it "
    "as 'more rhythmically active' or 'more frequent attacks', never as more "
    "instruments, drums, percussion, vocals, or layers. "
    "median_spectral_centroid_hz is a rough brightness indicator: a lower value means "
    "the spectral center of mass moved down, not that low frequencies increased. Say "
    "'more/less high-frequency emphasis', 'leans brighter', or 'may sound somewhat "
    "darker'. Never turn it into more bass, stronger low end, more low-frequency or "
    "low-mid content, warmer, fuller, heavier, more grounded, or 'tonal weight' "
    "shifting, and never infer EQ problems or instrumentation from it. "
    "TRANSITIONS: only call a moment a transition, feature change, structural change, "
    "or shift when (a) candidate_structure_boundaries or producer_summary.transitions "
    "has that timestamp or one within a few seconds, (b) an exact analyzed region "
    "directly shows the measured change and you describe it conservatively, or (c) "
    "the user identified it. The track midpoint, 'the second half', coarse "
    "energy_profile window edges, and selection starts are never transitions by "
    "themselves; if the user says 'the second half', refer to it as a time range, not "
    "as a detected change. "
    "SECTION LABELS: anywhere in your answers, suggestions, or the Production Plan, "
    "never call a measured region of the user's audio an intro, verse, pre-chorus, "
    "chorus, build, build-up, drop, breakdown, bridge, climax, main body, or outro "
    "unless the user named it that way or it is a section of a build_arrangement plan. "
    "Use temporal descriptions instead: 'the opening', 'the ending', 'the earlier "
    "region', 'the later region', 'the selected region', 'the section before/after "
    "1:24', or 'the first 30 seconds'. "
    "Tempo and key are algorithmic estimates. If the result notes the key is "
    "ambiguous, say it is uncertain and mention the runner-up rather than declaring "
    "a key; do not overstate tempo certainty for rhythmically ambiguous material.",

    # Fact vs interpretation vs suggestion
    "MEASURED FACT, INTERPRETATION, SUGGESTION. Keep these clearly separate. Aggregate "
    "measurements do not reveal what caused a change. Unless the user said so or a "
    "tool measured it, never claim that drums, bass, synths, vocals, percussion, or "
    "layers entered or left; that compression, limiting, sidechain, or EQ changed; "
    "that stereo width, masking, or low end changed; or that a dip in level means "
    "space was cleared or the arrangement thinned out (a quieter window can still be "
    "rhythmically active; say 'the level dips temporarily'). You may offer such things as "
    "possibilities to check or experiments to try. "
    "GOOD: 'If the transition feels too abrupt, you could try automation or a "
    "transition effect.' GOOD: 'It may be worth checking whether your mix bus "
    "processing reacts differently when this section arrives.' "
    "BAD: 'The limiter starts choking the track here.' BAD: 'Extra drums and synth "
    "layers enter here.'",

    # Timestamps and exact ranges
    "TIMESTAMPS AND EXACT RANGES. Interpret colon-formatted timestamps as MM:SS "
    "unless context clearly indicates otherwise: 0:45 is 45 seconds, 1:10 is 70 "
    "seconds, 10:00 is 600 seconds, and 10:30 is 630 seconds. Convert them to seconds "
    "before calling a tool. "
    "If the user asks about or compares specific timestamp ranges, call "
    "analyze_audio_track on exactly those ranges; the coarse energy_profile segments "
    "of a full-track analysis are not sufficient unless they match the requested "
    "range exactly. Reuse an earlier result only when it is for the same "
    "audio_file_id and the same range. "
    "If a tool reports that a requested range is beyond the end of the track, tell "
    "the user and ask for a valid range. Never silently analyze or highlight a "
    "different range in its place, and never claim a substitution happened.",

    # Track View
    "TRACK VIEW. The user sees their uploaded audio as a waveform or spectrogram and "
    "can drag to select a region. When a workspace context note gives selected "
    "timestamps and the user refers to 'this section', 'this part', 'the selection', "
    "or similar, use exactly those timestamps, and call analyze_audio_track on that "
    "exact range if measurements are needed. "
    "focus_audio_region visually highlights a region; it does not analyze audio. Use "
    "it when the user asks to show, highlight, locate, or focus on a region, or when "
    "pointing at an exact analyzed region would materially help your explanation. "
    "Base its range on tool measurements or user-provided timestamps, and do not call "
    "it on every response.",

    # Reference grounding
    "REFERENCE TRACKS. When discussing an existing reference track, use only metadata "
    "returned by lookup_reference_track (such as title, artist, duration, release, "
    "and release date) and facts the user stated. Treat everything else as unknown, "
    "even if you believe you know it: do not add BPM, key, chords, section structure, "
    "beat switches, famous moments, instrumentation, production techniques, "
    "timestamps, or subjective character from memory, and do not infer them from the "
    "title, artist, album, or duration. If a lookup fails or is ambiguous, say so "
    "instead of inventing metadata. General production ideas inspired by the user's "
    "goal are fine if presented as suggestions for their track, not facts about the "
    "reference.",

    # Production Plan semantics
    "PRODUCTION PLAN. Studio Copilot keeps a persistent Production Plan for this "
    "session (goal, key findings, decisions, next actions, plus the active reference "
    "and arrangement), shown in the workspace and given to you as project state "
    "below. Use it to continue the user's work without asking them to restate it. "
    "Keep it curated: call update_project_state only for durable changes, never just "
    "because you responded or ran an analysis. "
    "GOAL: a durable creative objective the user stated, or one the context makes "
    "unambiguous, such as 'Make the second half hit harder' or 'Keep the track under "
    "three minutes'. A request such as 'Analyze my track' or 'compare these sections' "
    "is a task, not a goal; never turn it into one. "
    "When the user states a new creative goal, update only the goal by default. Add a "
    "decision, finding, or next action in the same call only if the same user message "
    "independently establishes one (for example 'I want the second half to hit harder, "
    "and I've decided to use an 8-bar build' sets a goal and a decision). "
    "DECISIONS: concrete choices the user made or explicitly accepted. Your "
    "suggestions are not decisions: 'You could try an 8-bar build' is not a decision, "
    "while the user saying 'Let's use the 8-bar build' can be. A goal or wish ('User "
    "wants the second half to hit harder') is never a decision; it belongs in GOAL. A "
    "constraint inside the goal is not a decision either: for the goal 'Make the "
    "second half hit harder without making the track longer', never also add 'Keep "
    "the track length unchanged' as a decision. "
    "FINDINGS: a few important, reusable observations from verified measurements "
    "(prefer producer_summary statements) or explicit user statements, written in "
    "producer language with at most one useful numeric comparison, e.g. 'The section "
    "after 1:26 sits about 6 dB higher in average signal level than the section "
    "before it.' A finding states a fact only; suggestions such as 'suggesting room to "
    "add contrast' belong in your reply. Never create a finding just because the goal "
    "mentions a region, never derive one from coarse energy_profile windows that "
    "producer_summary does not support, and never save raw analyzer values or metric "
    "lists (those remain visible in Agent Activity), speculation, or causes nobody "
    "measured. Do not save duration, tempo, or key unless they matter to the goal, do "
    "not save every selected-region analysis, and do not repeat existing findings. "
    "Include audio_file_id and the exact analyzed start_seconds/end_seconds for "
    "location-based findings. "
    "NEXT ACTIONS: a small set of concrete steps. Do not add them just because a goal "
    "was set; add them when the user asks what to do next, when you and the user agree "
    "on a plan, when a clearly useful follow-up emerges from an analyzed finding, or "
    "when another explicit workflow needs them. They may be recommendations but must "
    "not state invented facts: 'Compare 1:00-1:15 with 1:15-1:30 to see how the second "
    "half changes' is fine; 'Examine the structural shift at 1:20' is not unless a "
    "transition was detected there. Mark a next action complete only when the user "
    "explicitly says they completed, finished, chose, or performed it, or when you "
    "performed it yourself with a tool call in the current turn. Never mark one "
    "complete because the user chose a related decision, the conversation moved on, "
    "or you assume it was handled. Remove one only when it is clearly superseded or "
    "obsolete. "
    "Pass update_project_state fields as top-level arguments; never nest them inside "
    "another object or key. "
    "focus_audio_region means 'look here right now'; update_project_state means "
    "'remember this as part of our production plan'. "
    "Findings marked as from an earlier upload describe a previous audio file; do not "
    "apply their timestamps to the current audio without re-analyzing it.",
])
MAX_TOOL_ROUNDS = 8

# Each model request gets at most this many attempts (backoff 1s, 2s, plus jitter).
MODEL_MAX_ATTEMPTS = 3
MODEL_BACKOFF_SECONDS = 1.0
RETRYABLE_MODEL_STATUS = {429, 500, 502, 503, 504}
RETRYABLE_MODEL_ERRORS = tuple(
    error_type
    for error_type in (
        getattr(litellm, name, None)
        for name in (
            "RateLimitError",
            "ServiceUnavailableError",
            "InternalServerError",
            "BadGatewayError",
            "APIConnectionError",
            "Timeout",
        )
    )
    if isinstance(error_type, type)
)


class ModelCallError(Exception):
    def __init__(self, cause: Exception, attempts: int):
        super().__init__(f"{type(cause).__name__}: {str(cause)[:300]}")
        self.attempts = attempts


def _is_retryable_model_error(exc: Exception) -> bool:
    if RETRYABLE_MODEL_ERRORS and isinstance(exc, RETRYABLE_MODEL_ERRORS):
        return True
    return getattr(exc, "status_code", None) in RETRYABLE_MODEL_STATUS


def _complete_with_retry(**kwargs):
    """One model request, retried only for transient provider failures.

    Retrying here is safe: tools run only after a reply has been received.
    """
    for attempt in range(1, MODEL_MAX_ATTEMPTS + 1):
        try:
            return litellm.completion(**kwargs)
        except Exception as exc:
            if attempt == MODEL_MAX_ATTEMPTS or not _is_retryable_model_error(exc):
                raise ModelCallError(exc, attempt) from exc
            time.sleep(MODEL_BACKOFF_SECONDS * 2 ** (attempt - 1) + random.uniform(0, 0.3))

# --- The Harness ---


def run_agent(messages: list[dict], context: dict | None = None) -> tuple[str, list[dict]]:
    """Complete until the model answers without asking for a tool.

    Returns the final text and a record of every tool call made along the way.
    context carries the session's project state and attached audio_file_id.
    """
    tool_calls = []

    for _ in range(MAX_TOOL_ROUNDS):
        model_messages = messages

        # Rebuilt every round so the model sees plan updates made earlier in this
        # turn; never stored in the session history.
        if context is not None:
            project_context = format_project_context(
                context["project_state"],
                context.get("audio_file_id"),
            )
            model_messages = [
                {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{project_context}"},
                *messages[1:],
            ]

        try:
            reply = _complete_with_retry(
                model="vertex_ai/gemini-3.5-flash-lite",
                vertex_location="global",
                messages=model_messages,
                tools=TOOLS,
            ).choices[0].message
        except ModelCallError as exc:
            # Keep tool calls that already ran this turn so the UI still reflects them.
            attempts = f"{exc.attempts} attempt{'s' if exc.attempts != 1 else ''}"
            return f"Model call failed after {attempts}: {exc}", tool_calls

        # Append assistant's reply (text, tool calls, or both) to the context.
        # model_dump() keeps it a plain dict: the raw object carries provider-specific
        # fields that trip Pydantic when LiteLLM re-serializes it next round.
        messages += [reply.model_dump()]

        if not reply.tool_calls:
            return reply.content, tool_calls

        # The harness, not the model, runs each tool and appends the result
        for call in reply.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
                result = json.dumps({"error": "Tool arguments were not valid JSON."})
            else:
                result = run_tool(call.function.name, args, context)

            if context is not None:
                record_tool_result(context["project_state"], call.function.name, args, result)

            tool_calls += [{"name": call.function.name, "args": args, "result": result}]

            messages += [{"role": "tool", "tool_call_id": call.id, "content": result}]

    return "Sorry, I hit my tool-call limit before finishing.", tool_calls


# --- Session Store ---

# session_id -> list of messages. In-memory, single process.
sessions: dict[str, list] = {}

# session_id -> Production Plan state (see project_state.py). Same lifetime as sessions.
project_states: dict[str, dict] = {}

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    audio_file_id: str | None = None
    selected_start_seconds: float | None = None
    selected_end_seconds: float | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


class NextActionStatusRequest(BaseModel):
    status: str


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "index.html")

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    original_filename = file.filename or ""

    suffix = Path(
        original_filename
    ).suffix.lower()

    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Use WAV, MP3, FLAC, or OGG."
            ),
        )

    audio_file_id = (
        f"{uuid.uuid4().hex}{suffix}"
    )

    destination = (
        AUDIO_UPLOAD_DIR / audio_file_id
    )

    total_bytes = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_AUDIO_UPLOAD_BYTES:
                    output.close()
                    destination.unlink(
                        missing_ok=True
                    )

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Audio file is too large. "
                            "Maximum upload size is 25 MB."
                        ),
                    )

                output.write(chunk)

    finally:
        await file.close()

    return {
        "audio_file_id": audio_file_id,
        "filename": original_filename,
        "size_bytes": total_bytes,
    }


@app.get("/spectrogram/{audio_file_id}")
def spectrogram(audio_file_id: str):
    try:
        image_path = render_spectrogram_png(audio_file_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Spectrogram generation failed: {type(exc).__name__}",
        )

    return FileResponse(
        image_path,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create the session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    project_state = project_states.setdefault(session_id, new_project_state())

    user_content = request.message
    safe_audio_file_id = None

    if request.audio_file_id:
        safe_audio_file_id = Path(
            request.audio_file_id
        ).name

        user_content += (
            "\n\n"
            "[Attached audio context: "
            f"The user has attached audio_file_id "
            f"'{safe_audio_file_id}'. "
            "When the user's request requires analyzing the "
            "actual audio, call analyze_audio_track using "
            "this exact audio_file_id.]"
        )

    selected_start = request.selected_start_seconds
    selected_end = request.selected_end_seconds

    if (
        request.audio_file_id
        and selected_start is not None
        and selected_end is not None
        and math.isfinite(selected_start)
        and math.isfinite(selected_end)
        and 0 <= selected_start < selected_end
    ):
        user_content += (
            "\n\n"
            "[Workspace context: The user currently has audio selected "
            f"from {selected_start:.1f}s to {selected_end:.1f}s in Track View. "
            'If the user refers to "this section", "this part", '
            '"the selected region", etc., use this exact range.]'
        )

    sessions[session_id] += [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    context = {
        "session_id": session_id,
        "project_state": project_state,
        "audio_file_id": safe_audio_file_id,
        "turn": new_turn_context(request.message),
    }

    try:
        response, tool_calls = run_agent(sessions[session_id], context)
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []

    return ChatResponse(response=response or "", session_id=session_id, tool_calls=tool_calls)


@app.get("/project-state/{session_id}")
def get_project_state(session_id: str):
    state = project_states.get(session_id)
    return public_project_state(state if state is not None else new_project_state())


@app.post("/project-state/{session_id}/next-actions/{action_id}")
def update_next_action(session_id: str, action_id: str, request: NextActionStatusRequest):
    state = project_states.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown session.")

    if request.status not in ("pending", "done"):
        raise HTTPException(status_code=400, detail="status must be 'pending' or 'done'.")

    if not set_next_action_status(state, action_id, request.status):
        raise HTTPException(status_code=404, detail="Unknown next action.")

    return public_project_state(state)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    project_states.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
