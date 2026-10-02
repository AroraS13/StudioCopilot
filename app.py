import json
import math
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

SYSTEM_PROMPT = (
    "You are Studio Copilot, a practical assistant for musicians and music producers. "
    "Give concrete, production-oriented advice rather than generic encouragement. "
    "When the user wants to adapt a sample or loop between tempos or keys, call "
    "transform_sample before giving numerical time-stretch or pitch-shift advice. "
    "Explain tool results in normal producer language and mention relevant caveats, "
    "such as artifacts from extreme stretching or a major/minor mode mismatch. "
    "You also have a build_arrangement tool. Use it whenever the user asks you to "
    "plan or restructure a song timeline around a BPM and target duration. "
    "Decide on an appropriate ordered section list from the user's request, then "
    "let the tool calculate bar counts and timestamps. "
    "Do not invent arrangement timestamps yourself when this tool is appropriate. "
    "When the user names an existing song as a reference and factual metadata such "
    "as its duration is needed, call lookup_reference_track instead of guessing. "
    "When the user names an existing song as a reference and factual metadata such "
    "as its duration is needed, call lookup_reference_track instead of guessing. "
    "If the user wants to create an arrangement based on the duration of a reference "
    "track, first call lookup_reference_track, then use the returned duration as the "
    "target_duration_seconds for build_arrangement. "
    "If a reference-track lookup fails or is ambiguous, explain that rather than "
    "inventing metadata. "
    "When discussing an existing reference track, only state factual details that "
    "were explicitly returned by lookup_reference_track or explicitly provided by "
    "the user. Treat any other track-specific fact as unknown, even if you believe "
    "you know it from prior knowledge. "
    "Do not use pretrained knowledge to claim a track's BPM, key, chord progression, "
    "section structure, beat switches, production techniques, instrumentation, or "
    "timestamps unless those details were returned by a tool or given by the user. "
    "Do not infer structural details from the track title, artist, album, or duration. "
    "You may give general production suggestions inspired by the user's goals, but "
    "clearly present them as suggestions for the user's track rather than facts about "
    "the reference recording. "
    "When the user asks you to analyze, diagnose, compare, or make claims about "
    "the actual sound of an attached audio file, call analyze_audio_track before "
    "answering. Do not pretend that you directly listened to the audio. "
    "Base audio-specific claims on the measurements returned by the tool. "
    "Treat estimated BPM, key, and structural boundaries as estimates rather than "
    "ground truth. RMS measurements are not LUFS loudness measurements. "
    "If the user gives timestamps for a specific section, analyze that region rather "
    "than assuming which part of the track is the verse, chorus, drop, or bridge. "
    "When interpreting analyze_audio_track results, clearly distinguish measured "
    "audio features from musical interpretation and creative production suggestions. "
    "Do not label regions as an intro, verse, chorus, drop, breakdown, bridge, climax, "
    "or outro unless the user explicitly identifies those sections. "
    "Do not infer specific low-end balance, stereo or mono compatibility, masking, "
    "compression needs, instrumentation, or EQ problems from RMS, onset density, "
    "or spectral centroid alone. "
    "Production suggestions may go beyond the measurements, but present them as "
    "possible experiments rather than problems proven by the analysis. "
    "If the same audio_file_id and analysis range were already analyzed earlier in "
    "the conversation, reuse those existing results when they are sufficient. "
    "Only call analyze_audio_track again when analyzing a different file or time range, "
    "or when new measurements are actually required. "
    "analyze_audio_track performs actual audio measurements. "
    "Interpret colon-formatted timestamps as MM:SS unless context clearly indicates "
    "otherwise: 0:45 is 45 seconds, 1:10 is 70 seconds, 10:00 is 600 seconds, and "
    "10:30 is 630 seconds. Convert them to seconds before calling a tool. "
    "If a tool reports that a requested range is beyond the end of the track, tell "
    "the user and ask for a valid range. Never silently analyze or highlight a "
    "different range in its place, and never claim a substitution happened. "
    "The user works in a Track View that shows their uploaded audio as a waveform or "
    "spectrogram. The user can drag to select a region there. When a workspace "
    "context note gives selected timestamps and the user refers to 'this section', "
    "'this part', 'the selection', or similar, use exactly those timestamps; if the "
    "request needs measurements of that region, call analyze_audio_track with those "
    "exact start_seconds and end_seconds. "
    "You have a focus_audio_region tool that visually highlights a timestamped region "
    "in the user's Track View. It changes visual focus only and does not analyze "
    "audio. Use it when the user asks you to show, highlight, locate, or focus on a "
    "specific audio region, or when directing visual attention to an exact analyzed "
    "region would materially improve your explanation. Base its range on tool "
    "measurements or user-provided timestamps. Do not call it on every response. "
    "Do not infer that individual instruments enter or drop out from aggregate DSP "
    "features such as RMS, onset density, or spectral centroid. "
    "If the user asks about an exact timestamp range and no earlier analysis covers "
    "exactly that range, call analyze_audio_track on the requested range; the coarse "
    "energy_profile segments of a full-track analysis are not sufficient for a "
    "precise timestamp comparison. "
    "Studio Copilot maintains a persistent Production Plan for this session, shown "
    "to the user in the workspace: a goal, key findings, decisions, and next actions, "
    "plus the active reference track and arrangement. Its current contents are given "
    "to you as project state below. Use it to continue the user's work without asking "
    "them to restate what was already established. "
    "Use update_project_state only for durable changes: when the user states or "
    "changes their goal, when analysis produces a genuinely useful finding, when the "
    "user makes an actual decision, when concrete next actions are agreed, or when a "
    "next action is completed or becomes obsolete. Do not call it merely because you "
    "responded, and do not save every observation. "
    "Goals reflect the user's stated intent, never your guess. Findings must be "
    "supported by analyze_audio_track measurements or explicit user statements; use "
    "the exact analyzed timestamps for location-based findings and never turn "
    "speculation into a finding. Your suggestions are not decisions; decisions are "
    "choices the user made. Next actions are concrete suggested steps (usually 2 to 5) "
    "and are not decisions or completed work unless marked done. "
    "focus_audio_region means 'look here right now' (temporary visual attention); "
    "update_project_state means 'remember this as part of our production plan' "
    "(durable project memory). They serve different purposes. "
    "Findings marked as from an earlier upload describe a previous audio file; do "
    "not apply their timestamps to the current audio without re-analyzing it. "
    )
MAX_TOOL_ROUNDS = 8

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

        reply = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=model_messages,
            tools=TOOLS,
        ).choices[0].message

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
