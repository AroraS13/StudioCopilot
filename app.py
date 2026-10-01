import json
import uuid
from pathlib import Path

import litellm
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from tools import AUDIO_UPLOAD_DIR, TOOLS, run_tool

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
    "Do not invent arrangement timestamps yourself when this tool is appropriate."
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
    )
MAX_TOOL_ROUNDS = 5

# --- The Harness ---


def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """Complete until the model answers without asking for a tool.

    Returns the final text and a record of every tool call made along the way.
    """
    tool_calls = []

    for _ in range(MAX_TOOL_ROUNDS):
        reply = litellm.completion(
            model="vertex_ai/gemini-3.5-flash-lite",
            vertex_location="global",
            messages=messages,
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
            args = json.loads(call.function.arguments)
            result = run_tool(call.function.name, args)
            tool_calls += [{"name": call.function.name, "args": args, "result": result}]

            messages += [{"role": "tool", "tool_call_id": call.id, "content": result}]

    return "Sorry, I hit my tool-call limit before finishing.", tool_calls


# --- Session Store ---

# session_id -> list of messages. In-memory, single process.
sessions: dict[str, list] = {}

# --- FastAPI App ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    audio_file_id: str | None = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]


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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Get or create the session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    user_content = request.message

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

    sessions[session_id] += [
        {
            "role": "user",
            "content": user_content,
        }
    ]

    try:
        response, tool_calls = run_agent(sessions[session_id])
    except Exception as e:
        # Auth, billing, a model that is not running: show it in the chat, not as a 500.
        response, tool_calls = f"Model call failed: {type(e).__name__}: {str(e)[:300]}", []

    return ChatResponse(response=response, session_id=session_id, tool_calls=tool_calls)


@app.post("/clear")
def clear(session_id: str | None = None):
    sessions.pop(session_id, None)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
