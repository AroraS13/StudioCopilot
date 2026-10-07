# Studio Copilot

Studio Copilot is a workspace for music producers with an AI agent built into it. I wanted to see whether an agent could actually be useful inside a production workflow instead of just sitting in a chat window answering questions.

You can upload a track, view it as a waveform or spectrogram, select a region, and ask questions about it. The agent can analyze audio, figure out how to retime or repitch a sample, build an arrangement, look up reference tracks, focus on specific parts of a song, and keep track of goals and decisions throughout the session.

The backend is built with FastAPI and Gemini through Vertex AI. Audio analysis is done with `librosa`, and reference track information comes from the MusicBrainz API.

## Tools

The agent has six tools:

- `analyze_audio_track` analyzes a whole track or selected region and estimates tempo, key, signal level, rhythmic activity, spectral characteristics, and possible change points.
- `transform_sample` calculates the tempo, duration, and pitch changes needed to move a sample from one BPM/key to another.
- `build_arrangement` creates a bar-accurate arrangement based on BPM, target length, and the sections you want.
- `lookup_reference_track` looks up metadata and duration for a reference track using MusicBrainz.
- `focus_audio_region` lets the agent highlight a specific part of the track in the UI.
- `update_project_state` keeps track of goals, findings, decisions, and next steps in the Production Plan.

All of the tools were built for this project. `lookup_reference_track` uses MusicBrainz as its external data source.

## Interface

The starter project was basically just a chat box, so I rebuilt the frontend to make the track the main part of the workspace.

The UI includes a waveform/spectrogram view, draggable region selection, the chat, an Agent Activity panel showing tool calls and outputs, and a Production Plan that keeps track of the current session.

## Running locally

You'll need Python 3.10+, `uv`, and a Google Cloud project with Vertex AI enabled.

Install dependencies with `uv sync`.

Log in with Google Cloud Application Default Credentials using `gcloud auth application-default login`.

If needed, set your Google Cloud project with `gcloud config set project YOUR_PROJECT_ID`.

Start the app with `uv run app.py`, then open `http://127.0.0.1:8000/`.

Uploads can be WAV, MP3, FLAC, or OGG, up to 25 MB.

## Things to try

After uploading a track:

Analyze this track. What are the 3 most important things I should know about it as a producer?

To focus on one part of a track:

Focus on the section from 1:00 to 1:20 and tell me what changes there compared with the rest of the track.

To transform a sample:

I have an 8-bar guitar loop at 92 BPM in D minor. I want to use it in a 124 BPM track in F minor. How should I transform it?

To build an arrangement:

Build me a 3:15 arrangement at 124 BPM with an intro, verse, buildup, drop, breakdown, final drop, and outro.

To use a reference track:

Look up "Nights" by Frank Ocean and use it as a reference track.

To try the Production Plan:

I want the second half to hit harder without making the track longer.

Then follow up with:

Let's use an 8-bar build before the second half.

## Notes

The audio analysis is meant to stay fairly conservative. RMS is used to compare levels within a track rather than as a replacement for LUFS, and spectral centroid is treated as a rough brightness measure rather than a bass measurement. Detected change points are not automatically labeled as verses, choruses, drops, etc.

Tempo and key are algorithmic estimates, so they can be wrong on rhythmically or harmonically ambiguous material.

Sessions and uploaded audio currently live in memory and temporary storage, so restarting the server clears them.

Studio Copilot will be deployed on Google Cloud Run.