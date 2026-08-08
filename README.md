# Adaptive Voice Interview Agent (PS2)

Full-voice, adaptive-difficulty AI interview agent. Backend: Express + Gemini
(with multi-key fallback). Frontend: single-file voice-first UI using the
browser's built-in Web Speech API (Chrome only) — zero cost, zero rate
limits, nothing to run out of mid-demo.

## Structure
```
backend/    Express server, Gemini calls, session + report logic
frontend/   index.html — voice UI (works standalone, just needs backend URL)
```

## Run locally

```bash
cd backend
cp .env.example .env      # fill in GEMINI_API_KEYS=key1,key2,key3
npm install
npm start                 # listens on :3000
```

Open `frontend/index.html` directly in Chrome (double-click, or `open
frontend/index.html`), leave the "Backend URL" field as
`http://localhost:3000`, pick a topic + persona, hit **Start speaking
interview**, allow mic access.

## Deploy (Render / Railway, free tier)

1. Push `backend/` to a public GitHub repo (`.env` is already gitignored).
2. Render/Railway → New Web Service → connect repo → root dir `backend`.
3. Build command: `npm install`. Start command: `npm start`.
4. Set env vars in the dashboard: `GEMINI_API_KEYS` (comma-separated for
   fallback), optionally `GEMINI_MODEL`.
5. Once live, open `frontend/index.html`, set "Backend URL" to your deployed
   URL, and it just works — the frontend has no build step, host it anywhere
   (or even just open the file locally) as long as it can reach the backend.

## API

- `POST /api/interview/start` `{ topic, persona }` → `{ sessionId, question, difficulty, questionNumber, maxQuestions }`
- `POST /api/interview/answer` `{ sessionId, answer }` → verdict, competence, confidence, evidence-based feedback, next question, done flag
- `GET /api/interview/:sessionId/report` → hiring probability, confidence, communication, problem-solving scores, full timeline, personalized revision plan

## Notes

- Difficulty adapts per-answer (1–5) based on the model's own judgement of
  correctness — it digs deeper on strong answers, steps back on weak ones.
- Confidence is scored separately from competence (hedging language vs.
  assertive language), so a correct-but-hesitant answer and a
  wrong-but-confident answer get different, explicit feedback.
- Every feedback line is evidence-based: the model is instructed to quote or
  closely paraphrase what the candidate actually said, never generic
  "weak in X" feedback.
- Interview ends automatically after 8 questions (`MAX_QUESTIONS` in
  `server.js` — change freely).
