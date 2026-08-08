# PROMPTS.md

Log of key prompts used while building this project (per submission
requirement — keep appending as you go, don't reconstruct at the end).

## 1. Architecture decision
Asked for a comparison of PS2 vs PS3 on pure win-probability, weighing
judging criteria — landed on PS2 for lower failure-mode surface.

## 2. Voice approach
Asked for the most reliable voice option for a 48h hackathon with no infra
risk — landed on the browser's Web Speech API (client-side STT via
`SpeechRecognition`, TTS via `speechSynthesis`), keeping Gemini scoped to
text-in/text-out interview reasoning only.

## 3. Backend scaffold
Generated Express skeleton with a dummy `/api/interview` endpoint first, to
deploy immediately and de-risk the deployment pipeline before writing real
logic.

## 4. Full build
Generated the adaptive interview engine: session state, Gemini system
prompt enforcing structured JSON output (verdict, competence, confidence,
evidence-based feedback, next difficulty, next question, done flag),
multi-key fallback for Gemini calls, and the `/report` endpoint that
computes hiring probability / confidence / communication / problem-solving
scores and a personalized revision plan from the session transcript.
Generated the voice-first frontend: setup → live interview room (orb,
difficulty ladder, timeline) → report screen, wired to STT/TTS and the
backend API.

---
*(continue logging notable prompts here as the project evolves)*
