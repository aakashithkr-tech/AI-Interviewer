require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const path = require('path');
const { v4: uuidv4 } = require('uuid');

const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// Serve the frontend directly, so http://localhost:3000 shows the UI
// (frontend/index.html lives one level up from backend/server.js)
app.use(express.static(path.join(__dirname, '../frontend')));

const PORT = process.env.PORT || 3000;

// ---------- Gemini multi-key fallback ----------
// GEMINI_API_KEYS="key1,key2,key3"  (or single GEMINI_API_KEY)
const RAW_KEYS = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
const API_KEYS = RAW_KEYS.split(',').map(k => k.trim()).filter(Boolean);
// Try multiple models in order — if one model's quota is exhausted (common on
// free tier), fall back to the next. Override with GEMINI_MODELS in .env
// (comma-separated) if you want a different order.
const RAW_MODELS = process.env.GEMINI_MODELS || process.env.GEMINI_MODEL || 'gemini-2.0-flash,gemini-2.0-flash-lite';
const MODELS = RAW_MODELS.split(',').map(m => m.trim()).filter(Boolean);

if (API_KEYS.length === 0) {
  console.warn('[WARN] No GEMINI_API_KEY / GEMINI_API_KEYS set. /api/interview calls will fail until you add one.');
}

async function callGemini(systemPrompt, userPrompt, { json = true } = {}) {
  let lastErr = null;
  for (const model of MODELS) {
    for (let i = 0; i < Math.max(API_KEYS.length, 1); i++) {
      const key = API_KEYS[i];
      if (!key) continue;
      try {
        // Google migrated to "Auth" keys (prefix "AQ.") in 2026, which must be sent
        // via the X-goog-api-key header rather than the old ?key= query param.
        // The header works for legacy "AIzaSy..." Standard keys too, so this is safe either way.
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
        const body = {
          contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
          systemInstruction: { role: 'system', parts: [{ text: systemPrompt }] },
          generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 1024,
            ...(json ? { responseMimeType: 'application/json' } : {})
          }
        };
        const res = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-goog-api-key': key
          },
          body: JSON.stringify(body)
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`[${model}] key #${i + 1} failed: ${res.status} ${text.slice(0, 200)}`);
        }
        const data = await res.json();
        const text = data?.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || '';
        if (!text) throw new Error(`[${model}] key #${i + 1} returned empty content`);
        return text;
      } catch (err) {
        console.error(err.message);
        lastErr = err;
        // try next key, then next model
      }
    }
  }
  throw lastErr || new Error('All Gemini keys/models failed');
}

// ---------- Groq fallback (free, instant key, no billing needed) ----------
// If ALL Gemini keys/models fail (e.g. Google free-tier quota issues), fall
// back to Groq so the demo never fully breaks. Get a free key instantly at
// https://console.groq.com/keys — no credit card required.
const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';

async function callGroq(systemPrompt, userPrompt, { json = true } = {}) {
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${GROQ_API_KEY}`
    },
    body: JSON.stringify({
      model: GROQ_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      temperature: 0.7,
      max_tokens: 1024,
      ...(json ? { response_format: { type: 'json_object' } } : {})
    })
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Groq failed: ${res.status} ${text.slice(0, 200)}`);
  }
  const data = await res.json();
  const text = data?.choices?.[0]?.message?.content || '';
  if (!text) throw new Error('Groq returned empty content');
  return text;
}

// ---------- unified LLM call: Gemini first, Groq as last-resort fallback ----------
async function callLLM(systemPrompt, userPrompt, opts = {}) {
  try {
    return await callGemini(systemPrompt, userPrompt, opts);
  } catch (geminiErr) {
    if (!GROQ_API_KEY) throw geminiErr;
    console.error('[fallback] All Gemini options failed, trying Groq…');
    try {
      return await callGroq(systemPrompt, userPrompt, opts);
    } catch (groqErr) {
      console.error(groqErr.message);
      throw geminiErr; // surface the original (more informative) error
    }
  }
}

function safeParseJSON(text) {
  // strip accidental markdown fences just in case
  const cleaned = text.replace(/```json/gi, '').replace(/```/g, '').trim();
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    console.error('JSON parse failed, raw text:', cleaned.slice(0, 300));
    return null;
  }
}

// ---------- In-memory session store ----------
// sessions: Map<sessionId, SessionState>
const sessions = new Map();

const MAX_QUESTIONS = 8;

const PERSONAS = {
  friendly: 'Friendly and encouraging. Use warm phrasing, gentle nudges, light positive reinforcement ("Nice, can you go a little deeper?").',
  strict: 'Strict and terse. No pleasantries. Point out gaps directly ("That is incomplete. Give an example.").',
  faang: 'FAANG-bar-raiser style. Frame questions as system design / real-world scenarios ("Imagine you are designing X. Continue."). High expectations, probing follow-ups.'
};

function buildSystemPrompt(persona, topic) {
  return `You are an adaptive AI technical interviewer conducting a live spoken interview on the topic: "${topic}".

Persona / tone: ${PERSONAS[persona] || PERSONAS.friendly}

Your job every turn:
1. Judge the candidate's latest answer for CORRECTNESS (correct / partial / incorrect) and COMPETENCE (0-100).
2. Separately judge CONFIDENCE (0-100) from the way they spoke — hedging words like "umm", "I think", "maybe", "not sure" lower confidence even if the content is correct; assertive phrasing raises it even if the content is wrong. Confidence and competence are independent signals — call out mismatches (e.g. low confidence + correct answer, or high confidence + wrong answer).
3. Adapt difficulty: if the answer is strong, ask a harder follow-up on the SAME thread (dig deeper, don't just jump topics). If the answer is weak, step back to a simpler foundational question on the same concept before moving on. Track difficulty on a 1-5 scale.
4. Give evidence-based feedback: quote or closely paraphrase a short specific piece of what the candidate said and explain what it reveals. Never give generic feedback like "weak in X" without pointing to the specific statement that shows it.
5. Decide the next question. Prefer follow-ups that build a thread (e.g. embeddings -> vector DB -> RAG -> full pipeline) over jumping randomly.
6. After roughly ${MAX_QUESTIONS} questions, or if the topic has been thoroughly covered, set "done": true and ask no further question.

Always reply with STRICT JSON only, no markdown, matching this schema exactly:
{
  "verdict": "correct" | "partial" | "incorrect",
  "competence": <0-100 integer>,
  "confidence": <0-100 integer>,
  "evidenceFeedback": "<one or two sentences, quoting/paraphrasing the candidate's own words>",
  "nextDifficulty": <1-5 integer>,
  "nextQuestion": "<the next interview question, in the persona's tone, or empty string if done>",
  "topicTag": "<short tag for the concept just tested, e.g. 'embeddings', 'chunking-strategy', 'vector-db'>",
  "done": <true|false>
}`;
}

function buildFirstQuestionPrompt(persona, topic) {
  return `Generate the FIRST interview question for a live spoken technical interview on "${topic}". Start at a medium-easy difficulty (level 2 of 5) to establish a baseline. Reply with STRICT JSON only:
{
  "nextQuestion": "<the opening question, in tone: ${PERSONAS[persona] || PERSONAS.friendly}>",
  "nextDifficulty": 2,
  "topicTag": "<short tag>"
}`;
}

// ---------- Routes ----------

app.get('/api/health', (req, res) => {
  res.json({ ok: true, models: MODELS, keysConfigured: API_KEYS.length, groqFallback: !!GROQ_API_KEY });
});

app.post('/api/interview/start', async (req, res) => {
  try {
    const { topic = 'Machine Learning & AI Systems', persona = 'friendly' } = req.body || {};
    const sessionId = uuidv4();

    const raw = await callLLM(
      'You are an interview question generator. Reply with strict JSON only.',
      buildFirstQuestionPrompt(persona, topic)
    );
    const parsed = safeParseJSON(raw) || {
      nextQuestion: `Let's start with the basics — can you explain what ${topic} means in your own words?`,
      nextDifficulty: 2,
      topicTag: 'intro'
    };

    const session = {
      id: sessionId,
      topic,
      persona,
      difficulty: parsed.nextDifficulty || 2,
      turns: [],
      createdAt: Date.now(),
      lastActive: Date.now(),
      done: false,
      _pendingQuestion: parsed.nextQuestion
    };
    sessions.set(sessionId, session);

    res.json({
      sessionId,
      question: parsed.nextQuestion,
      difficulty: session.difficulty,
      questionNumber: 1,
      maxQuestions: MAX_QUESTIONS,
      done: false
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to start interview', detail: err.message });
  }
});

app.post('/api/interview/answer', async (req, res) => {
  try {
    const { sessionId, answer } = req.body || {};
    const session = sessions.get(sessionId);
    if (!session) return res.status(404).json({ error: 'Session not found' });
    if (session.done) return res.status(400).json({ error: 'Interview already complete', done: true });
    if (!answer || !answer.trim()) return res.status(400).json({ error: 'Empty answer' });

    const questionNumber = session.turns.length + 1;

    // Build conversation transcript for context
    const transcript = session.turns.map((t, i) =>
      `Q${i + 1} (difficulty ${t.difficulty}): ${t.question}\nCandidate: ${t.answer}\nVerdict: ${t.verdict}, competence ${t.competence}, confidence ${t.confidence}`
    ).join('\n\n');

    const userPrompt = `Topic: ${session.topic}
Questions asked so far: ${questionNumber} of max ${MAX_QUESTIONS}
Current difficulty: ${session.difficulty}

Transcript so far:
${transcript || '(none yet)'}

Current question asked: ${session._pendingQuestion}
Candidate's spoken answer (transcribed): "${answer}"

Evaluate this answer and produce the next step, per the schema.${questionNumber >= MAX_QUESTIONS ? ' This is the FINAL question — set done: true and nextQuestion to "".' : ''}`;

    const raw = await callLLM(buildSystemPrompt(session.persona, session.topic), userPrompt);
    const parsed = safeParseJSON(raw);

    if (!parsed) {
      return res.status(502).json({ error: 'Model returned unparsable response' });
    }

    const turn = {
      qNum: questionNumber,
      question: session._pendingQuestion,
      answer,
      verdict: parsed.verdict || 'partial',
      competence: clamp(parsed.competence, 0, 100, 50),
      confidence: clamp(parsed.confidence, 0, 100, 50),
      evidenceFeedback: parsed.evidenceFeedback || '',
      difficulty: session.difficulty,
      topicTag: parsed.topicTag || 'general',
      timestamp: Date.now()
    };
    session.turns.push(turn);
    session.lastActive = Date.now();

    const isDone = !!parsed.done || questionNumber >= MAX_QUESTIONS;
    session.done = isDone;
    session.difficulty = clamp(parsed.nextDifficulty, 1, 5, session.difficulty);
    session._pendingQuestion = isDone ? null : (parsed.nextQuestion || '');

    res.json({
      sessionId,
      verdict: turn.verdict,
      competence: turn.competence,
      confidence: turn.confidence,
      evidenceFeedback: turn.evidenceFeedback,
      difficulty: session.difficulty,
      nextQuestion: session._pendingQuestion,
      questionNumber: questionNumber + (isDone ? 0 : 1),
      maxQuestions: MAX_QUESTIONS,
      done: isDone
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to process answer', detail: err.message });
  }
});

app.get('/api/interview/:sessionId/report', async (req, res) => {
  try {
    const session = sessions.get(req.params.sessionId);
    if (!session) return res.status(404).json({ error: 'Session not found' });
    if (session.turns.length === 0) return res.status(400).json({ error: 'No turns recorded yet' });

    const turns = session.turns;
    const avg = (arr) => Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);

    const competence = avg(turns.map(t => t.competence));
    const confidence = avg(turns.map(t => t.confidence));
    const problemSolving = Math.round(
      (turns.reduce((sum, t) => sum + t.difficulty * (t.verdict === 'correct' ? 1 : t.verdict === 'partial' ? 0.5 : 0), 0) / (turns.length * 5)) * 100
    );
    const communication = Math.round((confidence * 0.6 + competence * 0.4));
    const hiringProbability = Math.round((competence * 0.45 + confidence * 0.15 + problemSolving * 0.4));

    let recommendation = 'No Hire';
    if (hiringProbability >= 80) recommendation = 'Strong Hire';
    else if (hiringProbability >= 65) recommendation = 'Hire';
    else if (hiringProbability >= 45) recommendation = 'Leaning No Hire';

    const weakTopics = turns.filter(t => t.verdict !== 'correct').map(t => t.topicTag);
    const uniqueWeak = [...new Set(weakTopics)];

    const timeline = turns.map(t => ({
      qNum: t.qNum,
      question: t.question,
      verdict: t.verdict,
      topicTag: t.topicTag,
      difficulty: t.difficulty,
      evidenceFeedback: t.evidenceFeedback
    }));

    res.json({
      sessionId: session.id,
      topic: session.topic,
      persona: session.persona,
      scores: { hiringProbability, confidence, communication, problemSolving, competence },
      recommendation,
      timeline,
      revisionPlan: uniqueWeak.map(tag => ({ topic: tag, estimatedMinutes: 15 })),
      totalEstimatedMinutes: uniqueWeak.length * 15
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to build report', detail: err.message });
  }
});

function clamp(val, min, max, fallback) {
  const n = Number(val);
  if (Number.isNaN(n)) return fallback;
  return Math.max(min, Math.min(max, Math.round(n)));
}

app.listen(PORT, () => {
  console.log(`Interview agent backend listening on port ${PORT}`);
  console.log(`Models (fallback order): ${MODELS.join(' -> ')} | Keys configured: ${API_KEYS.length}`);
});