require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// Serve the frontend directly, so http://localhost:3000 shows the UI
// (frontend/index.html lives one level up from backend/server.js)
app.use(express.static(path.join(__dirname, '../frontend')));

const PORT = process.env.PORT || 3000;

// ---------- Gemini multi-key fallback ----------
// GEMINI_API_KEYS="key1,key2,key3"  (or single GEMINI_API_KEY)
const RAW_KEYS = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
const API_KEYS = RAW_KEYS.split(',').map(k => k.trim()).filter(Boolean);
// Try multiple models in order — if one model's quota is exhausted (common on
// free tier), fall back to the next. gemini-2.0-flash-lite goes FIRST now:
// it's the lower-latency model and the live per-turn loop (the thing the
// candidate is actually waiting on) cares about speed far more than the
// marginal quality gain of full "flash". Override with GEMINI_MODELS in
// .env (comma-separated) if you want a different order.
const RAW_MODELS = process.env.GEMINI_MODELS || process.env.GEMINI_MODEL || 'gemini-2.0-flash-lite,gemini-2.0-flash';
const MODELS = RAW_MODELS.split(',').map(m => m.trim()).filter(Boolean);
const EMBED_MODEL = process.env.GEMINI_EMBED_MODEL || 'text-embedding-004';

if (API_KEYS.length === 0) {
  console.warn('[WARN] No GEMINI_API_KEY / GEMINI_API_KEYS set. /api/interview calls will fail until you add one.');
}

// Guards every outbound LLM call with a hard timeout — without this, a single
// hung/slow key or model attempt can block the whole request indefinitely,
// which is what was surfacing to the frontend as a generic "could not reach
// backend" (the connection just never resolved either way).
async function fetchWithTimeout(url, options, timeoutMs = 12000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === 'AbortError') throw new Error(`timed out after ${timeoutMs}ms`);
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

// Per-attempt timeout AND a global wall-clock budget for the whole
// model x key cascade. Previously only the per-attempt timeout existed
// (7000ms) — with 2 models x N keys, a bad key/model combo could still eat
// 7s * 2 * N before ever reaching Groq. That's the single biggest
// contributor to "next question feels slow": every one of those seconds is
// dead air after the candidate already stopped talking. Now the whole
// Gemini cascade is capped at GEMINI_BUDGET_MS (default 4500ms) — as soon as
// that's exceeded we stop trying more models/keys and fail over to Groq
// (or the raw error) immediately, instead of grinding through every
// remaining combination.
async function callGemini(systemPrompt, userPrompt, { json = true, maxOutputTokens = 1024, attemptTimeoutMs = 5000, budgetMs = 4500 } = {}) {
  let lastErr = null;
  const cascadeStart = Date.now();
  for (const model of MODELS) {
    for (let i = 0; i < Math.max(API_KEYS.length, 1); i++) {
      if (Date.now() - cascadeStart > budgetMs) {
        console.warn(`[latency] gemini cascade budget (${budgetMs}ms) exceeded — bailing to Groq/fallback`);
        throw lastErr || new Error('Gemini cascade exceeded latency budget');
      }
      const key = API_KEYS[i];
      if (!key) continue;
      const attemptStart = Date.now();
      try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
        const body = {
          contents: [{ role: 'user', parts: [{ text: userPrompt }] }],
          systemInstruction: { role: 'system', parts: [{ text: systemPrompt }] },
          generationConfig: {
            temperature: 0.7,
            maxOutputTokens,
            ...(json ? { responseMimeType: 'application/json' } : {})
          }
        };
        // Remaining budget also caps this specific attempt, so the last
        // attempt before the budget expires can't itself blow the budget.
        const remainingBudget = budgetMs - (Date.now() - cascadeStart);
        const thisAttemptTimeout = Math.max(1000, Math.min(attemptTimeoutMs, remainingBudget));
        const res = await fetchWithTimeout(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-goog-api-key': key
          },
          body: JSON.stringify(body)
        }, thisAttemptTimeout);
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`[${model}] key #${i + 1} failed: ${res.status} ${text.slice(0, 200)}`);
        }
        const data = await res.json();
        const text = data?.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || '';
        if (!text) throw new Error(`[${model}] key #${i + 1} returned empty content`);
        console.log(`[latency] gemini ${model} key#${i + 1} OK in ${Date.now() - attemptStart}ms`);
        return text;
      } catch (err) {
        console.error(`[latency] gemini ${model} key#${i + 1} FAILED in ${Date.now() - attemptStart}ms:`, err.message);
        lastErr = err;
      }
    }
  }
  throw lastErr || new Error('All Gemini keys/models failed');
}

const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const GROQ_MODEL = process.env.GROQ_MODEL || 'llama-3.3-70b-versatile';

async function callGroq(systemPrompt, userPrompt, { json = true, maxOutputTokens = 1024, attemptTimeoutMs = 5000 } = {}) {
  const attemptStart = Date.now();
  try {
    const res = await fetchWithTimeout('https://api.groq.com/openai/v1/chat/completions', {
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
        max_tokens: maxOutputTokens,
        ...(json ? { response_format: { type: 'json_object' } } : {})
      })
    }, attemptTimeoutMs);
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Groq failed: ${res.status} ${text.slice(0, 200)}`);
    }
    const data = await res.json();
    const text = data?.choices?.[0]?.message?.content || '';
    if (!text) throw new Error('Groq returned empty content');
    console.log(`[latency] groq OK in ${Date.now() - attemptStart}ms`);
    return text;
  } catch (err) {
    console.error(`[latency] groq FAILED in ${Date.now() - attemptStart}ms:`, err.message);
    throw err;
  }
}

async function callLLM(systemPrompt, userPrompt, opts = {}) {
  const start = Date.now();
  try {
    const result = await callGemini(systemPrompt, userPrompt, opts);
    console.log(`[latency] callLLM total (gemini path) ${Date.now() - start}ms`);
    return result;
  } catch (geminiErr) {
    if (!GROQ_API_KEY) throw geminiErr;
    console.error('[fallback] Gemini cascade failed/timed out, trying Groq…');
    try {
      const result = await callGroq(systemPrompt, userPrompt, opts);
      console.log(`[latency] callLLM total (groq fallback path) ${Date.now() - start}ms`);
      return result;
    } catch (groqErr) {
      throw geminiErr;
    }
  }
}

function safeParseJSON(text) {
  const cleaned = text.replace(/```json/gi, '').replace(/```/g, '').trim();
  try {
    return JSON.parse(cleaned);
  } catch (e) {
    console.error('JSON parse failed, raw text:', cleaned.slice(0, 300));
    return null;
  }
}

const DATA_DIR = path.join(__dirname, 'data');
const DATA_FILE = path.join(DATA_DIR, 'candidates.json');
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(DATA_FILE)) fs.writeFileSync(DATA_FILE, '{}');

function loadStore() {
  try { return JSON.parse(fs.readFileSync(DATA_FILE, 'utf8')); }
  catch (e) { console.error('[store] read failed, starting fresh:', e.message); return {}; }
}
function saveStore(store) {
  try { fs.writeFileSync(DATA_FILE, JSON.stringify(store, null, 2)); }
  catch (e) { console.error('[store] write failed:', e.message); }
}
function slugKey(name) {
  return (name || 'guest').trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '') || 'guest';
}

// Embeddings are only used for cross-day memory (once at /start) and for
// persisting concepts after the interview ends — never on the hot per-turn
// path. Still, they previously used bare `fetch` with no timeout, so a
// hung embedding call could stall /start indefinitely. Now timeout-guarded
// and short (3s) since a missed embedding just falls back to keyword
// overlap scoring, which is fine.
async function embedText(text) {
  if (!API_KEYS.length) return null;
  for (const key of API_KEYS) {
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:embedContent`;
      const res = await fetchWithTimeout(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-goog-api-key': key },
        body: JSON.stringify({ content: { parts: [{ text }] } })
      }, 3000);
      if (!res.ok) continue;
      const data = await res.json();
      const values = data?.embedding?.values;
      if (Array.isArray(values) && values.length) return values;
    } catch (e) { /* try next key */ }
  }
  return null;
}
function cosineSim(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
function keywordOverlapScore(a, b) {
  const wa = new Set(String(a).toLowerCase().split(/\W+/).filter(Boolean));
  const wb = new Set(String(b).toLowerCase().split(/\W+/).filter(Boolean));
  let hits = 0;
  for (const w of wa) if (wb.has(w)) hits++;
  return hits / Math.max(1, Math.min(wa.size, wb.size));
}

async function retrieveCrossDayMemory(candidateKey, topic, k = 3) {
  const store = loadStore();
  const candidate = store[candidateKey];
  if (!candidate || !candidate.sessions || !candidate.sessions.length) return { items: [], dayCount: 0 };

  const allConcepts = [];
  candidate.sessions.forEach((s, idx) => {
    (s.concepts || []).forEach(c => allConcepts.push({ ...c, day: idx + 1, sessionTopic: s.topic, date: s.date }));
  });
  if (!allConcepts.length) return { items: [], dayCount: candidate.sessions.length };

  const queryVec = await embedText(topic);
  let scored;
  if (queryVec) {
    scored = allConcepts.map(c => ({ c, score: cosineSim(queryVec, c.embedding) }));
  } else {
    scored = allConcepts.map(c => ({ c, score: keywordOverlapScore(topic, c.tag + ' ' + (c.summary || '')) }));
  }
  scored.sort((a, b) => b.score - a.score);
  const items = scored.slice(0, k).filter(s => s.score > 0.15).map(s => s.c);
  return { items, dayCount: candidate.sessions.length };
}

async function persistSession(candidateKey, session, report) {
  const store = loadStore();
  if (!store[candidateKey]) store[candidateKey] = { name: session.candidateName || candidateKey, sessions: [] };

  const byTag = new Map();
  session.turns.forEach(t => {
    if (!byTag.has(t.topicTag)) byTag.set(t.topicTag, []);
    byTag.get(t.topicTag).push(t);
  });
  // Concept embeddings only happen once, at the very end of the interview
  // (never per-turn), and this call is fire-and-forget from the /report
  // route (it doesn't block the response) — but there's no reason to make
  // the candidate's browser or the server wait on N sequential embedding
  // calls when they're all independent. Parallelized with Promise.all.
  const tags = [...byTag.entries()];
  const concepts = await Promise.all(tags.map(async ([tag, turns]) => {
    const avgComp = Math.round(turns.reduce((a, t) => a + t.competence, 0) / turns.length);
    const summary = `${tag} (avg competence ${avgComp}%, ${turns.length} question(s))`;
    const embedding = await embedText(`${tag}: ${turns.map(t => t.evidenceFeedback).join('. ')}`);
    return { tag, summary, avgCompetence: avgComp, embedding };
  }));

  store[candidateKey].sessions.push({
    sessionId: session.id,
    topic: session.topic,
    date: new Date(session.startedAt).toISOString(),
    scores: report ? report.scores : null,
    concepts
  });
  if (store[candidateKey].sessions.length > 60) store[candidateKey].sessions.shift();
  saveStore(store);
}

const sessions = new Map();

const MAX_QUESTIONS = parseInt(process.env.MAX_QUESTIONS || '20', 10);
const TARGET_MINUTES = parseFloat(process.env.INTERVIEW_MINUTES || '10');
const MIN_QUESTIONS_BEFORE_TIME_END = 5;

const PERSONAS = {
  friendly: 'Friendly and encouraging. Use warm phrasing, gentle nudges, light positive reinforcement ("Nice, can you go a little deeper?").',
  strict: 'Strict and terse. No pleasantries. Point out gaps directly ("That is incomplete. Give an example.").',
  faang: 'FAANG-bar-raiser style. Frame questions as system design / real-world scenarios ("Imagine you are designing X. Continue."). High expectations, probing follow-ups.'
};

function buildSystemPrompt(persona, topic, memoryContext, targetMinutes, interviewType, experience, resumeContext) {
  targetMinutes = targetMinutes || TARGET_MINUTES;
  return `You are an adaptive AI technical interviewer conducting a live spoken interview on the topic: "${topic}".

Persona / tone: ${PERSONAS[persona] || PERSONAS.friendly}
Interview type: ${interviewType || 'Technical'} — ${
    interviewType === 'Behavioral' ? 'favor behavioral/situational questions (STAR-style) over pure technical trivia.'
    : interviewType === 'Mixed' ? 'blend technical questions with behavioral/situational ones.'
    : interviewType === 'System Design' ? 'favor open-ended system design / architecture scenarios over narrow trivia.'
    : 'favor technical depth questions.'
  }
Candidate experience level: ${experience || 'Intermediate'} — calibrate question difficulty and expected depth accordingly.
${resumeContext ? `\nCandidate's resume / target job description (use this to tailor questions to their actual claimed experience — probe specific technologies, projects, or responsibilities they mention, and don't be afraid to test whether they can really back up a claim on their resume):\n"""\n${resumeContext}\n"""\n` : ''}
This is a TIME-BOXED interview targeting roughly ${targetMinutes} minutes total, not a fixed question count. Pace yourself against the elapsed time you're given each turn — don't stop after a small fixed number of questions, and don't run drastically over the target either.
${memoryContext ? `\n${memoryContext}\n` : ''}
Your job every turn — keep this LIGHTWEIGHT, this is a live low-latency loop, not the final report:
1. Judge the candidate's latest answer for CORRECTNESS (correct / partial / incorrect) and COMPETENCE (0-100).
2. Separately judge CONFIDENCE (0-100) from the way they spoke — hedging words like "umm", "I think", "maybe", "not sure" lower confidence even if the content is correct; assertive phrasing raises it even if the content is wrong. Confidence and competence are independent signals — call out mismatches (e.g. low confidence + correct answer, or high confidence + wrong answer).
3. Adapt difficulty: if the answer is strong, ask a harder follow-up on the SAME thread (dig deeper, don't just jump randomly). If the answer is weak, step back to a simpler foundational question on the same concept before moving on. Track difficulty on a 1-5 scale.
4. BREADTH RULE: don't dig into one single concept/thread indefinitely. After at most 2-3 consecutive questions deepening one specific thread (e.g. stacks), deliberately move to a different but related core concept within the same subject (e.g. queues, trees, hashing, recursion — whatever fits "${topic}") so the interview covers a spread of the subject, not just one corner of it.
5. Give ONE short evidence-based feedback sentence: paraphrase a short specific piece of what the candidate said and what it reveals. Keep it to one sentence — this is a live-pace check, not written feedback. Never give generic feedback like "weak in X" without pointing to the specific statement that shows it.
6. Write a short INTERNAL interviewer's note (5-12 words, like a scratchpad jotting a human interviewer would make, e.g. "Couldn't justify embedding choice, needed a hint") — never shown to the candidate live, only in the final report.
7. If relevant past concepts were provided above, occasionally (not every turn) weave in a genuine cross-day linking question that asks the candidate to connect the current topic to something they covered before — this is the single most valuable kind of question, use it when it fits naturally.
8. Decide the next question, building a natural thread within whichever subtopic you're currently on.
9. End the interview (set "done": true, "nextQuestion": "") once you judge the subject has been reasonably covered for a ${targetMinutes}-minute session, or when told elapsed time is at/past target — whichever comes first. Do not end before at least ${MIN_QUESTIONS_BEFORE_TIME_END} questions have been asked.

Do NOT attempt full scoring, communication analysis, or a hiring recommendation here — that is a separate, one-time step run only after the interview ends. Keep this response minimal and fast.

Always reply with STRICT JSON only, no markdown, matching this schema exactly:
{
  "verdict": "correct" | "partial" | "incorrect",
  "competence": <0-100 integer>,
  "confidence": <0-100 integer>,
  "evidenceFeedback": "<ONE short sentence, quoting/paraphrasing the candidate's own words>",
  "interviewerNote": "<short internal scratchpad note, 5-12 words>",
  "nextDifficulty": <1-5 integer>,
  "nextQuestion": "<the next interview question, in the persona's tone, or empty string if done>",
  "topicTag": "<short tag for the concept just tested, e.g. 'stacks', 'queues', 'recursion'>",
  "crossDayLink": <true|false, whether this question deliberately links to a past-day concept>,
  "done": <true|false>
}`;
}

function buildFirstQuestionPrompt(persona, topic, memoryContext, interviewType, experience, resumeContext) {
  return `Generate the FIRST interview question for a live spoken ${interviewType || 'Technical'}-style interview on "${topic}", for a candidate at ${experience || 'Intermediate'} experience level.${memoryContext ? ` ${memoryContext}` : ''}${resumeContext ? `\nCandidate's resume / target job description:\n"""\n${resumeContext}\n"""\nIf it clearly connects to "${topic}", ground the opening question in something specific from it (a real project, technology, or responsibility they listed) instead of a generic textbook question.` : ''} Start at a medium-easy difficulty (level 2 of 5) to establish a baseline. If relevant past-day concepts were given above and one connects naturally to "${topic}", you may open with a light callback ("Last time you covered X — today let's build on that with ${topic}.") but keep the actual question itself foundational. Reply with STRICT JSON only:
{
  "nextQuestion": "<the opening question, in tone: ${PERSONAS[persona] || PERSONAS.friendly}>",
  "nextDifficulty": 2,
  "topicTag": "<short tag>",
  "crossDayLink": <true|false>
}`;
}

function formatMemoryContext(memory) {
  if (!memory || !memory.items || !memory.items.length) return '';
  const lines = memory.items.map(c => `- Day linked to "${c.tag}": ${c.summary}`).join('\n');
  return `Candidate history (Day ${memory.dayCount + 1} for this candidate — this is a RETURNING candidate). Relevant concepts they covered on previous days, retrieved for cross-day linking:\n${lines}`;
}

// ---------- Routes ----------

app.get('/api/health', (req, res) => {
  res.json({ ok: true, models: MODELS, keysConfigured: API_KEYS.length, groqFallback: !!GROQ_API_KEY });
});

app.post('/api/interview/start', async (req, res) => {
  const t0 = Date.now();
  try {
    const {
      topic = 'Machine Learning & AI Systems', persona = 'friendly', candidateName = '',
      experience = 'Intermediate', interviewType = 'Technical', durationMinutes, resumeContext = ''
    } = req.body || {};
    const sessionId = uuidv4();
    const candidateKey = slugKey(candidateName);
    const targetMinutes = [15, 30, 45].includes(Number(durationMinutes)) ? Number(durationMinutes) : TARGET_MINUTES;
    const cleanResumeContext = String(resumeContext || '').trim().slice(0, 4000);

    const tMemStart = Date.now();
    const memory = await retrieveCrossDayMemory(candidateKey, topic, 3);
    const memoryContext = formatMemoryContext(memory);
    const memMs = Date.now() - tMemStart;

    const tLlmStart = Date.now();
    const raw = await callLLM(
      'You are an interview question generator. Reply with strict JSON only.',
      buildFirstQuestionPrompt(persona, topic, memoryContext, interviewType, experience, cleanResumeContext),
      { maxOutputTokens: 250, attemptTimeoutMs: 5000, budgetMs: 4500 } // tiny schema — no need for a big budget/token cap
    );
    const llmMs = Date.now() - tLlmStart;
    const parsed = safeParseJSON(raw) || {
      nextQuestion: `Let's start with the basics — can you explain what ${topic} means in your own words?`,
      nextDifficulty: 2,
      topicTag: 'intro'
    };

    const session = {
      id: sessionId,
      candidateKey,
      candidateName,
      topic,
      persona,
      experience,
      interviewType,
      targetMinutes,
      resumeContext: cleanResumeContext,
      difficulty: parsed.nextDifficulty || 2,
      turns: [],
      createdAt: Date.now(),
      startedAt: Date.now(),
      lastActive: Date.now(),
      done: false,
      memory,
      _pendingQuestion: parsed.nextQuestion
    };
    sessions.set(sessionId, session);

    const totalMs = Date.now() - t0;
    console.log(`[latency] /start memMs=${memMs} llmMs=${llmMs} totalMs=${totalMs}`);

    res.json({
      sessionId,
      question: parsed.nextQuestion,
      difficulty: session.difficulty,
      questionNumber: 1,
      targetMinutes,
      crossDayLink: !!parsed.crossDayLink,
      dayNumber: memory.dayCount + 1,
      linkedConcepts: memory.items.map(c => c.tag),
      done: false,
      timingMs: { memMs, llmMs, totalMs }
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to start interview', detail: err.message });
  }
});

// Keeps only the last N turns verbatim, and folds anything older into a
// compact one-line aggregate (tags + average competence) instead of full
// Q&A text. Without this, transcript size — and therefore prompt tokens and
// per-turn LLM latency — grows without bound as the interview goes on; by
// question 15 the model was re-reading 14 full Q&A exchanges every single
// turn just to decide question 15. Recent turns still get full detail since
// that's what actually matters for "ask a harder follow-up on the same
// thread" (rule 3) and "don't dig into one thread too long" (rule 4).
const RECENT_TURNS_FULL_DETAIL = 4;
function buildCompactTranscript(turns) {
  if (turns.length <= RECENT_TURNS_FULL_DETAIL) {
    return turns.map((t, i) =>
      `Q${i + 1} (difficulty ${t.difficulty}): ${t.question}\nCandidate: ${t.answer}\nVerdict: ${t.verdict}, competence ${t.competence}, confidence ${t.confidence}`
    ).join('\n\n');
  }
  const older = turns.slice(0, turns.length - RECENT_TURNS_FULL_DETAIL);
  const recent = turns.slice(turns.length - RECENT_TURNS_FULL_DETAIL);

  const tagStats = new Map();
  older.forEach(t => {
    if (!tagStats.has(t.topicTag)) tagStats.set(t.topicTag, []);
    tagStats.get(t.topicTag).push(t.competence);
  });
  const summaryLine = 'Earlier in this interview (Q1-Q' + older.length + '), covered: ' +
    [...tagStats.entries()].map(([tag, arr]) => `${tag} (avg ${Math.round(arr.reduce((a,b)=>a+b,0)/arr.length)}%)`).join(', ') + '.';

  const recentText = recent.map((t, i) => {
    const qNum = older.length + i + 1;
    return `Q${qNum} (difficulty ${t.difficulty}): ${t.question}\nCandidate: ${t.answer}\nVerdict: ${t.verdict}, competence ${t.competence}, confidence ${t.confidence}`;
  }).join('\n\n');

  return summaryLine + '\n\n' + recentText;
}

app.post('/api/interview/answer', async (req, res) => {
  const t0 = Date.now();
  try {
    const { sessionId, answer, visualConfidence } = req.body || {};
    const session = sessions.get(sessionId);
    if (!session) return res.status(404).json({ error: 'Session not found' });
    if (session.done) return res.status(400).json({ error: 'Interview already complete', done: true });
    if (!answer || !answer.trim()) return res.status(400).json({ error: 'Empty answer' });

    const targetMinutes = session.targetMinutes || TARGET_MINUTES;
    const questionNumber = session.turns.length + 1;
    const elapsedMin = (Date.now() - session.startedAt) / 60000;
    const overTime = elapsedMin >= targetMinutes && questionNumber > MIN_QUESTIONS_BEFORE_TIME_END;
    const hitAbsoluteCap = questionNumber >= MAX_QUESTIONS;

    // Lightweight, bounded-size transcript context — see buildCompactTranscript above.
    // This keeps per-turn latency roughly flat instead of growing with interview length.
    const transcript = buildCompactTranscript(session.turns);

    const memoryContext = formatMemoryContext(session.memory);

    const hasVisualForPrompt = typeof visualConfidence === 'number' && !Number.isNaN(visualConfidence);

    const userPrompt = `Topic: ${session.topic}
Question number: ${questionNumber}
Elapsed time: ${elapsedMin.toFixed(1)} minutes of a ~${targetMinutes}-minute target
Current difficulty: ${session.difficulty}

Transcript so far:
${transcript || '(none yet)'}

Current question asked: ${session._pendingQuestion}
Candidate's spoken answer (transcribed): "${answer}"
${hasVisualForPrompt ? `On-device webcam posture/eye-contact score for this answer: ${Math.round(visualConfidence)}/100 (derived from shoulder level, head position, and gaze steadiness — a rough physical-presence signal, separate from vocal tone). Weigh this alongside vocal cues for your CONFIDENCE judgment, and if it clearly diverges from what the words/tone suggest (e.g. strong words but poor posture score, or shaky words but steady posture), you may briefly note that mismatch in evidenceFeedback.` : ''}

Evaluate this answer and produce the next step, per the schema. Keep evidenceFeedback to ONE short sentence — this is a live low-latency turn, not the final report.${overTime || hitAbsoluteCap ? ' Time is up (or the safety question cap was hit) — this is the FINAL question, set done: true and nextQuestion to "".' : ''}`;

    const tLlmStart = Date.now();
    const raw = await callLLM(
      buildSystemPrompt(session.persona, session.topic, memoryContext, targetMinutes, session.interviewType, session.experience, session.resumeContext),
      userPrompt,
      // This is the hottest path in the whole app — the candidate is staring
      // at "Processing…" waiting on it. Small token budget (the schema is
      // short), a short per-attempt timeout, and a tight overall cascade
      // budget (4.5s) so a bad key/model never turns into multi-second dead
      // air; if Gemini can't answer inside that budget we fail over to Groq
      // (which is itself fast) rather than exhausting every combination.
      { maxOutputTokens: 350, attemptTimeoutMs: 4500, budgetMs: 4500 }
    );
    const llmMs = Date.now() - tLlmStart;
    const parsed = safeParseJSON(raw);

    if (!parsed) {
      return res.status(502).json({ error: 'Model returned unparsable response' });
    }

    const vocalConfidence = clamp(parsed.confidence, 0, 100, 50);
    const hasVisual = typeof visualConfidence === 'number' && !Number.isNaN(visualConfidence);
    const blendedConfidence = hasVisual
      ? clamp(Math.round(vocalConfidence * 0.65 + visualConfidence * 0.35), 0, 100, vocalConfidence)
      : vocalConfidence;

    const turn = {
      qNum: questionNumber,
      question: session._pendingQuestion,
      answer,
      verdict: parsed.verdict || 'partial',
      competence: clamp(parsed.competence, 0, 100, 50),
      confidence: blendedConfidence,
      vocalConfidence,
      visualConfidence: hasVisual ? clamp(visualConfidence, 0, 100, null) : null,
      evidenceFeedback: parsed.evidenceFeedback || '',
      interviewerNote: parsed.interviewerNote || '',
      difficulty: session.difficulty,
      topicTag: parsed.topicTag || 'general',
      crossDayLink: !!parsed.crossDayLink,
      timestamp: Date.now()
    };
    session.turns.push(turn);
    session.lastActive = Date.now();

    const isDone = !!parsed.done || overTime || hitAbsoluteCap;
    session.done = isDone;
    session.difficulty = clamp(parsed.nextDifficulty, 1, 5, session.difficulty);
    session._pendingQuestion = isDone ? null : (parsed.nextQuestion || '');

    const totalMs = Date.now() - t0;
    console.log(`[latency] /answer Q${questionNumber} llmMs=${llmMs} totalMs=${totalMs} transcriptChars=${transcript.length}`);

    res.json({
      sessionId,
      verdict: turn.verdict,
      competence: turn.competence,
      confidence: turn.confidence,
      evidenceFeedback: turn.evidenceFeedback,
      difficulty: session.difficulty,
      nextQuestion: session._pendingQuestion,
      questionNumber: questionNumber + (isDone ? 0 : 1),
      elapsedMinutes: Math.round(elapsedMin * 10) / 10,
      targetMinutes,
      crossDayLink: turn.crossDayLink,
      done: isDone,
      timingMs: { llmMs, totalMs }
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to process answer', detail: err.message });
  }
});

// ---- Heavy analysis lives ONLY here, run once at the end, never per-turn ----
app.get('/api/interview/:sessionId/report', async (req, res) => {
  const t0 = Date.now();
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
      answer: t.answer,
      verdict: t.verdict,
      topicTag: t.topicTag,
      difficulty: t.difficulty,
      evidenceFeedback: t.evidenceFeedback,
      crossDayLink: t.crossDayLink
    }));

    const byTag = new Map();
    turns.forEach(t => {
      if (!byTag.has(t.topicTag)) byTag.set(t.topicTag, []);
      byTag.get(t.topicTag).push(t.competence);
    });
    const skillRadar = [...byTag.entries()].map(([tag, arr]) => ({
      tag,
      score: Math.round(arr.reduce((a, b) => a + b, 0) / arr.length),
      stars: Math.round((arr.reduce((a, b) => a + b, 0) / arr.length) / 20 * 2) / 2
    }));

    const interviewerNotes = turns.filter(t => t.interviewerNote).map(t => ({
      qNum: t.qNum, note: t.interviewerNote, topicTag: t.topicTag
    }));

    const crossDayLinksUsed = turns.filter(t => t.crossDayLink).length;

    const report = {
      sessionId: session.id,
      topic: session.topic,
      persona: session.persona,
      scores: { hiringProbability, confidence, communication, problemSolving, competence },
      recommendation,
      timeline,
      skillRadar,
      interviewerNotes,
      crossDayLinksUsed,
      dayNumber: (session.memory ? session.memory.dayCount : 0) + 1,
      revisionPlan: uniqueWeak.map(tag => ({ topic: tag, estimatedMinutes: 15 })),
      totalEstimatedMinutes: uniqueWeak.length * 15
    };

    persistSession(session.candidateKey, session, report).catch(e => console.error('[persist]', e.message));

    console.log(`[latency] /report totalMs=${Date.now() - t0}`);
    res.json(report);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to build report', detail: err.message });
  }
});

app.get('/api/candidate/:name/summary', (req, res) => {
  const key = slugKey(req.params.name);
  const store = loadStore();
  const candidate = store[key];
  if (!candidate || !candidate.sessions.length) return res.json({ isReturning: false });
  const last = candidate.sessions[candidate.sessions.length - 1];
  res.json({
    isReturning: true,
    dayNumber: candidate.sessions.length + 1,
    lastTopic: last.topic,
    lastDate: last.date,
    recentTags: [...new Set(candidate.sessions.slice(-3).flatMap(s => (s.concepts || []).map(c => c.tag)))]
  });
});

app.get('/api/candidate/:name/history', (req, res) => {
  const key = slugKey(req.params.name);
  const store = loadStore();
  const candidate = store[key];
  if (!candidate || !candidate.sessions.length) return res.json({ hasHistory: false });

  const scored = candidate.sessions.filter(s => s.scores && typeof s.scores.hiringProbability === 'number');
  if (!scored.length) return res.json({ hasHistory: false });

  const recent = scored.slice(-8).reverse();
  const trend = scored.slice(-5).map(s => s.scores.hiringProbability);
  const readiness = trend[trend.length - 1];

  const latestScores = recent[0].scores;
  const skillBars = [
    { label: 'Technical', score: latestScores.competence },
    { label: 'Communication', score: latestScores.communication },
    { label: 'Problem Solving', score: latestScores.problemSolving }
  ];

  const tagAgg = new Map();
  candidate.sessions.forEach(s => {
    (s.concepts || []).forEach(c => {
      if (!tagAgg.has(c.tag)) tagAgg.set(c.tag, []);
      tagAgg.get(c.tag).push(c.avgCompetence);
    });
  });
  let weakest = null;
  for (const [tag, arr] of tagAgg.entries()) {
    const avg = Math.round(arr.reduce((a, b) => a + b, 0) / arr.length);
    if (!weakest || avg < weakest.score) weakest = { tag, score: avg };
  }

  res.json({
    hasHistory: true,
    readiness,
    trend,
    skillBars,
    weakest,
    sessions: recent.map(s => ({ topic: s.topic, date: s.date, score: s.scores.hiringProbability }))
  });
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
