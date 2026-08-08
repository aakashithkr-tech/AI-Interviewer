require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { v4: uuidv4 } = require('uuid');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const multer = require('multer');

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
// free tier), fall back to the next. Override with GEMINI_MODELS in .env
// (comma-separated) if you want a different order.
const RAW_MODELS = process.env.GEMINI_MODELS || process.env.GEMINI_MODEL || 'gemini-2.0-flash,gemini-2.0-flash-lite';
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
        const res = await fetchWithTimeout(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-goog-api-key': key
          },
          body: JSON.stringify(body)
        }, 12000);
        if (!res.ok) {
          const text = await res.text();
          throw new Error(`[${model}] key #${i + 1} failed: ${res.status} ${text.slice(0, 200)}`);
        }
        const data = await res.json();
        const text = data?.candidates?.[0]?.content?.parts?.map(p => p.text).join('') || '';
        if (!text) throw new Error(`[${model}] key #${i + 1} returned empty content`);
        return text;
      } catch (err) {
        console.error(`[${model}] key #${i + 1}:`, err.message);
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
      max_tokens: 1024,
      ...(json ? { response_format: { type: 'json_object' } } : {})
    })
  }, 12000);
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

// =====================================================================
// RAG memory store — cross-day linking
// A lightweight, dependency-free JSON store (no native DB build step,
// so it deploys cleanly on Render's free tier). Each candidate's past
// sessions are kept with an embedding per covered concept; new sessions
// retrieve the most semantically-relevant past concepts via cosine
// similarity so the interviewer can weave in genuine "you covered X on
// Day 3, now let's connect it to Y" continuity — not just random recall.
// =====================================================================
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

async function embedText(text) {
  if (!API_KEYS.length) return null;
  for (const key of API_KEYS) {
    try {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:embedContent`;
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-goog-api-key': key },
        body: JSON.stringify({ content: { parts: [{ text }] } })
      });
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

// Retrieve the top-K most relevant past concepts for this candidate,
// across ALL previous sessions/days — this is the actual "RAG" step.
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

// Persist a finished session's concepts (with embeddings) for future retrieval,
// AND the full report object so History → "View Report" can re-render it exactly
// as it looked right after the interview, without re-running the LLM.
async function persistSession(candidateKey, session, report) {
  const store = loadStore();
  if (!store[candidateKey]) store[candidateKey] = { name: session.candidateName || candidateKey, sessions: [] };

  // Build one concept entry per distinct topicTag covered this session.
  const byTag = new Map();
  session.turns.forEach(t => {
    if (!byTag.has(t.topicTag)) byTag.set(t.topicTag, []);
    byTag.get(t.topicTag).push(t);
  });
  const concepts = [];
  for (const [tag, turns] of byTag.entries()) {
    const avgComp = Math.round(turns.reduce((a, t) => a + t.competence, 0) / turns.length);
    const summary = `${tag} (avg competence ${avgComp}%, ${turns.length} question(s))`;
    const embedding = await embedText(`${tag}: ${turns.map(t => t.evidenceFeedback).join('. ')}`);
    concepts.push({ tag, summary, avgCompetence: avgComp, embedding });
  }

  store[candidateKey].sessions.push({
    sessionId: session.id,
    topic: session.topic,
    interviewType: session.interviewType || 'Technical',
    durationMinutes: session.targetMinutes,
    questionCount: session.turns.length,
    date: new Date(session.startedAt).toISOString(),
    scores: report ? report.scores : null,
    concepts,
    report: report || null
  });
  // Keep it bounded so the file doesn't grow forever in a long demo.
  if (store[candidateKey].sessions.length > 60) store[candidateKey].sessions.shift();
  saveStore(store);
}

// =====================================================================

// =====================================================================
// Auth — simple JWT + bcrypt, backed by the same dependency-free JSON
// store pattern as the candidate RAG data (no DB build step needed).
// The signing secret is generated once and persisted to disk so tokens
// stay valid across server restarts even without a JWT_SECRET env var.
// =====================================================================
const USERS_FILE = path.join(DATA_DIR, 'users.json');
if (!fs.existsSync(USERS_FILE)) fs.writeFileSync(USERS_FILE, '{}');
const SECRET_FILE = path.join(DATA_DIR, 'jwt_secret.txt');
let JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET) {
  if (fs.existsSync(SECRET_FILE)) {
    JWT_SECRET = fs.readFileSync(SECRET_FILE, 'utf8').trim();
  } else {
    JWT_SECRET = crypto.randomBytes(48).toString('hex');
    fs.writeFileSync(SECRET_FILE, JWT_SECRET);
  }
}

function loadUsers() {
  try { return JSON.parse(fs.readFileSync(USERS_FILE, 'utf8')); }
  catch (e) { console.error('[users] read failed, starting fresh:', e.message); return {}; }
}
function saveUsers(users) {
  try { fs.writeFileSync(USERS_FILE, JSON.stringify(users, null, 2)); }
  catch (e) { console.error('[users] write failed:', e.message); }
}
function publicUser(u) { return { id: u.id, name: u.name, email: u.email }; }
function signToken(u) { return jwt.sign({ id: u.id, email: u.email }, JWT_SECRET, { expiresIn: '30d' }); }

// Attaches req.user from the Bearer token; 401s if missing/invalid.
function requireAuth(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Not signed in' });
  try {
    const payload = jwt.verify(token, JWT_SECRET);
    const users = loadUsers();
    const user = users[payload.email];
    if (!user || user.id !== payload.id) return res.status(401).json({ error: 'Session expired, please sign in again' });
    req.user = publicUser(user);
    next();
  } catch (e) {
    return res.status(401).json({ error: 'Session expired, please sign in again' });
  }
}

app.post('/api/auth/signup', async (req, res) => {
  try {
    const { name = '', email = '', password = '' } = req.body || {};
    const cleanEmail = String(email).trim().toLowerCase();
    if (!name.trim() || !cleanEmail || !password) return res.status(400).json({ error: 'Name, email, and password are all required' });
    if (password.length < 6) return res.status(400).json({ error: 'Password must be at least 6 characters' });
    const users = loadUsers();
    if (users[cleanEmail]) return res.status(409).json({ error: 'An account with that email already exists' });
    const passwordHash = await bcrypt.hash(password, 10);
    const user = { id: uuidv4(), name: name.trim(), email: cleanEmail, passwordHash, createdAt: Date.now() };
    users[cleanEmail] = user;
    saveUsers(users);
    res.json({ token: signToken(user), user: publicUser(user) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Signup failed', detail: err.message });
  }
});

app.post('/api/auth/login', async (req, res) => {
  try {
    const { email = '', password = '' } = req.body || {};
    const cleanEmail = String(email).trim().toLowerCase();
    const users = loadUsers();
    const user = users[cleanEmail];
    if (!user) return res.status(401).json({ error: 'No account found with that email' });
    const ok = await bcrypt.compare(password, user.passwordHash);
    if (!ok) return res.status(401).json({ error: 'Incorrect password' });
    res.json({ token: signToken(user), user: publicUser(user) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Login failed', detail: err.message });
  }
});

app.get('/api/auth/me', requireAuth, (req, res) => {
  res.json({ user: req.user });
});

const sessions = new Map();

const MAX_QUESTIONS = parseInt(process.env.MAX_QUESTIONS || '20', 10); // absolute safety cap, not the normal ending condition
const TARGET_MINUTES = parseFloat(process.env.INTERVIEW_MINUTES || '10'); // interview paces itself to roughly this long
const MIN_QUESTIONS_BEFORE_TIME_END = 8; // product requirement: every interview must reach at least 8 questions

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
Your job every turn:
1. Judge the candidate's latest answer for CORRECTNESS (correct / partial / incorrect) and COMPETENCE (0-100).
2. Separately judge CONFIDENCE (0-100) from the way they spoke — hedging words like "umm", "I think", "maybe", "not sure" lower confidence even if the content is correct; assertive phrasing raises it even if the content is wrong. Confidence and competence are independent signals — call out mismatches (e.g. low confidence + correct answer, or high confidence + wrong answer).
3. Adapt difficulty: if the answer is strong, ask a harder follow-up on the SAME thread (dig deeper, don't just jump randomly). If the answer is weak, step back to a simpler foundational question on the same concept before moving on. Track difficulty on a 1-5 scale.
4. BREADTH RULE: don't dig into one single concept/thread indefinitely. After at most 2-3 consecutive questions deepening one specific thread (e.g. stacks), deliberately move to a different but related core concept within the same subject (e.g. queues, trees, hashing, recursion — whatever fits "${topic}") so the interview covers a spread of the subject, not just one corner of it.
5. Give evidence-based feedback: quote or closely paraphrase a short specific piece of what the candidate said and explain what it reveals. Never give generic feedback like "weak in X" without pointing to the specific statement that shows it.
6. Write a short INTERNAL interviewer's note (5-12 words, like a scratchpad jotting a human interviewer would make, e.g. "Couldn't justify embedding choice, needed a hint") — never shown to the candidate live, only in the final report.
7. If relevant past concepts were provided above, occasionally (not every turn) weave in a genuine cross-day linking question that asks the candidate to connect the current topic to something they covered before — this is the single most valuable kind of question, use it when it fits naturally.
8. Decide the next question, building a natural thread within whichever subtopic you're currently on.
9. End the interview (set "done": true, "nextQuestion": "") once you judge the subject has been reasonably covered for a ${targetMinutes}-minute session, or when told elapsed time is at/past target — whichever comes first. Do not end before at least ${MIN_QUESTIONS_BEFORE_TIME_END} questions have been asked.

Always reply with STRICT JSON only, no markdown, matching this schema exactly:
{
  "verdict": "correct" | "partial" | "incorrect",
  "competence": <0-100 integer>,
  "confidence": <0-100 integer>,
  "evidenceFeedback": "<one or two sentences, quoting/paraphrasing the candidate's own words>",
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

app.post('/api/interview/start', requireAuth, async (req, res) => {
  try {
    const {
      topic = 'Machine Learning & AI Systems', persona = 'friendly',
      experience = 'Intermediate', interviewType = 'Technical', durationMinutes, resumeContext = ''
    } = req.body || {};
    const sessionId = uuidv4();
    const candidateKey = req.user.id;
    const candidateName = req.user.name;
    const targetMinutes = [10, 15, 30, 45].includes(Number(durationMinutes)) ? Number(durationMinutes) : TARGET_MINUTES;
    // Cap length so a giant pasted resume/JD can't blow out prompt size or cost.
    const cleanResumeContext = String(resumeContext || '').trim().slice(0, 4000);

    const memory = await retrieveCrossDayMemory(candidateKey, topic, 3);
    const memoryContext = formatMemoryContext(memory);

    const raw = await callLLM(
      'You are an interview question generator. Reply with strict JSON only.',
      buildFirstQuestionPrompt(persona, topic, memoryContext, interviewType, experience, cleanResumeContext)
    );
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

    res.json({
      sessionId,
      question: parsed.nextQuestion,
      difficulty: session.difficulty,
      questionNumber: 1,
      targetMinutes,
      crossDayLink: !!parsed.crossDayLink,
      dayNumber: memory.dayCount + 1,
      linkedConcepts: memory.items.map(c => c.tag),
      done: false
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to start interview', detail: err.message });
  }
});

app.post('/api/interview/answer', async (req, res) => {
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

    // Build conversation transcript for context
    const transcript = session.turns.map((t, i) =>
      `Q${i + 1} (difficulty ${t.difficulty}): ${t.question}\nCandidate: ${t.answer}\nVerdict: ${t.verdict}, competence ${t.competence}, confidence ${t.confidence}`
    ).join('\n\n');

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

Evaluate this answer and produce the next step, per the schema.${overTime || hitAbsoluteCap ? ' Time is up (or the safety question cap was hit) — this is the FINAL question, set done: true and nextQuestion to "".' : ''}`;

    const raw = await callLLM(
      buildSystemPrompt(session.persona, session.topic, memoryContext, targetMinutes, session.interviewType, session.experience, session.resumeContext),
      userPrompt
    );
    const parsed = safeParseJSON(raw);

    if (!parsed) {
      return res.status(502).json({ error: 'Model returned unparsable response' });
    }

    // Blend the model's vocal-cue confidence read with the on-device
    // posture/eye-contact signal from the webcam (if the candidate enabled
    // the camera). Vocal cues stay the majority signal; posture nudges it.
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
      answer: t.answer,
      verdict: t.verdict,
      topicTag: t.topicTag,
      difficulty: t.difficulty,
      evidenceFeedback: t.evidenceFeedback,
      crossDayLink: t.crossDayLink
    }));

    // ---- Skill radar: average competence per distinct concept tag ----
    const byTag = new Map();
    turns.forEach(t => {
      if (!byTag.has(t.topicTag)) byTag.set(t.topicTag, []);
      byTag.get(t.topicTag).push(t.competence);
    });
    const skillRadar = [...byTag.entries()].map(([tag, arr]) => ({
      tag,
      score: Math.round(arr.reduce((a, b) => a + b, 0) / arr.length),
      stars: Math.round((arr.reduce((a, b) => a + b, 0) / arr.length) / 20 * 2) / 2 // nearest 0.5, out of 5
    }));

    // ---- Interviewer's notes (hidden scratchpad, revealed only here) ----
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

    // Fire-and-forget: persist this session's concepts for future cross-day RAG retrieval.
    persistSession(session.candidateKey, session, report).catch(e => console.error('[persist]', e.message));

    res.json(report);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to build report', detail: err.message });
  }
});

// Small endpoint the frontend can use to greet a returning candidate on the
// welcome screen ("Welcome back — Day 4, last time: RAG, Vector DBs...").
app.get('/api/me/summary', requireAuth, (req, res) => {
  const store = loadStore();
  const candidate = store[req.user.id];
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

// Powers the Dashboard's "Interview Readiness" / "Recent Interviews" and the
// Report's "Progress" trend — derived from the same candidate store used for
// cross-day RAG linking, no extra persistence needed.
app.get('/api/dashboard', requireAuth, (req, res) => {
  const store = loadStore();
  const candidate = store[req.user.id];
  if (!candidate || !candidate.sessions.length) return res.json({ hasHistory: false });

  const scored = candidate.sessions.filter(s => s.scores && typeof s.scores.hiringProbability === 'number');
  if (!scored.length) return res.json({ hasHistory: false });

  const recent = scored.slice(-8).reverse(); // newest first, for the "Recent Interviews" list
  const trend = scored.slice(-5).map(s => s.scores.hiringProbability); // oldest -> newest, for trend chips
  const readiness = trend[trend.length - 1];

  const latestScores = recent[0].scores;
  const skillBars = [
    { label: 'Technical', score: latestScores.competence },
    { label: 'Communication', score: latestScores.communication },
    { label: 'Problem Solving', score: latestScores.problemSolving }
  ];

  // Weakest concept tag across all sessions, by average competence.
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
    recent: recent.map(s => ({ topic: s.topic, date: s.date, score: s.scores.hiringProbability }))
  });
});

// Full interview history, optionally filtered by interview type, plus the
// same progress trend shown on the dashboard/report (kept consistent across all three).
app.get('/api/history', requireAuth, (req, res) => {
  const store = loadStore();
  const candidate = store[req.user.id];
  if (!candidate || !candidate.sessions.length) return res.json({ interviews: [], trend: [] });

  const typeFilter = req.query.type;
  const scored = candidate.sessions.filter(s => s.scores && typeof s.scores.hiringProbability === 'number');
  const trend = scored.slice(-5).map(s => s.scores.hiringProbability);

  let list = [...scored].reverse(); // newest first
  if (typeFilter && typeFilter !== 'All') list = list.filter(s => (s.interviewType || 'Technical') === typeFilter);

  res.json({
    interviews: list.map(s => ({
      id: s.sessionId,
      topic: s.topic,
      interviewType: s.interviewType || 'Technical',
      durationMinutes: s.durationMinutes,
      questionCount: s.questionCount,
      date: s.date,
      score: s.scores.hiringProbability
    })),
    trend
  });
});

// Re-opens a past session's full report (built at the time the interview
// finished) so History → "View Report" / "Replay" work without re-scoring.
app.get('/api/history/:sessionId', requireAuth, (req, res) => {
  const store = loadStore();
  const candidate = store[req.user.id];
  const entry = candidate && candidate.sessions.find(s => s.sessionId === req.params.sessionId);
  if (!entry || !entry.report) return res.status(404).json({ error: 'Report not found' });
  res.json({ report: entry.report });
});

// ---------- resume / JD upload — extracts raw text + a quick skill scan ----------
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 5 * 1024 * 1024 } });
const KNOWN_SKILLS = [
  'Java', 'JavaScript', 'TypeScript', 'Python', 'C++', 'C#', 'Go', 'Rust',
  'React', 'Angular', 'Vue', 'Node.js', 'Express', 'Spring Boot', 'Django', 'Flask',
  'MongoDB', 'MySQL', 'PostgreSQL', 'Redis', 'Firebase', 'AWS', 'Azure', 'GCP', 'Docker', 'Kubernetes',
  'Flutter', 'React Native', 'Android', 'iOS', 'GraphQL', 'REST API', 'Machine Learning',
  'TensorFlow', 'PyTorch', 'Git', 'CI/CD', 'Kafka', 'RabbitMQ', 'Microservices', 'DSA'
];
function detectSkills(text) {
  const lower = text.toLowerCase();
  return KNOWN_SKILLS.filter(sk => lower.includes(sk.toLowerCase()));
}
app.post('/api/resume/parse', requireAuth, upload.single('resume'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    const { originalname, mimetype, size, buffer } = req.file;
    let text = '';
    if (mimetype === 'application/pdf' || originalname.toLowerCase().endsWith('.pdf')) {
      const pdfParse = require('pdf-parse');
      const parsed = await pdfParse(buffer);
      text = parsed.text || '';
    } else if (originalname.toLowerCase().endsWith('.docx') || mimetype.includes('wordprocessingml')) {
      const mammoth = require('mammoth');
      const parsed = await mammoth.extractRawText({ buffer });
      text = parsed.value || '';
    } else {
      return res.status(400).json({ error: 'Only PDF or DOCX files are supported' });
    }
    text = text.trim().slice(0, 6000); // keep it bounded for prompt-size sanity
    if (!text) return res.status(422).json({ error: 'Could not extract any text from that file — try pasting the text instead' });
    res.json({ text, filename: originalname, sizeBytes: size, skills: detectSkills(text) });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Could not parse that file', detail: err.message });
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