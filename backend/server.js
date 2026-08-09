require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fetch = require('node-fetch');
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 5 * 1024 * 1024 } });

const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

app.use(express.static(path.join(__dirname, '../frontend')));

const PORT = process.env.PORT || 3000;

// ---------- Gemini multi-key fallback ----------
const RAW_KEYS = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
const API_KEYS = RAW_KEYS.split(',').map(k => k.trim()).filter(Boolean);
const RAW_MODELS = process.env.GEMINI_MODELS || process.env.GEMINI_MODEL || 'gemini-2.0-flash-lite,gemini-2.0-flash';
const MODELS = RAW_MODELS.split(',').map(m => m.trim()).filter(Boolean);
const EMBED_MODEL = process.env.GEMINI_EMBED_MODEL || 'text-embedding-004';

if (API_KEYS.length === 0) {
  console.warn('[WARN] No GEMINI_API_KEY / GEMINI_API_KEYS set. /api/interview calls will fail until you add one.');
}

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
        const remainingBudget = budgetMs - (Date.now() - cascadeStart);
        const thisAttemptTimeout = Math.max(1000, Math.min(attemptTimeoutMs, remainingBudget));
        const res = await fetchWithTimeout(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-goog-api-key': key },
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
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${GROQ_API_KEY}` },
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
  try { return JSON.parse(cleaned); }
  catch (e) { console.error('JSON parse failed, raw text:', cleaned.slice(0, 300)); return null; }
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

// ---------- Demo account seed data ----------
// Gives the "Aakashi Thakur" demo login a believable interview history right
// out of the box (Dashboard "Recent Interviews", readiness trend, and the
// History page all read from this same store) instead of showing only
// whatever the presenter has personally run through the live demo.
// Runs once at startup and is idempotent — it checks for a "demo-seed-"
// sessionId marker so it never duplicates entries on server restarts.
function buildDemoSession(sessionIdSuffix, topic, daysAgo, scores, questions) {
  const date = new Date(Date.now() - daysAgo * 24 * 60 * 60 * 1000).toISOString();
  const timeline = questions.map((q, i) => ({
    qNum: i + 1, question: q.q, answer: q.a, verdict: q.verdict,
    topicTag: q.tag, difficulty: q.difficulty, evidenceFeedback: q.feedback,
    crossDayLink: false, nextAction: q.next || null
  }));
  const byTag = new Map();
  timeline.forEach(t => {
    if (!byTag.has(t.topicTag)) byTag.set(t.topicTag, []);
    byTag.get(t.topicTag).push(t);
  });
  const concepts = [...byTag.entries()].map(([tag, turns]) => {
    const avgComp = Math.round(turns.reduce((a, t) => a + (t.verdict === 'correct' ? 85 : t.verdict === 'partial' ? 60 : 35), 0) / turns.length);
    return { tag, summary: `${tag} (avg competence ${avgComp}%, ${turns.length} question(s))`, avgCompetence: avgComp, embedding: null };
  });
  const skillRadar = [...byTag.entries()].map(([tag, turns]) => {
    const s = Math.round(turns.reduce((a, t) => a + (t.verdict === 'correct' ? 85 : t.verdict === 'partial' ? 60 : 35), 0) / turns.length);
    return { tag, score: s, stars: Math.round(s / 20 * 2) / 2 };
  });
  const weakTopics = timeline.filter(t => t.verdict !== 'correct').map(t => t.topicTag);
  const uniqueWeak = [...new Set(weakTopics)];
  const report = {
    sessionId: 'demo-seed-' + sessionIdSuffix,
    topic, persona: 'friendly', scores,
    recommendation: scores.hiringProbability >= 80 ? 'Strong Hire' : scores.hiringProbability >= 65 ? 'Hire' : scores.hiringProbability >= 45 ? 'Leaning No Hire' : 'No Hire',
    timeline, skillRadar,
    interviewerNotes: questions.filter(q => q.note).map((q, i) => ({ qNum: i + 1, note: q.note, topicTag: q.tag })),
    crossDayLinksUsed: 0, dayNumber: 1,
    revisionPlan: uniqueWeak.map(tag => ({ topic: tag, estimatedMinutes: 15 })),
    totalEstimatedMinutes: uniqueWeak.length * 15,
    practiceMode: false, practiceResult: null
  };
  return {
    sessionId: 'demo-seed-' + sessionIdSuffix, topic, date, scores, concepts,
    practiceMode: false, practiceFocus: null, report
  };
}

function seedDemoHistoryFor(fullName, demoSessions) {
  const store = loadStore();
  const key = slugKey(fullName);
  if (!store[key]) store[key] = { name: fullName, sessions: [] };
  const already = store[key].sessions.some(s => (s.sessionId || '').startsWith('demo-seed-'));
  if (already) return;

  // Oldest first, so they land before any real sessions chronologically.
  store[key].sessions = [...demoSessions, ...store[key].sessions];
  saveStore(store);
  console.log('[seed] demo history added for ' + fullName);
}

function seedDemoHistory() {
  seedDemoHistoryFor('Aakashi Thakur', [
    buildDemoSession('aak-1', 'Web Development', 8,
      { hiringProbability: 78, confidence: 74, communication: 79, problemSolving: 76, competence: 80 },
      [
        { q: 'Explain the difference between let, const, and var in JavaScript.', a: 'Covered scoping and hoisting differences clearly.', verdict: 'correct', tag: 'JavaScript Fundamentals', difficulty: 2, feedback: 'Clear, accurate explanation with a good example.' },
        { q: 'How does the virtual DOM improve rendering performance in React?', a: 'Explained diffing and batched updates.', verdict: 'correct', tag: 'React', difficulty: 3, feedback: 'Solid understanding of reconciliation.' },
        { q: 'What is the CSS box model?', a: 'Described content, padding, border, margin.', verdict: 'correct', tag: 'CSS', difficulty: 1, feedback: 'Textbook-accurate answer.' },
        { q: 'How would you optimize a slow-loading web page?', a: 'Mentioned lazy loading and image compression, missed code splitting.', verdict: 'partial', tag: 'Performance', difficulty: 3, feedback: 'Good instincts but missed bundle-level optimizations.', next: 'Review code-splitting and lazy-loading routes.', note: 'Strong on assets, weaker on JS bundle strategy.' },
        { q: 'Explain how async/await works under the hood.', a: 'Connected it correctly to promises and the event loop.', verdict: 'correct', tag: 'JavaScript Fundamentals', difficulty: 3, feedback: 'Confident, technically sound answer.' },
        { q: 'What are React hooks and why were they introduced?', a: 'Explained useState/useEffect and functional component motivation.', verdict: 'correct', tag: 'React', difficulty: 2, feedback: 'Well-structured explanation.' }
      ]),
    buildDemoSession('aak-2', 'Data Structures & Algorithms', 5,
      { hiringProbability: 72, confidence: 68, communication: 70, problemSolving: 74, competence: 71 },
      [
        { q: 'Explain how a hash table resolves collisions.', a: 'Mentioned chaining but was fuzzy on open addressing.', verdict: 'partial', tag: 'hashing', difficulty: 3, feedback: 'Partial understanding — open addressing wasn\u2019t clear.', next: 'Revisit open addressing and probing strategies.', note: 'Confused linear probing with chaining.' },
        { q: 'Compare arrays vs linked lists for insertion-heavy workloads.', a: 'Correctly reasoned about O(1) insertion for linked lists.', verdict: 'correct', tag: 'arrays vs linked lists', difficulty: 2, feedback: 'Good trade-off reasoning.' },
        { q: 'Reverse a doubly linked list — walk through your approach.', a: 'Described the pointer-swapping approach accurately.', verdict: 'correct', tag: 'doubly linked lists', difficulty: 3, feedback: 'Clean, correct approach.' },
        { q: 'What\u2019s the time complexity of your hashing-based solution?', a: 'Said O(n) but couldn\u2019t justify amortized analysis.', verdict: 'incorrect', tag: 'hashing', difficulty: 4, feedback: 'Answer was close but reasoning was incomplete.', next: 'Practice amortized time-complexity analysis.' },
        { q: 'When would you choose a linked list over an array?', a: 'Gave frequent-insertion/deletion scenario correctly.', verdict: 'correct', tag: 'data structures basics', difficulty: 1, feedback: 'Accurate, well-reasoned.' }
      ]),
    buildDemoSession('aak-3', 'Behavioral', 3,
      { hiringProbability: 81, confidence: 83, communication: 85, problemSolving: 74, competence: 78 },
      [
        { q: 'Tell me about a time you disagreed with a teammate.', a: 'Used a structured STAR-style answer with a clear resolution.', verdict: 'correct', tag: 'Communication', difficulty: 2, feedback: 'Well-structured, specific, and reflective.' },
        { q: 'Describe a project where you had to learn something new quickly.', a: 'Gave a concrete example with a clear learning process.', verdict: 'correct', tag: 'Adaptability', difficulty: 2, feedback: 'Confident delivery, good specificity.' },
        { q: 'How do you handle tight deadlines?', a: 'Explained prioritization but answer ran long and lost focus.', verdict: 'partial', tag: 'Communication', difficulty: 2, feedback: 'Good content, could be more concise.', next: 'Practice trimming answers to the core STAR points.', note: 'Answer was thorough but over-long — tighten delivery.' },
        { q: 'Tell me about a time you received difficult feedback.', a: 'Reflected maturely and described concrete behavior change.', verdict: 'correct', tag: 'Growth Mindset', difficulty: 2, feedback: 'Genuine, self-aware answer.' }
      ])
  ]);

  // Second demo persona — different subject mix and a lower/rising score
  // trend so the two demo accounts don't look identical during a live demo.
  seedDemoHistoryFor('Rahul Verma', [
    buildDemoSession('rah-1', 'Machine Learning Fundamentals', 9,
      { hiringProbability: 58, confidence: 55, communication: 61, problemSolving: 57, competence: 60 },
      [
        { q: 'Explain the bias-variance tradeoff.', a: 'Got the general idea but mixed up which one overfitting relates to.', verdict: 'partial', tag: 'ML Theory', difficulty: 2, feedback: 'Right direction, but the overfitting/underfitting mapping was reversed.', next: 'Re-derive the bias-variance decomposition with a concrete example.', note: 'Mixed up overfitting vs underfitting — worth a focused re-test.' },
        { q: 'What is the difference between supervised and unsupervised learning?', a: 'Correct, with good examples of each.', verdict: 'correct', tag: 'ML Theory', difficulty: 1, feedback: 'Clear and accurate.' },
        { q: 'How does gradient descent find a minimum?', a: 'Described the update rule but was vague on learning rate effects.', verdict: 'partial', tag: 'Optimization', difficulty: 3, feedback: 'Core idea was right, but learning-rate intuition was thin.', next: 'Practice explaining learning-rate too-high/too-low failure modes.' },
        { q: 'What is regularization and why do we use it?', a: 'Explained L2 regularization correctly.', verdict: 'correct', tag: 'ML Theory', difficulty: 2, feedback: 'Solid, precise answer.' }
      ]),
    buildDemoSession('rah-2', 'System Design', 6,
      { hiringProbability: 65, confidence: 62, communication: 66, problemSolving: 68, competence: 64 },
      [
        { q: 'How would you design a URL shortener?', a: 'Covered hashing and a basic DB schema, missed caching layer.', verdict: 'partial', tag: 'System Design Basics', difficulty: 3, feedback: 'Good foundation, but no caching or read-scaling discussion.', next: 'Practice adding a caching layer (Redis) to reduce DB load.', note: 'Jumped to implementation before discussing scale requirements.' },
        { q: 'What is the difference between horizontal and vertical scaling?', a: 'Explained both correctly with trade-offs.', verdict: 'correct', tag: 'Scalability', difficulty: 2, feedback: 'Clear, well-reasoned comparison.' },
        { q: 'How would you handle a sudden traffic spike?', a: 'Mentioned load balancers and auto-scaling correctly.', verdict: 'correct', tag: 'Scalability', difficulty: 3, feedback: 'Confident, accurate answer.' }
      ]),
    buildDemoSession('rah-3', 'RAG & AI Agents', 2,
      { hiringProbability: 74, confidence: 71, communication: 76, problemSolving: 73, competence: 75 },
      [
        { q: 'What problem does Retrieval-Augmented Generation solve?', a: 'Correctly tied it to reducing hallucination with grounded context.', verdict: 'correct', tag: 'RAG Concepts', difficulty: 2, feedback: 'Precise, well-articulated answer.' },
        { q: 'How do vector embeddings enable semantic search?', a: 'Explained embeddings and cosine similarity accurately.', verdict: 'correct', tag: 'RAG Concepts', difficulty: 3, feedback: 'Strong technical grasp.' },
        { q: 'What are the tradeoffs of chunk size in a RAG pipeline?', a: 'Partially covered it — mentioned context loss but not retrieval noise.', verdict: 'partial', tag: 'RAG Concepts', difficulty: 4, feedback: 'Good start, missed the retrieval-noise side of the tradeoff.', next: 'Compare small vs large chunk sizes on both recall and precision.' }
      ])
  ]);
}
seedDemoHistory();

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
  if (queryVec) scored = allConcepts.map(c => ({ c, score: cosineSim(queryVec, c.embedding) }));
  else scored = allConcepts.map(c => ({ c, score: keywordOverlapScore(topic, c.tag + ' ' + (c.summary || '')) }));
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
    concepts,
    practiceMode: !!session.practiceMode,
    practiceFocus: session.practiceFocus || null,
    report: report || null
  });
  if (store[candidateKey].sessions.length > 60) store[candidateKey].sessions.shift();
  saveStore(store);
}

// ---------- Concept-trend memory (feature: dashboard "AI Interview Memory") ----------
// Walks every past session in chronological order and, per concept tag,
// keeps the last two avgCompetence data points so we can classify each
// concept as improved / needs practice / re-test recommended. This is what
// powers the "🧠 Your AI Interview Memory" dashboard card.
function computeConceptTrends(candidateKey, limit = 6) {
  const store = loadStore();
  const candidate = store[candidateKey];
  if (!candidate || !candidate.sessions || !candidate.sessions.length) return [];

  const byTag = new Map(); // tag -> [{score, date}] chronological
  candidate.sessions.forEach(s => {
    (s.concepts || []).forEach(c => {
      if (!byTag.has(c.tag)) byTag.set(c.tag, []);
      byTag.get(c.tag).push({ score: c.avgCompetence, date: s.date });
    });
  });

  const results = [];
  for (const [tag, points] of byTag.entries()) {
    const latest = points[points.length - 1];
    const previous = points.length >= 2 ? points[points.length - 2] : null;
    let status;
    if (!previous) {
      status = 'retest_recommended'; // only ever tested once — worth confirming
    } else if (latest.score - previous.score >= 8) {
      status = 'improved';
    } else if (latest.score < 60) {
      status = 'needs_practice';
    } else if (previous.score - latest.score >= 8) {
      status = 'retest_recommended'; // regressed — flag for another look
    } else {
      status = 'stable';
    }
    results.push({
      tag,
      latestScore: latest.score,
      previousScore: previous ? previous.score : null,
      lastSeen: latest.date,
      status
    });
  }

  results.sort((a, b) => new Date(b.lastSeen) - new Date(a.lastSeen));
  return results.slice(0, limit);
}

const sessions = new Map();

const MAX_QUESTIONS = parseInt(process.env.MAX_QUESTIONS || '20', 10);
const TARGET_MINUTES = parseFloat(process.env.INTERVIEW_MINUTES || '10');
const MIN_QUESTIONS_BEFORE_TIME_END = 5;

const PRACTICE_MAX_QUESTIONS = 5;
const PRACTICE_MIN_QUESTIONS = 3;
const PRACTICE_TARGET_MINUTES = 8;

const PERSONAS = {
  friendly: 'Friendly and encouraging. Use warm phrasing, gentle nudges, light positive reinforcement ("Nice, can you go a little deeper?").',
  strict: 'Strict and terse. No pleasantries. Point out gaps directly ("That is incomplete. Give an example.").',
  faang: 'FAANG-bar-raiser style. Frame questions as system design / real-world scenarios ("Imagine you are designing X. Continue."). High expectations, probing follow-ups.'
};

function buildSystemPrompt(persona, topic, memoryContext, targetMinutes, interviewType, experience, resumeContext, practiceMode, practiceFocus) {
  targetMinutes = targetMinutes || TARGET_MINUTES;

  if (practiceMode) {
    return `You are an adaptive AI technical interviewer running a FOCUSED PRACTICE DRILL on exactly one concept: "${practiceFocus}".

Persona / tone: ${PERSONAS[persona] || PERSONAS.friendly}
Candidate experience level: ${experience || 'Intermediate'}.
This is a SHORT, TARGETED re-test — exactly ${PRACTICE_MAX_QUESTIONS} questions total, ALL of them about "${practiceFocus}" specifically. Do NOT branch to other concepts (no breadth rule here — depth on this one topic is the entire point).
${memoryContext ? `\n${memoryContext}\n` : ''}
Your job every turn — keep this LIGHTWEIGHT, this is a live low-latency loop:
1. Judge the candidate's latest answer for CORRECTNESS (correct / partial / incorrect) and COMPETENCE (0-100).
2. Judge CONFIDENCE (0-100) from phrasing (hedging vs assertive).
3. Ramp difficulty across the 5 questions: start foundational (level 1-2), end at a harder applied/scenario question (level 4-5) on the SAME concept — this is a deliberate difficulty ramp to test mastery, not a random walk.
4. Give ONE short evidence-based feedback sentence.
5. Write a short INTERNAL interviewer's note (5-12 words).
6. Decide the next question — always still about "${practiceFocus}".
7. topicTag MUST be "${practiceFocus}" (or a very close variant) on every turn — this drill's whole point is a clean before/after score on this exact concept.
8. Set "done": true after exactly ${PRACTICE_MAX_QUESTIONS} questions have been asked (not before ${PRACTICE_MIN_QUESTIONS}).

Always reply with STRICT JSON only, no markdown, matching this schema exactly:
{
  "verdict": "correct" | "partial" | "incorrect",
  "competence": <0-100 integer>,
  "confidence": <0-100 integer>,
  "evidenceFeedback": "<ONE short sentence>",
  "interviewerNote": "<short internal note, 5-12 words>",
  "nextDifficulty": <1-5 integer>,
  "nextQuestion": "<next question, or empty string if done>",
 "topicTag": "${practiceFocus}",
  "crossDayLink": false,
  "nextAction": "<one short phrase, 5-10 words, on WHY you're asking the next question>",
  "done": <true|false>
}`;
  }

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
  "nextAction": "<one short phrase, 5-10 words, on WHY you're asking the next question / what it's testing — e.g. 'Testing if they can apply this under load'. Shown only in the final report, never live.>",
  "done": <true|false>
}`;
}

function buildFirstQuestionPrompt(persona, topic, memoryContext, interviewType, experience, resumeContext, practiceMode, practiceFocus, practiceBaseline) {
  if (practiceMode) {
    return `Generate the FIRST question of a FOCUSED PRACTICE DRILL re-testing the concept "${practiceFocus}", for a candidate at ${experience || 'Intermediate'} level.${typeof practiceBaseline === 'number' ? ` Their score on this concept last time was ${practiceBaseline}%.` : ''} Start at a foundational difficulty (level 1-2 of 5) — this is a 5-question ramp that should end harder. Reply with STRICT JSON only:
{
  "nextQuestion": "<the opening question, in tone: ${PERSONAS[persona] || PERSONAS.friendly}>",
  "nextDifficulty": 1,
  "topicTag": "${practiceFocus}",
  "crossDayLink": false
}`;
  }
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

// ---- Resume/JD file upload: extracts plain text so the frontend can drop it
// straight into the existing resumeContext textarea. PDF via pdf-parse;
// anything else (.txt/.md) is read as-is. Nothing is written to disk —
// parsed in memory and returned once. Optionally saved to the candidate's
// profile below if a name is provided, so it auto-fills next login.
app.post('/api/resume/parse', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    const name = (req.file.originalname || '').toLowerCase();
    let text = '';
    if (name.endsWith('.pdf') || req.file.mimetype === 'application/pdf') {
      const data = await pdfParse(req.file.buffer);
      text = data.text || '';
    } else {
      text = req.file.buffer.toString('utf8');
    }
    text = text.replace(/\r/g, '').trim().slice(0, 8000);
    if (!text) return res.status(422).json({ error: 'Could not extract any text from that file — try pasting it instead.' });

    const candidateName = (req.body && req.body.candidateName) || '';
    if (candidateName.trim()) {
      const key = slugKey(candidateName);
      const store = loadStore();
      if (!store[key]) store[key] = { name: candidateName, sessions: [] };
      store[key].lastResumeText = text;
      saveStore(store);
    }

    res.json({ text });
  } catch (err) {
    console.error('[resume-parse]', err.message);
    res.status(500).json({ error: 'Failed to parse file', detail: err.message });
  }
});

// Lets the welcome screen auto-fill a previously uploaded resume for a returning candidate.
app.get('/api/candidate/:name/resume', (req, res) => {
  const key = slugKey(req.params.name);
  const store = loadStore();
  const text = store[key] && store[key].lastResumeText;
  res.json({ hasResume: !!text, text: text || '' });
});
// ---- Skill Gap: compares resume vs a job description, returns a match table
// + top risk areas. One-shot LLM call, no session/state involved.

app.post('/api/resume/skill-gap', async (req, res) => {
  try {
    const { resumeContext = '', jdContext = '' } = req.body || {};
    const resume = String(resumeContext || '').trim().slice(0, 6000);
    const jd = String(jdContext || '').trim().slice(0, 4000);
    if (!resume || !jd) return res.status(400).json({ error: 'Both resume and job description text are required' });

    const prompt = `Compare this candidate's resume against the job description below and produce a skill gap analysis.

Resume:
"""
${resume}
"""

Job Description:
"""
${jd}
"""

List the key skills/technologies the JD requires. For each, mark whether the resume shows it: "✓" (clearly shown), "~" (partial/implied), or "—" (missing). Then name the top 2-3 biggest interview risk areas.

Reply with STRICT JSON only:
{
  "skills": [ { "skill": "<name>", "resume": "✓"|"~"|"—", "note": "<optional short note>" } ],
  "riskAreas": ["<skill>", "<skill>"],
  "summary": "<one sentence, e.g. 'Your biggest interview risk is AWS and System Design.'>"
}`;

    const raw = await callLLM(
      'You are a technical recruiter analyzing skill gaps. Reply with strict JSON only.',
      prompt,
      { maxOutputTokens: 600, attemptTimeoutMs: 6000, budgetMs: 5500 }
    );
    const parsed = safeParseJSON(raw);
    if (!parsed) return res.status(502).json({ error: 'Model returned unparsable response' });
    res.json(parsed);
  } catch (err) {
    console.error('[skill-gap]', err.message);
    res.status(500).json({ error: 'Failed to analyze skill gap', detail: err.message });
  }
});
app.post('/api/interview/start', async (req, res) => {
  const t0 = Date.now();
  try {
    const {
      topic = 'Machine Learning & AI Systems', persona = 'friendly', candidateName = '',
      experience = 'Intermediate', interviewType = 'Technical', durationMinutes, resumeContext = '',
      practiceFocus = '', practiceBaselineScore
    } = req.body || {};
    const sessionId = uuidv4();
    const candidateKey = slugKey(candidateName);
    const isPractice = !!(practiceFocus && String(practiceFocus).trim());
    const cleanPracticeFocus = String(practiceFocus || '').trim().slice(0, 80);
    const effectiveTopic = isPractice ? `Focused practice: ${cleanPracticeFocus}` : topic;
    const targetMinutes = isPractice
      ? PRACTICE_TARGET_MINUTES
      : ([10, 15, 30, 45].includes(Number(durationMinutes)) ? Number(durationMinutes) : TARGET_MINUTES);
    const cleanResumeContext = String(resumeContext || '').trim().slice(0, 4000);

    const tMemStart = Date.now();
    const memory = isPractice ? { items: [], dayCount: 0 } : await retrieveCrossDayMemory(candidateKey, topic, 3);
    const memoryContext = formatMemoryContext(memory);
    const memMs = Date.now() - tMemStart;

    const tLlmStart = Date.now();
    const raw = await callLLM(
      'You are an interview question generator. Reply with strict JSON only.',
      buildFirstQuestionPrompt(persona, effectiveTopic, memoryContext, interviewType, experience, cleanResumeContext, isPractice, cleanPracticeFocus, practiceBaselineScore),
      { maxOutputTokens: 250, attemptTimeoutMs: 5000, budgetMs: 4500 }
    );
    const llmMs = Date.now() - tLlmStart;
    const parsed = safeParseJSON(raw) || {
      nextQuestion: isPractice
        ? `Let's re-test ${cleanPracticeFocus} — can you explain the core idea in your own words?`
        : `Let's start with the basics — can you explain what ${topic} means in your own words?`,
      nextDifficulty: isPractice ? 1 : 2,
      topicTag: isPractice ? cleanPracticeFocus : 'intro'
    };

    const session = {
      id: sessionId,
      candidateKey,
      candidateName,
      topic: effectiveTopic,
      persona,
      experience,
      interviewType,
      targetMinutes,
      resumeContext: cleanResumeContext,
      difficulty: parsed.nextDifficulty || (isPractice ? 1 : 2),
      turns: [],
      createdAt: Date.now(),
      startedAt: Date.now(),
      lastActive: Date.now(),
      done: false,
      memory,
      practiceMode: isPractice,
      practiceFocus: isPractice ? cleanPracticeFocus : null,
      practiceBaseline: isPractice && typeof practiceBaselineScore === 'number' ? practiceBaselineScore : null,
      _pendingQuestion: parsed.nextQuestion
    };
    sessions.set(sessionId, session);

    const totalMs = Date.now() - t0;
    console.log(`[latency] /start memMs=${memMs} llmMs=${llmMs} totalMs=${totalMs} practiceMode=${isPractice}`);

    res.json({
      sessionId,
      question: parsed.nextQuestion,
      difficulty: session.difficulty,
      questionNumber: 1,
      targetMinutes,
      crossDayLink: !!parsed.crossDayLink,
      dayNumber: memory.dayCount + 1,
      linkedConcepts: memory.items.map(c => c.tag),
      practiceMode: isPractice,
      practiceFocus: session.practiceFocus,
      done: false,
      timingMs: { memMs, llmMs, totalMs }
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to start interview', detail: err.message });
  }
});

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

    const maxQ = session.practiceMode ? PRACTICE_MAX_QUESTIONS : MAX_QUESTIONS;
    const minQBeforeEnd = session.practiceMode ? PRACTICE_MIN_QUESTIONS : MIN_QUESTIONS_BEFORE_TIME_END;
    const overTime = elapsedMin >= targetMinutes && questionNumber > minQBeforeEnd;
    const hitAbsoluteCap = questionNumber >= maxQ;

    const transcript = buildCompactTranscript(session.turns);
    const memoryContext = formatMemoryContext(session.memory);
    const hasVisualForPrompt = typeof visualConfidence === 'number' && !Number.isNaN(visualConfidence);

    const userPrompt = `Topic: ${session.topic}
Question number: ${questionNumber}${session.practiceMode ? ` of ${PRACTICE_MAX_QUESTIONS} (focused practice drill)` : ''}
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
      buildSystemPrompt(session.persona, session.topic, memoryContext, targetMinutes, session.interviewType, session.experience, session.resumeContext, session.practiceMode, session.practiceFocus),
      userPrompt,
      { maxOutputTokens: 350, attemptTimeoutMs: 4500, budgetMs: 4500 }
    );
    const llmMs = Date.now() - tLlmStart;
    const parsed = safeParseJSON(raw);
    if (!parsed) return res.status(502).json({ error: 'Model returned unparsable response' });

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
      nextAction: parsed.nextAction || '',
      difficulty: session.difficulty,
      topicTag: session.practiceMode ? session.practiceFocus : (parsed.topicTag || 'general'),
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
      practiceMode: session.practiceMode,
      done: isDone,
      timingMs: { llmMs, totalMs }
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Failed to process answer', detail: err.message });
  }
});

app.get('/api/interview/:sessionId/report', async (req, res) => {
  const t0 = Date.now();
  try {
    const session = sessions.get(req.params.sessionId);
    if (!session) {
      // Live session isn't in memory anymore (server restarted, or this is an
      // older interview being reopened from History) — fall back to the
      // full report we persisted to disk right after the interview finished.
      const store = loadStore();
      for (const key in store) {
        const past = (store[key].sessions || []).find(s => s.sessionId === req.params.sessionId);
        if (past && past.report) return res.json(past.report);
      }
      return res.status(404).json({ error: 'Session not found' });
    }
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
      qNum: t.qNum, question: t.question, answer: t.answer, verdict: t.verdict,
      topicTag: t.topicTag, difficulty: t.difficulty, evidenceFeedback: t.evidenceFeedback, crossDayLink: t.crossDayLink,
      nextAction: t.nextAction
    }));

    const byTag = new Map();
    turns.forEach(t => {
      if (!byTag.has(t.topicTag)) byTag.set(t.topicTag, []);
      byTag.get(t.topicTag).push(t.competence);
    });
    const skillRadar = [...byTag.entries()].map(([tag, arr]) => ({
      tag, score: Math.round(arr.reduce((a, b) => a + b, 0) / arr.length),
      stars: Math.round((arr.reduce((a, b) => a + b, 0) / arr.length) / 20 * 2) / 2
    }));

    const interviewerNotes = turns.filter(t => t.interviewerNote).map(t => ({ qNum: t.qNum, note: t.interviewerNote, topicTag: t.topicTag }));
    const crossDayLinksUsed = turns.filter(t => t.crossDayLink).length;

    let practiceResult = null;
    if (session.practiceMode) {
      const afterScore = avg(turns.map(t => t.competence));
      practiceResult = {
        tag: session.practiceFocus,
        before: typeof session.practiceBaseline === 'number' ? session.practiceBaseline : null,
        after: afterScore,
        improvement: typeof session.practiceBaseline === 'number' ? afterScore - session.practiceBaseline : null
      };
    }

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
      totalEstimatedMinutes: uniqueWeak.length * 15,
      practiceMode: session.practiceMode,
      practiceResult
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
    hasHistory: true, readiness, trend, skillBars, weakest,
    sessions: recent.map(s => ({
      topic: s.topic, date: s.date, score: s.scores.hiringProbability,
      sessionId: s.sessionId,
      concepts: (s.concepts || []).map(c => ({ tag: c.tag, score: c.avgCompetence }))
    }))
  });
});

// ---- "🧠 Your AI Interview Memory" dashboard data ----
app.get('/api/candidate/:name/memory', (req, res) => {
  const key = slugKey(req.params.name);
  const concepts = computeConceptTrends(key, 6);
  res.json({ hasMemory: concepts.length > 0, concepts });
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