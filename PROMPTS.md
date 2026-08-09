# PROMPTS.md

Started keeping this properly once I realized I'd never remember the order
of everything otherwise. Rough chronological order, grouped loosely by
what stretch of the build I was in. Some of these took multiple back-and-
forth passes before I got what I wanted, I've tried to capture the actual
reasoning, not just the final clean version of the ask, because honestly
the messy iteration is most of where the real thinking happened.
Prompts below were largely drafted/refined with ChatGPT first — for architecture decisions, comparisons, and reasoning through tradeoffs — then handed to Claude for actual implementation, code generation, and iterative debugging inside the codebase. Where an entry below describes "asking for X," that reasoning pass often happened in ChatGPT before the resulting instruction was given to Claude to build. This log is a synthesized, chronological account of that combined process rather than a raw, single-tool transcript.
---

## Picking the direction

### 1. Which problem statement, and why it actually matters for judging
Was stuck between PS2 and PS3 for a good chunk of the first evening. My
first instinct was just to pick whichever sounded more impressive to say
out loud, but that felt like a trap — a hackathon demo lives or dies on
whether it actually works live, not on how it sounds on paper. So instead
I asked for a straight comparison weighted specifically against typical
judging criteria: technical depth, demo reliability, originality,
completeness in the time available. I pushed further and asked it to
specifically reason about failure-mode surface area — not "which is
cooler" but which one has fewer distinct ways to visibly break on stage in
front of judges, since a broken demo tanks a score way harder than a
slightly-less-ambitious-but-solid one. PS2 (the adaptive interview agent)
won on that basis — it degrades gracefully (worst case, a slightly dumber
interviewer) rather than catastrophically (a totally broken flow), and
that mattered more to me than raw ambition.

### 2. Voice — figuring out what's actually reliable in a 48h window
Didn't want to depend on a paid voice API and then have it rate-limit or
time out during the actual live demo — that's the kind of failure that's
completely out of my control once it's happening on stage, which is
exactly the risk category I was trying to avoid from prompt 1. Asked for a
breakdown of the realistic options: paid STT/TTS APIs (ElevenLabs,
Deepgram, Whisper-based services) versus the browser's own native Web
Speech API. Specifically asked it to weigh added latency hops, number of
extra API keys/vendors I'd need to manage under time pressure, risk of
hitting a rate limit specifically during the live judged demo, and total
added cost for a project with a zero budget. Ended up going with the
browser-native option — `SpeechRecognition` for the mic, `speechSynthesis`
for the voice out — because it has genuinely zero external dependency
surface for voice specifically. Gemini only ever touches the actual
interview reasoning, text in and text out; nothing voice-related routes
through any paid service at all. That decision alone removed a whole
category of "what if this API is down during my 3-minute demo slot"
anxiety.

### 3. Which LLM to build around, and whether I need a backup provider
Compared Gemini's flash-lite/flash models against a couple of other
realistic options purely on speed-per-turn and how generous the free tier
was, since this isn't a single one-shot generation — it's a live,
repeated back-and-forth loop where every extra second of latency is felt
directly by whoever's mid-interview. Landed on Gemini as primary. But I'd
been burned before by a single API key hitting quota mid-demo at a past
hackathon, so I explicitly asked what a sane fallback strategy would look
like — ended up deciding to keep Groq in reserve as a secondary provider,
not because it's necessarily as good, but because "something answered" is
strictly better than "nothing answered" when you're live in front of
judges.

### 4. Basic repo structure and why I avoided any build tooling
Told it to set up a plain `backend/` + `frontend/` split — Express on the
backend, a single static HTML file on the frontend with absolutely no
build step, no framework, no bundler. I was explicit about the reasoning:
I did not want to be debugging a webpack or vite config at 3am twelve
hours before submission when that time would be far better spent on the
actual interview logic. The frontend just needs to be openable as a raw
file or hosted anywhere that serves static files, full stop.

### 5. Getting a dummy endpoint deployed before writing any real logic
Before touching any real interview logic, I got the most barebones
possible Express server running — one fake `/api/interview` route that
just returns hardcoded JSON — and deployed that to Render immediately.
This is a habit from a previous hackathon where I left deployment to the
very end and spent the last two hours fighting environment variable
issues on the host instead of polishing the actual product. Getting the
pipeline proven working on day one, even with nothing real behind it yet,
meant every feature I added after that point just had to slot into an
already-working deploy, not fight for one.

---

## Backend plumbing

### 6. Env vars, CORS, and basic server wiring
Set up `.env` loading via dotenv for `GEMINI_API_KEYS` as a comma-
separated list (deliberately plural — wanted the option of multiple keys
from the start so a single key hitting quota mid-testing wouldn't block
me), added `cors()` and JSON body parsing middleware, and had Express
serve the `frontend/` folder statically so the whole thing ships as one
deployed service instead of two separate origins I'd have to keep in
sync.

### 7. Building a real health-check endpoint, then a second one out of paranoia
Added `/api/health` that reports back how many Gemini keys and models are
actually loaded and whether the Groq fallback is configured — the idea
being that right before a demo, or after redeploying, I can hit one URL
and know instantly whether the environment is misconfigured instead of
discovering it only once I'm mid-interview and everything's silently
falling back to nothing. Also, slightly embarrassingly, added a `/check`
alias for the exact same thing because I kept fat-fingering the real path
while testing on my phone and didn't want to think about it twice.

### 8. A fetch wrapper that can't hang forever
Ran into a real issue early on where a slow upstream API call just hung
the entire request indefinitely with no feedback — the frontend spinner
just sat there and I had no idea if it was still working or dead. Wrote a
timeout wrapper around fetch using `AbortController` so any outbound HTTP
call is guaranteed to fail fast with a clear error instead of blocking the
request thread indefinitely. This matters especially for a live voice
loop — the person on the other end is sitting there waiting on a spoken
response, and "waiting forever" is a much worse experience than "failed
after 5 seconds and gracefully recovered."

### 9. The multi-key, multi-model Gemini caller with a hard latency budget
This was genuinely the piece I iterated on the most in the whole backend.
First pass was naive — just try one key, one model, and if it fails,
throw. That was clearly not going to survive a live demo. So I rewrote it
to loop through my configured models (flash-lite first since it's faster,
falling back to flash if that's unavailable) and, within each model, loop
through all configured API keys. The part I went back and forth on the
most was the latency budget — I wanted a hard overall time ceiling
(ended up around 4.5 seconds) across the *entire* cascade of attempts, so
that if the cumulative elapsed time crosses that budget partway through
trying keys, it bails out immediately and throws rather than continuing
to burn through remaining keys and blowing the whole interview turn's
response time. Without that budget, a bad key combined with a slow
network could silently turn a 1-second turn into an 8-second one, and in
a live voice conversation that's an eternity that makes the whole thing
feel broken even though it technically "worked" eventually.

### 10. Groq as the actual fallback provider
Once Gemini is fully exhausted — every key on every model failed, or the
latency budget got blown — the exact same system/user prompt pair gets
sent to Groq instead, as a last resort before actually failing the
request. Wrote one `callLLM()` wrapper as the single entry point
everything else in the app calls, specifically so the rest of the
codebase never has to know or care which provider actually ended up
answering — it's an implementation detail fully hidden behind one
function boundary.

### 11. Making JSON parsing from the LLM actually robust
Even with JSON-mode explicitly requested, I kept running into cases where
the model would wrap its answer in markdown code fences, or occasionally
tack on a stray sentence of preamble before the actual JSON despite being
told not to. Wrote a small safe-parse helper that strips that kind of
junk out before attempting to parse, and critically, made it never throw
on genuinely malformed input — it returns a safe fallback object instead.
This sits right in the middle of a live interview turn, so a raw parse
failure crashing the request was not acceptable; it needs to degrade to
"something sensible happened" rather than a 500 error mid-conversation.

### 12. Persistence — deliberately just a JSON file, no database
Consciously decided against setting up any real database, because this is
a hackathon project judges will actually clone and run locally, and every
extra piece of infrastructure is one more thing that can fail to start on
someone else's machine. Everything — candidates, sessions, full history —
lives in a single JSON file on disk, loaded and saved through two small
functions. It's not "production grade" by any stretch, and I said as much
when asking for it, but the tradeoff of zero setup friction for anyone
running `npm install && npm start` was worth far more than any real
persistence guarantees for a 48-hour build.

### 13. Wrapping every route so a stray exception can't take the whole server down
Went through and made sure every route handler is wrapped so an
unexpected exception returns a clean JSON error response instead of an
unhandled promise rejection silently crashing the whole Node process.
Learned this one from watching the server just die mid-testing session
once with zero explanation in the terminal — added proper try/catch with
logging around every handler so a bug in one endpoint can never take down
every other endpoint along with it.

---

## The core adaptive interview loop

### 14. The actual interviewer prompt — by far the most iterated piece of the entire project
This is the one I rewrote the most times, easily five or six real
revisions before it felt right. The core ask was: on every single turn,
the model has to (1) judge the candidate's latest answer for correctness
and assign a competence score, (2) *independently* judge a confidence
score based purely on how they spoke rather than what they said — hedging
language like "umm," "I think," "maybe," "not totally sure" should lower
it even when the underlying content is fully correct, and conversely
assertive, confident phrasing should raise it even when the content is
actually wrong. I specifically asked it to call out the mismatch cases —
low confidence paired with a correct answer, or high confidence paired
with a wrong one — because that's the kind of nuance a canned "you scored
7/10" system completely misses, and it's what makes the feedback feel
like it came from a human who was actually listening rather than a
scoring rubric being mechanically applied.

On top of that: adapt difficulty on a 1–5 scale, where a strong answer
earns a harder follow-up on the exact same thread (dig deeper into the
same concept) and a weak one steps back to something simpler on that same
concept before the interview moves on. I added an explicit breadth rule
after noticing in early testing that the model would happily interrogate
one narrow sub-topic for the entire interview if I let it — so now, after
at most 2–3 consecutive questions deepening one thread, it's forced to
pivot to a different-but-related concept within the same subject.

I also had to explicitly forbid generic feedback. Early drafts kept
producing lines like "you seem weak in recursion" with zero connection to
anything the candidate had actually said, which felt hollow and
unconvincing. The fix was requiring every feedback line to paraphrase
something specific and real from their actual answer — evidence-based,
not vibes-based.

Finally, there's a short internal `interviewerNote` (kept to 5–12 words)
that's a private scratchpad jotting never shown live, only surfaced later
in the final report — and I explicitly told it not to attempt any
scoring/hiring-recommendation logic in this prompt at all, since that
needs to be one clean separate step at the very end, kept entirely out of
the live per-turn loop for latency reasons.

Output is strict JSON only, no markdown, matching an exact schema:
verdict, competence, confidence, evidenceFeedback, interviewerNote,
nextDifficulty, nextQuestion, topicTag, crossDayLink, nextAction, done.

### 15. Separating out the first-question prompt
Realized fairly early that reusing the same giant judging prompt for
question one made no sense — there's nothing to judge yet, so most of
that prompt's instructions are just dead weight burning tokens and adding
latency to the very first response of the interview, which is exactly the
moment where speed matters most for first impressions. Split it into its
own smaller, simpler prompt whose only job is generating the opener.
Starts at difficulty 2 (medium-easy) to establish a baseline before
anything adaptive kicks in, and if there's a resume or job description
pasted in, it tries to ground the very first question in something real
and specific from it — an actual project or technology they listed —
rather than opening with something generic and textbook-sounding that
could apply to literally anyone.

### 16. `/api/interview/start` — wiring the session kickoff
Takes topic, persona, experience level, interview type, resume text, job
description, and optional practice-mode fields. Creates a new session
keyed by a generated session id, resolves the candidate's key, pulls any
relevant cross-day memory for them, calls the first-question prompt, and
sends back the session id plus the first question, difficulty, and
question count metadata the frontend needs to render the room screen.

### 17. `/api/interview/answer` — the actual live loop, turn by turn
Takes the session id and whatever the candidate just said. Appends it to
the session's turn history, builds a compact version of the transcript
(explicitly not the full raw history every single time — I found that
once interviews got past around five or six questions, sending the whole
raw conversation back every turn started meaningfully increasing both
token cost and response latency), calls the LLM with the full judging
prompt, parses the response safely, stores the new turn with every field
from the schema, and returns the next question plus the updated
difficulty. There's a hard ceiling of 8 questions total regardless of
what the time-based ending logic inside the prompt itself decides, purely
as a backstop against the interview running unexpectedly long.

### 18. The compact transcript builder specifically
Wrote this as its own small, focused function after noticing the latency
creep mentioned above — it condenses the full turn history down into just
what's actually needed per turn (question, a short one-line answer
summary, verdict, topic tag) rather than the complete raw text of every
exchange. This was purely a performance fix, discovered by actually
timing requests as interviews got longer and noticing later turns were
visibly slower than earlier ones.

### 19. Deciding what happens on a blank, garbled, or clearly-misheard answer
Speech recognition isn't perfect, and I didn't want a mic hiccup or a
moment of genuine silence to unfairly tank someone's score by getting
judged as a confidently wrong answer. Added a short-circuit on the
backend: if the answer is empty, is obviously just noise, or is far too
short to meaningfully judge (under roughly three words), treat it as "no
real answer was given" rather than forcing the LLM to score nonsense as
if it were a real attempt. In that case it gently rephrases the same
question rather than moving on, and that turn doesn't get counted in a
way that unfairly drags the final score down for what was really just a
recognition glitch, not a knowledge gap.

### 20. A pass specifically on question variety, because early testing felt repetitive
After running through a handful of test interviews back to back, I
noticed the model had a tendency to fall into similar question
*patterns* even across genuinely different topics — a lot of "explain the
difference between X and Y" phrasing. Went back into the system prompt
and explicitly asked for a mix of question styles: conceptual
explanations, "what would happen if" scenario questions, "walk me through
how you'd design/debug this" open-ended ones, and the occasional direct
factual check — specifically to avoid the interview feeling
mechanically templated across different sessions.

---

## Practice drills for weak spots

### 21. Practice mode as its own dedicated branch of the interviewer prompt
Added a distinct branch for when someone wants to specifically drill one
concept they were previously weak on, rather than running a full general
interview. It's a short, focused five-question sequence, entirely on one
concept, with no topic-branching at all — I actually turned off the
breadth rule from prompt 14 here on purpose, since depth on exactly one
thing is the entire point of this mode and constantly pivoting away would
defeat it. Difficulty ramps deliberately across those five questions,
starting foundational and ending harder/applied, specifically so there's
a clean, legible before-and-after comparison at the end rather than a
noisy jumble of difficulty levels.

### 22. Making sure the practice opener actually knows the candidate's baseline
If I already have their competence score on this specific concept from a
past interview, I pass that in as context so the very first practice
question can be calibrated to roughly where they struggled last time,
instead of blindly guessing a difficulty from zero and potentially either
boring them with something too easy or overwhelming them right out of the
gate with something too hard.

### 23. Wiring "Practice Now" so the entire loop is genuinely one click
Connected both the dashboard's weak-area callout and the report screen's
"Practice Weak Areas" button so they launch a fully pre-filled practice
session — correct topic, correct baseline score, practice mode flag all
set automatically. Didn't want someone who was just told "you're weakest
at X" to then have to go manually retype "X" into a topic box themselves;
that's exactly the kind of unnecessary friction that makes a feature feel
half-finished even when the underlying logic works fine.

---

## Scoring and the final report

### 24. Deciding report scoring should be deterministic, not another model call
Seriously considered having the final report scores come from one more
LLM call summarizing the whole interview, but decided against it — an
extra call adds real latency right at the moment the candidate is
waiting most eagerly for their results, and more importantly, it makes
the scoring genuinely unpredictable and hard to defend if a judge asks
"why did this candidate get exactly this score." Instead I computed
everything directly and deterministically from the raw turn data:
competence and confidence are straightforward averages across all turns,
problem-solving weights the difficulty level of each question against
whether it was actually answered correctly (so a hard question answered
well counts for meaningfully more than an easy one), communication
blends confidence and competence with confidence weighted slightly
higher, and hiring probability is a weighted combination of all three.
The final recommendation label — Strong Hire, Hire, Leaning No Hire, No
Hire — is just clean threshold buckets off the hiring percentage. Simple,
fully explainable, reproducible, and every number in it can be traced
back to something a human could independently verify by rereading the
transcript.

### 25. Building the full report payload the frontend actually needs
Beyond just the four headline scores, the report endpoint also builds a
full timeline array for the UI to render turn-by-turn, groups average
competence by topic tag to feed the radar chart, surfaces the internal
interviewer notes (which, again, are never shown anywhere during the live
interview itself, only here), counts how many cross-day memory callbacks
actually got used, and builds a revision plan out of every topic that
wasn't answered fully correctly.

### 26. Persisting the report the instant it's generated, not waiting on anything else
Made sure the completed session gets written to the candidate's permanent
history the exact moment the report is built, as a fire-and-forget write
that's logged on failure but never blocks the actual response back to the
frontend. The reasoning: if the candidate closes the tab the second they
see their score, or their connection drops right after, the interview
still needs to count and show up in their history later — the save can't
be contingent on anything happening after the report is already in front
of them.

### 27. The clamp utility, added after a very specific bug
Added a `clamp(val, min, max, fallback)` helper directly after tracking
down a NaN that had shown up in the readiness score on the dashboard —
traced it back to one LLM-derived competence value that had come back
slightly out of the expected 0–100 range on a single turn, which then
silently poisoned an average downstream. Went through and applied clamp
everywhere a model-derived number feeds into any scoring math after that,
so a single out-of-range value from the model can never again corrupt a
larger calculation or produce something visibly broken in the UI.

### 28. Tuning the revision-plan time estimates to feel earned, not arbitrary
Originally every weak topic in the revision plan just got a flat
15-minute estimate regardless of how badly it actually went, which felt
a little arbitrary and disconnected once I looked at it critically.
Adjusted it to scale a bit with how far off the average competence on
that specific tag was from a passing threshold — a topic that was almost
right gets a shorter suggested block than one that was completely missed
— while still keeping the whole thing simple arithmetic rather than
turning it into yet another model call.

---

## Resume and job-description handling

### 29. Getting resume text out of an uploaded PDF
Added multer for the upload handling and pdf-parse for text extraction,
so someone can upload an actual PDF (or paste plain txt/md) and have the
text pulled out automatically instead of manually copy-pasting from their
resume file, which is the kind of small friction that makes a demo feel
noticeably smoother. Capped the file size at 5MB and kept it entirely
in-memory rather than writing to disk, since there's no reason to persist
the raw file once the text has been extracted from it.

### 30. Handling every realistic way a resume upload can go wrong
Went through the actual failure modes one at a time rather than just
wrapping the whole thing in one generic try/catch: a file over the size
limit, a wrong file extension entirely, a PDF that's actually a scanned
image with essentially zero extractable text (pdf-parse returns near-
empty content in that case, which needed its own specific message rather
than just silently "succeeding" with an empty textarea that looks like it
worked), and a genuinely corrupt file that throws partway through
parsing. Each of these now returns a distinct, human-readable error the
frontend can show inline, rather than one generic 500 that gives no clue
what actually went wrong or a silent failure that looks successful.

### 31. Remembering a candidate's resume across logins
Added a small endpoint that returns whatever resume text was parsed the
last time that candidate used the app, so it auto-fills on their next
session instead of asking them to re-upload the exact same file every
single time they want to start a new interview — a small thing, but it
adds up to the whole experience feeling considered rather than
disposable.

### 32. Skill-gap analysis against a pasted job description
When a job description is also provided, a separate call compares it
against the resume text and returns a structured table of which required
skills are actually evidenced in the resume, which are missing entirely,
and a short plain-English summary of overall fit for that specific role —
shown to the candidate before they even start the interview, so they walk
in already knowing roughly where they stand for that job rather than only
finding out after the fact.

---

## Cross-day memory

### 33. Bringing in embeddings for genuine semantic recall
Wanted the interviewer to be able to actually "remember" a returning
candidate's past sessions in a way that felt natural rather than just
quietly logging data nobody ever sees. Used Gemini's embedding model to
convert past interview turns and the current topic into vectors and
compare them by cosine similarity, so the system can surface the most
*semantically relevant* past concepts rather than just the most recent
ones, which often aren't the same thing at all.

### 34. A keyword-overlap fallback for when embeddings aren't available
If the embeddings API call fails for any reason — network issue, quota,
whatever — there's a fallback that does a much simpler keyword-overlap
comparison instead of just silently returning no memory at all. Not
nearly as good at finding genuinely related concepts, but a degraded
version of the feature is a much better outcome than the feature quietly
disappearing without any explanation.

### 35. Writing the actual retrieval function, and then explicitly capping it
Built a function that pulls a candidate's full session history, ranks
past question/answer/feedback entries by similarity to the current topic
(embeddings first, keyword fallback second), and returns just the top
few. I had to go back and add an explicit cap on how long the resulting
memory-context text block is allowed to get — a candidate with a dozen-
plus past sessions was starting to produce a memory block that ate a
meaningfully large chunk of the system prompt's token budget on its own,
which isn't what "top-3 most relevant" should mean. Now it's strictly
capped regardless of how much history exists.

### 36. Persisting full sessions and computing trends over time
Every completed interview gets appended into that candidate's session
history in the store, and a separate function looks back across all of
their past sessions to work out, per concept tag, whether they're
trending upward, staying flat, or actually regressing over time. This is
what powers the "AI remembers your progress" section on the dashboard —
without it, cross-day memory would just be a neat trick in the live
interview with nothing to show for it anywhere else in the product.

### 37. The three read-only candidate endpoints the frontend actually consumes
Built `summary` (basic returning-candidate info — last topic, last date,
recent tags), `history` (readiness trend, per-skill bars, weakest area,
recent sessions list), and `memory` (the concept trends from the function
above) as three clean, simple GET endpoints. The dashboard, studio, and
report screens all just call these directly and render whatever comes
back — kept the frontend intentionally dumb here so all the actual logic
lives in one place on the backend.

---

## Demo data, so the app never looks empty in front of a judge

### 38. Building a realistic-but-clearly-synthetic session generator
Didn't want the very first thing a judge sees after logging in to be a
completely blank dashboard with nothing on it — that's a genuinely bad
first impression regardless of how good the underlying product is. Built
a function that generates a handful of plausible-looking past interviews
(varied topics, dates spread across several days, believable score
trajectories) reusing the exact same schema real interviews produce, so
nothing downstream needs any special-case handling for demo versus real
data.

### 39. Seeding one demo login on server boot, and making sure it's idempotent
Wired that generator to run once at server startup for a single demo
candidate, so logging in as that user immediately shows a readiness
trend, a weakest-area callout, and recallable memory rather than a cold
empty state. Had to specifically make sure this seeding step is
idempotent — safe to run again every time the server restarts without
duplicating sessions each time — since I was restarting the server
constantly during development and definitely noticed duplicate entries
piling up before I fixed that.

---

## Frontend — getting in the door

### 40. Landing screen
Built a landing screen with a hero headline, a primary "Start Mock
Interview" call to action, a secondary "View Demo" option for someone who
just wants to look around first, and a small preview card to give a
glimpse of the actual product before committing to signing up.

### 41. Login and signup screens
Built both screens with name, email, password, and a remember-me checkbox
on login. Kept authentication entirely client-side for this build — there
was no real need for a proper backend auth system for a hackathon demo —
but structured the code so a real auth endpoint could be dropped in later
without having to touch the UI layer at all. Added a dismissible warning
area for validation errors and a simple link to switch between the two
screens.

### 42. Deciding exactly what gets persisted across a page reload
Defined precisely what gets written to local storage on login or signup —
name, email, a remembered flag — and what should happen automatically on
a fresh page load: the app should silently restore the logged-in state
and route straight to the dashboard rather than bouncing a returning user
back to the landing page on every single refresh, which felt broken the
first time I tested it without this.

### 43. The persistent nav bar
Added a top nav bar that appears once logged in — a connection-status dot
(idle, connecting, live) with a text label, a small monospace session-id
readout, and a user avatar chip showing their initial and display name.

---

## Frontend — dashboard

### 44. The readiness dashboard itself
Built the dashboard around an overall numeric readiness score with a
labeled progress track, one progress bar per tracked skill area, and a
clearly called-out "weakest area" section with a direct "Practice Now"
button that launches a practice session scoped to exactly that topic.

### 45. Explicitly designing the brand-new-user, zero-history state
Realized partway through that I'd only ever tested the dashboard with a
candidate who already had history — went back and specifically designed
what it looks like for someone brand new with nothing yet. Rather than
rendering a readiness score of 0%, empty progress bars, and a weakest-
area callout with nothing to actually call out, that entire section gets
replaced with one clear, simple prompt pointing straight at Interview
Studio. Empty charts everywhere would have looked like something was
broken rather than just new.

### 46. Returning-candidate and practice-mode banners
Added a returning-candidate welcome banner ("last time you covered X")
with recent topic tags shown as pills, and a completely separate banner
specifically for when practice mode is active, styled distinctly (a red
accent border) with its own start and cancel actions — deliberately made
visually different so it could never be confused with a regular
interview session.

### 47. Recent interviews list and the two quick-action cards
Added a recent-interviews list pulled from session history (newest first,
with a proper empty-state message for brand-new users) plus two quick-
action cards — "Start Interview" and "View History" — for the two things
someone lands on the dashboard most often wanting to do.

### 48. The memory recall card
Added a small "Your AI Interview Memory" card that lists concepts the
system recalls from past sessions along with their trend, specifically to
make the cross-day continuity feature *visible and tangible* to the
candidate, rather than something that only quietly happens behind the
scenes on the backend where nobody would ever notice it existed at all.

---

## Frontend — interview studio (setup screen)

### 49. Topic selection
Built the studio around a grid of preset subject chips for one-click
topic selection, plus a free-text custom topic input for anything not
already covered by the presets.

### 50. Experience, interview type, duration, and a live plan preview
Added three selectable option rows — experience level, interview type
(technical, behavioral, mixed, system design), and duration — along with
a live plan-preview card that recalculates in real time as any of those
change, showing an estimated technical question count, behavioral
question count, adaptive follow-up count, and a total questions/time
summary line.

### 51. Making sure the plan-card numbers don't visibly disagree with reality
Went back after initial testing and made sure the estimated counts shown
in that plan card were at least a reasonable approximation of what the
backend's actual max-questions and target-time logic would really
produce — I'd noticed the preview promising something like "10 questions"
while the real interview consistently ended at 8, which is exactly the
kind of small inconsistency that quietly erodes trust in a demo even
though the core feature works fine.

### 52. Resume input — paste and upload, side by side
Added a collapsible resume section with a textarea, a "paste from
clipboard" button (with a fallback hint for browsers that block
programmatic clipboard access), and an upload button wired to the resume-
parsing endpoint, auto-filling the textarea the moment extraction
succeeds.

### 53. Job description input and the skill-gap trigger
Added a job-description textarea with an "analyze skill gap" button that
calls the skill-gap endpoint and renders the returned comparison table
and summary directly inline, before the candidate ever starts the actual
interview.

---

## Adding the cohort-grounded mode — this is where the project got substantially bigger

This whole section happened after the base interview loop was already
solid and demo-able on its own. I wanted an entirely different mode where
the interview isn't built around a generic topic at all, but is instead
grounded in one specific candidate's *actual, real* progress through a
31-day AI/ML cohort — genuine mission data, genuine pass/fail/attempt
counts, genuine skipped days — rather than anything hypothetical.

### 54. Modeling the curriculum and candidate progress as plain data
Set up a curriculum file (day number, title, module, for every day of the
31-day program) and a separate cohort-candidates file holding each
candidate's actual per-day mission records — passed or not, number of
attempts it took, whether it was skipped entirely — plus basic profile
info like their target role and years of experience. Kept both as plain
readable JSON, same philosophy as everything else in the backend: no
database, nothing a judge couldn't open directly and understand in ten
seconds.

### 55. A simple endpoint to list the available cohort profiles
Added an endpoint that returns the list of cohort candidates on file, so
the studio screen can present an actual dropdown of "whose real progress
do you want this interview grounded in" instead of hardcoding a single
fixed demo person, which would've made the whole feature feel much less
real.

### 56. Turning raw progress data into something the model can actually reason about
Wrote a function that takes a candidate's raw mission records and turns
them into a plain-language briefing paragraph: which days they passed on
the first try, which took multiple attempts (explicitly flagged as likely
weak spots worth probing deeper), and which were skipped outright (worth
checking whether the concept got picked up some other way). This briefing
is the only thing that actually reaches the interviewer prompt — the
model never sees the raw JSON structure directly, only a readable
narrative built from it.

### 57. The cohort-grounded interviewer prompt itself
Built as a stricter sibling of the original interview prompt. Every
question now has to trace back to a specific, real day from that
candidate's actual curriculum record — no generic questions allowed at
all in this mode. It has to touch at least four distinct days across the
interview so it can't just camp on one topic, and has to ask a minimum of
eight questions total before it's allowed to end. Kept the same adaptive-
difficulty and evidence-based-feedback rules from the original prompt,
same private internal-notes concept, but every question now also tags
which curriculum day it's grounded in, so the eventual report can show
that mapping back clearly.

### 58. A separate, simpler opening-question prompt for cohort mode
Same underlying idea as the original first-question split — a distinct,
shorter prompt just for question one — except now it's specifically
instructed to open on a day that either took multiple attempts or was
skipped, since those tend to be the most genuinely interesting starting
points, and to greet the candidate by their actual name, since in this
mode we genuinely have it.

### 59. Wiring an entirely new panel into the studio screen for this mode
Added a full panel: pick a cohort candidate from the dropdown, see a
short hint summarizing their progress, and choose which flavor of
interview to run against them — standard cohort technical, a GitHub deep-
dive variant, a coding-round-only variant, or the full combined
interview. Deliberately styled this whole panel to look visually distinct
from the plain generic-topic flow, so it's immediately obvious you've
switched into a fundamentally different mode.

---

## GitHub deep-dive

### 60. Pulling a candidate's actual public repos
Added a GitHub integration that, given a username, fetches their most
recently active public non-fork repositories, capped at a small handful
so this can't blow up prompt size or take forever to fetch, and does a
best-effort pull of a short README excerpt from each one. Made the
README fetch specifically fail silently on a per-repo basis if it doesn't
exist or errors out — one repo missing a README should never block the
rest of a perfectly good profile from loading.

### 61. Handling GitHub's real failure modes explicitly, not generically
Went through the actual ways this call can fail and handled each on its
own terms: a username that simply doesn't exist gets a clear, specific
404-style message rather than a generic error; hitting GitHub's own rate
limit doesn't hard-fail the whole interview setup — it just continues
without any GitHub context and clearly tells the frontend that's what
happened, so a candidate isn't ever blocked from starting an interview
just because I hadn't configured a token.

### 62. Summarizing the pulled profile for interview prep
Once repos and README excerpts are in hand, they get sent to the LLM for
a short two-to-three sentence summary of the candidate's tech stack,
general interests, and the kind of projects they actually build — this is
exactly what shows up in the GitHub Summary card in studio, before the
interview itself even begins.

### 63. Feeding that GitHub context into the interview without touching the original function
Added a small wrapper that appends the GitHub summary onto the same
candidate-context object used for cohort grounding, rather than modifying
the original context-building function directly. Wanted this to be
strictly additive — a cohort interview run with no GitHub username
provided at all needed to behave *exactly* as it did before this feature
existed, with zero risk of a regression in the simpler path.

---

## Coding / DSA round

### 64. Being upfront from the start about what this feature actually does
Made a deliberate decision early on: there is no sandboxed code execution
anywhere in this stack, and I was not going to fake or imply otherwise.
So the coding round is explicitly reasoning-based — the model reads a
candidate's submitted code the way an experienced human interviewer would
and evaluates it by reading, not by running it. Made sure this is stated
clearly in the UI copy itself too; I didn't want it to accidentally imply
that test cases were actually executed against the code when they
weren't.

### 65. Generating a tailored coding problem on demand
Built an endpoint that takes an optional stack hint, a weak-areas hint,
and a target difficulty, and asks the model for one LeetCode-style
problem — a title, a full description including constraints and at least
one worked example, a difficulty rating, a small starter function stub in
the given language, and two to three test cases expressed as readable
input/output pairs, since there's no runner available to actually execute
against them.

### 66. Reviewing a submitted solution by reasoning, not execution
Built a second endpoint that takes the original problem plus the
candidate's submitted code and asks the model to reason carefully about
correctness against the stated test cases, estimate time and space
complexity, give short interviewer-style feedback, and suggest one
natural follow-up question — the kind of thing a real interviewer would
actually ask about edge cases or possible optimizations in a live coding
round.

### 67. A standalone panel for the coding round on its own
Added a dedicated panel — generate a problem, write a solution directly
in a textarea, submit, see the evaluation — specifically so a candidate
can exercise just the coding round on its own without necessarily having
to go through a full spoken interview first.

---

## The combined "Full Interview" flow

### 68. Deciding what "full" should actually mean before building it
Wanted a single mode that strings the cohort-grounded conversation and
the coding round together end to end, finishing with one combined report
covering both halves — rather than making a candidate manually run three
separate flows and then mentally stitch the results together themselves,
which felt like exactly the kind of friction I'd been trying to design
away from everywhere else in the product.

### 69. A slightly different conversational endpoint shape for this flow
Built this as a distinct endpoint from the original start/answer pair —
takes the running session id, the full candidate record, and the latest
message, and simply returns a reply plus a done flag and which curriculum
day was just covered, each turn. Same grounding rules as the standalone
cohort prompt (at least four distinct days, at least eight questions) but
a simpler conversational shape, since this piece was built specifically
for this combined flow rather than reusing the original adaptive loop's
shape.

### 70. Stitching conversation into coding into one combined report
This endpoint runs the cohort conversation through to natural completion,
then hands off directly into a generated coding problem, then produces a
single combined report covering both halves — deliberately reusing the
scoring approach from the original report logic rather than inventing a
second, separate scoring system, just extended to fold a code-review
score in alongside the interview-derived scores.

### 71. Explicitly checking that "full" mode didn't quietly break anything simpler
Went back through everything after this was working and specifically
verified this was purely additive — that the original start/answer/report
trio, and the plain standalone cohort mode, still behaved exactly as they
had before, completely untouched, if the full-interview UI was never
opened at all. Adding one large feature regressing something that was
already demo-ready would have been a genuinely bad trade.

### 72. Building the frontend for the combined flow
Added the combined-flow screen: the conversation renders up top, a coding
panel appears once the conversation hands off naturally, a textarea for
the solution, a submit action, and then it routes into the same report
screen used everywhere else, with both halves of the session properly
reflected in it.

---

## Back to the live room — voice, in real depth

### 73. Checking for browser support before wiring up anything else
Before writing a single line of actual voice logic, added a feature-
detection check for `SpeechRecognition` and `speechSynthesis` support
right on page load. If either is missing — which in practice means almost
anything that isn't Chromium-based — the app shows a clear, immediate
blocking message rather than letting someone click "start speaking
interview" and silently hit a dead mic with zero explanation of why
nothing is happening. This single check alone prevented more confusing
demo-day moments than anything else in the entire voice layer.

### 74. Setting up continuous listening with live interim captions
Configured recognition for continuous listening so it keeps going across
natural pauses instead of stopping after one short utterance, and turned
on interim results so partial, not-yet-final transcripts stream live into
the caption area while the candidate is still mid-sentence. This mattered
a lot for trust in a voice-only interface — seeing your own words appear
in near-real-time is a strong visual signal that the mic is actually
working, versus just hoping it's picking you up and finding out only
after you finish talking.

### 75. Silence-based auto-submit, with a manual override kept alongside it
Implemented a silence-detection timer — once recognition stops producing
new speech for roughly a second and a half to two seconds after some real
text has already accumulated, the answer is treated as complete and gets
auto-submitted. This is genuinely what made the whole thing feel like an
actual conversation instead of a walkie-talkie exchange where you have to
press a button every single turn. But I deliberately didn't rely on that
alone — kept an "I'm done, submit now" button permanently available too,
for noisy environments or for candidates who just naturally pause
mid-thought longer than the timer expects.

### 76. Handling each speech-recognition error case on its own terms
Went through the actual distinct error events one at a time rather than
lumping them into one generic handler: no speech detected just quietly
restarts listening without alarming anyone, no microphone found shows a
real and specific error since that one genuinely can't self-heal,
permission denied shows a clear message with concrete instructions on
what to click to fix it rather than a vague failure, and transient
network errors get a couple of automatic retries with backoff before
finally giving up. The interview session itself should never be torn down
just because recognition hiccuped once.

### 77. Requesting microphone permission early, before wasting an LLM call
Moved the mic permission request to happen right when the candidate
clicks "start speaking interview," before the first question is even
generated — not silently, deep inside the first `recognition.start()`
call somewhere later in the flow. If permission gets denied, this way
we've stopped before burning an actual LLM call generating a question
that literally cannot be answered by voice.

### 78. Letting the candidate genuinely interrupt the AI mid-question
If the candidate starts speaking while a question is still being read
aloud, the voiceover gets cancelled immediately and the interface switches
straight to listening mode — they're never forced to sit through the
entire question audio before being "allowed" to start answering. Out of
every small change in the voice layer, this is the one that made the
whole loop actually feel conversational rather than like navigating a
phone menu system.

### 79. Tuning voice, rate, and pitch per persona
Picked a consistent speech-synthesis voice and tuned rate and pitch
slightly differently per persona — the strict interviewer persona speaks
a touch faster and flatter, the friendly one a touch slower and warmer —
so the choice of persona is actually felt in how the AI sounds, not just
read in the text of the questions it asks. Also made sure any in-flight
utterance always gets cancelled before a new one is queued, so questions
can never overlap or stack up if a candidate answers unusually quickly.

### 80. Mute and skip controls that behave exactly as labeled
Wired "mute voiceover" to stop only future speech synthesis calls,
leaving recognition completely untouched, so a candidate can still answer
fully by voice, just without the AI talking back to them; and wired
"skip" to immediately cancel whatever's currently playing and jump
straight into listening mode, for candidates who read faster than the
synthesized voice speaks and don't want to sit through every single
question being read aloud in full.

### 81. Automatically recovering from a recognition session that silently drops
Noticed during testing that Chrome's recognition will occasionally stop
itself even with continuous mode enabled, with no visible error at all —
it just silently goes quiet. Added a listener on the end event that
automatically restarts recognition if the interview is still meant to be
active and hasn't been intentionally paused, so a silent drop mid-
interview never looks to the candidate like the app simply stopped
listening to them for no reason.

### 82. The visual difficulty ladder and its accompanying toast
Added a small visual ladder from 1 to 5 that highlights the current
difficulty level after every turn, along with a brief toast notification
whenever it changes, explaining the direction it moved and roughly why —
specifically so the adaptive behavior that's actually happening under the
hood is *visible* to the candidate, not just happening invisibly on the
backend where they'd have no way of ever noticing it.

### 83. The optional camera "vitals" panel
Added an entirely opt-in camera-monitoring panel that shows live posture,
eye-contact, and confidence "vitals" bars while the candidate is speaking,
with an explicit "on-device only, nothing leaves your browser" note
front and center, since a camera during an interview app is exactly the
kind of thing that makes people reasonably nervous if it isn't clearly
explained.

### 84. The running side timeline, cross-day badge, and exit control
Added a timeline that builds up one entry per question-and-answer turn as
the interview progresses, a small badge that appears only on turns where
a genuine cross-day memory callback was used, and a straightforward exit
control for leaving the interview early if needed.

---

## Report screen, in full

### 85. The four headline score cards
Built the report around four score cards — hiring probability,
confidence, communication, problem solving — bound directly to the
computed scores from the report payload, alongside a clear recommendation
badge.

### 86. Loading and failure states for the report call specifically
Added an explicit loading state while the report request is in flight,
since I noticed this call can genuinely take a moment right after an
eight-question interview finishes and a blank screen in that window feels
broken. Also added a specific failure state — for a stale or invalid
session link, for instance — with a clear "this report couldn't be found"
message and a way back to the dashboard, rather than a silent, dead
screen with no path forward.

### 87. The report timeline and a full-transcript toggle
Rendered the per-question timeline as the default view, with a "full
transcript" toggle that expands or collapses the complete raw question-
and-answer text, so the default view stays clean but the full detail is
always one click away for anyone who wants it.

### 88. Replay mode
Added a step-through replay mode with previous, play, and next controls
and a clear position counter, showing one question/answer/feedback set at
a time — the goal was for it to feel like scrubbing through a recording
of the actual interview rather than just scrolling a static list.

### 89. The revision plan section
Rendered the computed revision plan as a clean checklist — topic plus a
time estimate for each — with the total estimated time shown as a header
summary above the list.

### 90. Progress trend and the skill radar chart
For any candidate with more than one past session, added a progress-trend
line and an SVG radar chart across topic tags with a matching legend,
plus a "Practice Weak Areas" call to action that sends the candidate
straight back into a pre-filled practice drill targeting whichever axis
came out lowest.

### 91. Day pill, cross-day pill, and the practice-result comparison card
Added small header pills showing which day of the ongoing interview
series this session was, and how many cross-day memory callbacks got
used during it, plus a completely separate before-and-after comparison
card specifically for practice-drill sessions, so the improvement is
immediately visible at a glance rather than something you'd have to dig
through the timeline to piece together yourself.

---

## Wrapping up

### 92. The profile screen
Kept this deliberately minimal — avatar, display name, readiness number,
total interview count, sign-out action. Nothing elaborate; it mostly
exists so the nav avatar chip has somewhere real to link to.

### 93. Cleaning up secrets thoroughly before pushing anything public
Created an `.env.example` documenting the exact expected key format,
double-checked that both `.env` and the data files containing real
candidate information are properly gitignored, and specifically searched
back through the whole codebase for any hardcoded key or token I might
have pasted in during testing and forgotten to remove before making the
repository public.

### 94. Writing a README that actually holds up for someone who isn't me
Wrote up the full project structure, exact local run instructions,
every API route and what it does — including all the newer cohort,
GitHub, and coding-round endpoints on top of the original interview
ones — and the deploy steps, specifically written so it would be usable
by someone opening this repository cold, with zero prior context on how
any of it fits together.

### 95. A dedicated cross-browser and cross-device sanity pass
Beyond the main Chrome desktop flow I'd been developing against the whole
time, specifically checked mic permission behavior on Chrome for Android,
since that flow prompts differently than desktop does, and confirmed the
unsupported-browser message from prompt 73 actually triggers correctly on
Firefox and Safari instead of silently half-working in a confusing state.
Wasn't trying to fully support every browser in 48 hours — just making
absolutely sure that anything unsupported fails loudly and clearly rather
than quietly and confusingly.

### 96. A full, repeated run-through of every mode before actually submitting
Went through the entire product multiple full times, out loud, start to
finish, across every mode: a plain generic-topic interview, a cohort-
grounded interview run against a real candidate profile, the GitHub deep-
dive variant, a standalone coding round on its own, and the full combined
flow end to end. Checked the report screen carefully after each one, went
back to the dashboard afterward to confirm the new session actually
appeared in history and the readiness score updated correctly. Fixed a
handful of small empty-state and edge-case issues along the way that
only actually surfaced once real data — not the seeded demo data — was
flowing through the whole system for the first time.

---

*will keep adding to this if anything else notable comes up before
submission*
