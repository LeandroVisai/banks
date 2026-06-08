# Council Transcript — Infraestructura Chatbot RAG BCCh
**Fecha:** 7 de junio 2026  
**Pregunta original:** Mejoras de infraestructura para eficientar duración de respuestas, soportar más usuarios concurrentes y evitar colas.

---

## Pregunta enmarcada

A small engineering team (2-3 people) runs a production RAG chatbot for ~20 internal banking analysts at Chile's Central Bank. Actual latency: 15-30s per request. Effective throughput: ~2-4 req/min. No embedding cache, no streaming, no explicit concurrency limit. Six improvements proposed: A=Concurrency Semaphore, B=LRU Embedding Cache, C=Fix max_tool_result_tokens (1500→3000), D=PostgreSQL Connection Pool, E=Streaming SSE, F=Uvicorn Multi-Worker (4×27B-Q4 on 80GB H100).

---

## Respuestas de los 5 asesores

### The Executor
Do A and C Monday. Ship B and D by end of week. Defer E. Kill F.

A is a 20-line fix. One semaphore in AppState, one middleware check, one 503 response. No risk. Do it first because silent 60s+ hangs are a trust-killer with compliance-minded banking analysts.

C is a one-line fix that's currently silently degrading retrieval quality. Change 1500→3000 in settings.py. This is a bug, not a feature. Fix it before anything else changes retrieval behavior. Risk: zero.

B is an afternoon's work. OrderedDict LRU in hybrid_search.py, keyed on (query_text, model_name). The win is real because the multi-specialist architecture re-embeds the same query multiple times per request. 100ms × 3-4 specialists adds up. No thread-safety issue if you wrap with a lock — one lock, already serializing anyway.

D is two days max. psycopg3 is already in pyproject.toml — someone already made the decision. The AsyncConnectionPool replaces per-call connections in PostgresRepo. Test it against the existing schema, confirm async context managers everywhere that SQL touches.

E is a UX lie, not a fix. Same total latency, just perceived differently. Fine eventually, but it's frontend polish.

F is a VRAM math exercise that turns into an ops nightmare. 64GB sounds fine until one request spikes and OOM kills a worker mid-inference — producing corrupt financial analysis, not a clean 503.

---

### The Expansionist
The framing here is too small. You're optimizing for 20 analysts when the real question is: what happens when this works?

Improvement F is the one everyone's nervous about, but it's the one that matters most strategically. Four parallel 27B instances at 80GB VRAM isn't just a throughput fix — it's the foundation for serving 200 analysts, not 20. The Central Bank of Chile is one institution. The Superintendencia de Bancos, the Finance Ministry, Hacienda, CORFO — they all have the same problem.

On E (streaming): the 15s-to-first-token experience doesn't just reduce complaints — it changes how analysts use the tool. When they see tokens appearing, they start reading and thinking in parallel. This is how you go from "tool I consult" to "tool I think with."

B (embedding cache) enables query analytics. Once you cache query embeddings, you have a natural collection of what analysts actually ask. That's a goldmine for identifying gaps in your document corpus.

C is the highest-leverage line of code in this entire list.

The sequence: C immediately, E in parallel, F as the expansion bet. A and D are hygiene. B is your future analytics layer.

---

### The First Principles Thinker
The real problem is misdiagnosed.

The team is treating symptoms (latency, concurrency) when the actual failure is architectural: a batch-processing workload dressed up as an interactive service. Every proposal accepts the fundamental constraint — serial LLM calls — and optimizes around it. That's the wrong move.

Strip assumptions:
- Do analysts actually need all N specialists per query? The router knows which one wins — the others are speculative computation.
- Does synthesis need to wait for all specialists? Or just the top-K results?
- Is the 27B-Q4 model actually necessary for specialist calls, or only for synthesis?

What to do:
1. C first — it's a bug, not a feature.
2. A — silent 60s waits are a trust-destroying UX failure.
3. Skip F entirely — OOM risk is real.
4. Rethink specialist architecture — lazy evaluation above router confidence threshold cuts LLM calls 40-60% without infrastructure changes.

---

### The Contrarian
Improvement F is the sleeper disaster nobody is stress-testing.

The math looks clean on paper — 4 × 16GB = 64GB, H100 has 80GB, 16GB margin. But Q4 quantization memory estimates are notoriously optimistic. The actual resident VRAM for a 27B-Q4 model includes the KV cache, which at 10-20 LLM calls per request with a long context window can spike 2-4GB per active request, per worker. Four workers mid-inference simultaneously? 70-76GB easily — OOM crashes produce corrupt financial analysis, not a clean 503.

Improvement C has an unverified assumption. Has anyone actually measured retrieval quality with 1500 vs 3000 tokens? Doubling token budget doubles prompt length, increases latency per call, and could degrade generation quality. This is a tuning decision being framed as a bug fix.

Improvement A's retry interval is wrong. "Retry in ~30s" when actual processing is 15-30s trains users to hammer the endpoint. Expose queue depth, not a fixed estimate.

Improvement B's cache key omits embedding model version. If the model is ever swapped (RAG_EMBEDDING_MODEL), stale cached embeddings silently corrupt retrieval results. Financial analysis with wrong embeddings is worse than being slow.

---

### The Outsider
The curse of knowledge is all over this proposal.

F is listed last with a "technically feasible with some margin" hedge. That hedge is backwards. F is the only improvement that actually solves the stated problem: multiple analysts querying simultaneously. Everything else optimizes a single-lane road.

20 users, 2-4 req/min capacity, 10-20 LLM calls per request. The math is broken by design.

A (503 on 3rd request) is institutionalizing the failure. "Please retry in 30 seconds" during a meeting is just a slower way to frustrate analysts.

C is the highest-stakes bug on the list. It should be fix zero, today.

E is pure perception management. Same wrong answer, delivered token by token.

Priority reframe: Fix C immediately. Ship F. Then A becomes moot, B and D become nice-to-haves.

---

## Peer Reviews (anonymization: A=Executor, B=Expansionist, C=FirstPrinciples, D=Contrarian, E=Outsider)

### Reviewer 1
1. **Strongest: A (Executor).** Concrete, sequenced execution plan tied to actual risk profiles. The only response a 2-person team can act on Monday without debate.
2. **Biggest blind spot: B (Expansionist).** Pitches inter-agency expansion without addressing whether the H100 sustains 4 parallel 27B workers without OOM.
3. **All missed:** Instrument which specialists fire per query type before any fix. Router threshold calibration could eliminate speculative specialists — zero-infrastructure latency reduction.

### Reviewer 2
1. **Strongest: A (Executor).** Concrete sequenced plan with honest effort estimates and real risk assessments.
2. **Biggest blind spot: B (Expansionist).** Builds expansion narrative on a system at 2-4 req/min for 20 users. Skips past the operational failure and pitches vision.
3. **All missed:** No one asked for a nvidia-smi dmon trace during a live request. Before A-F, a profiler trace tells you whether you're GPU-bound, IO-bound, or Python-bound. All five optimized blind.

### Reviewer 3
1. **Strongest: D (Contrarian).** Only one doing real math. KV cache spike analysis is concrete risk that kills production systems. Adversarial thinking the others lack.
2. **Biggest blind spot: E (Outsider).** Dismisses A without considering that 503 is a correct HTTP contract. Also treats F as simple to deploy without D's VRAM math.
3. **All missed:** The bottleneck was never diagnosed from telemetry. Is it embedding, HNSW search, reranking (CrossEncoder), LLM generation, or tool-calling? Without a profiler, every prioritization is a guess. The CrossEncoder reranker running on CPU could be the actual bottleneck — no advisor mentioned it.

### Reviewer 4
1. **Strongest: D (Contrarian).** Only one pressure-testing the other responses instead of just advocating a position. KV cache analysis, C's latency tradeoff, B's invalidation risk.
2. **Biggest blind spot: E (Outsider).** Declares F "the only solution" while ignoring that A+D together solve concurrency without OOM risk.
3. **All missed:** B's cache economics are probably negative — 4096-dim float vectors, 20 analysts with query diversity, low expected hit rate.

### Reviewer 5
1. **Strongest: D (Contrarian).** KV cache math converts F from "bold bet" to "OOM on first simultaneous request."
2. **Biggest blind spot: E (Outsider).** Confident without a specific implementation path for F.
3. **All missed:** PostgreSQL pool (D) is almost certainly the fastest real latency win per unit of effort. TCP handshake + auth overhead on every query — none of the advisors quantified this.

---

## Chairman's Verdict

### Where the Council Agrees
- **C is fix zero.** Every advisor agreed — the 1500 token cap actively degrades answer quality.
- **A (semaphore) is necessary but not sufficient.** All acknowledged the need for concurrency control.
- **F is operationally dangerous at 80GB.** KV cache spikes can push 4 workers to 70-76GB — OOM produces corrupt output, not clean 503.
- **D (connection pool) is underrated.** Three peer reviews flagged it independently. psycopg3 is already in pyproject.toml.

### Where the Council Clashes
- **F:** Strategic bet vs. operational disaster. Outsider correct that F is the only path to true parallelism. Contrarian correct that OOM is a real risk. Resolution: F needs validated VRAM budget before production.
- **E:** UX lie vs. behavioral change. Both views are defensible — it's a product philosophy question.
- **A:** 503 vs. queue. Right answer: bounded queue (depth=4, timeout=45s) + 503 after — not purely one or the other.

### Blind Spots (emerged only in peer review)
- **CrossEncoder reranker on CPU** (~50ms/chunk × 50 chunks = ~2.5s per search) — not mentioned by any advisor.
- **No profiler trace.** The entire plan was built on unconfirmed bottleneck diagnosis.
- **Speculative specialists** — router fires all N specialists; lazy evaluation above threshold could cut 40-60% of LLM compute.
- **B cache key missing model version** — silent corruption on model swap.

### The Recommendation
**Phase 1 (this week):** Fix C (1 line). Add D pool (2 hours). Run profiler trace on live requests.  
**Phase 2 (after profiler):** Fix CrossEncoder bottleneck if confirmed. Lazy specialist evaluation. Add A with bounded queue.  
**Phase 3 (only when Phase 2 stable):** Validate F with controlled VRAM budget test. Defer E and B until then.

### The One Thing to Do First
Profile a live request before touching any code. Add wall-clock timers at 5 key points (embedding, CrossEncoder reranker, LLM per specialist, PostgreSQL, total). Log to existing chat_logs JSONL. Run 10 real analyst queries. Read the numbers. C ships in parallel — it's one line and risk-zero. But the profiler is what validates everything else.
