# ChatDKU

ChatDKU is a conversational-guide implementation for prospective students, applicants, parents, and other visitors who want to learn about Duke Kunshan University (DKU). The code supports questions about academics, student life, programs, campus resources, and the application journey. It searches the documents supplied by the person running it and gives answers grounded in those materials.

## Who it is designed for

ChatDKU is designed for people exploring DKU, including:

- Prospective undergraduate students
- Applicants and their families
- School counselors and educators
- Visitors learning about the DKU community

## What the agent can answer

Once configured with documents and model services, the agent can explore questions such as:

- What academic programs and courses are available?
- What is student life like at DKU?
- Which campus resources can support students?
- Where can I learn more about a program, activity, or university service?

For a strong answer, include the program, degree level, or topic you mean. For example, ask “What resources are available for undergraduate students interested in computer science?” rather than “What resources are available?”

## How answers are produced

ChatDKU searches the DKU documents configured by the people running it using both meaning-based and exact-term search. It checks whether the retrieved material directly supports the question before responding. When the available material does not provide a reliable answer, it will say so and ask for more context instead of guessing.

Information can change. Use ChatDKU as a starting point, then confirm important admissions, academic, financial, and policy details with the relevant official DKU office or published university materials.

## About this repository

This repository contains the public ChatDKU agent harness. It is a local-first implementation of document ingestion, retrieval, and the conversational agent workflow. It is intended for learning, evaluation, and adaptation; it is not the production ChatDKU backend or an application-submission system.

The repository provides code and documentation only. It does not provide a hosted chat service, DKU source documents, model endpoints, API keys, or other credentials. Anyone running it supplies their own documents and compatible language-model and embedding services.

Maintainers can find operational documentation in [the ingestion guide](https://chatgpt.com/c/docs/ingestion.md) and [the agent guide](https://chatgpt.com/c/docs/agent.md).

## Improvements Implemented

### 1. Source Attribution

Retrieval results now include source metadata alongside the retrieved passages, including source filenames or paths, page numbers when available, URLs when available, and retrieval scores. The answer-generation instructions were also updated to encourage a `Sources:` section listing relevant filenames and page numbers when metadata is available.

**Motivation:** Help users identify the documents behind an answer and verify important information.

**Limitation:** Including source metadata does not guarantee that every generated claim is fully supported by the cited passages. Citation accuracy requires further evaluation.

### 2. Agent Observability

A run summary is logged at the end of an agent execution. It records the original query, rewritten query, tools used, Program Gate decision, individual stage timings, and total execution time.

**Motivation:** Make the agent's execution process more transparent and help identify stages that contribute most to latency.

## Evaluation

### 1. Citation Test

**Test question:**

`What is the tuition for Duke Kunshan University for 2025-2026?`

**Observed result:**

The system returned tuition figures for international and Chinese students and cited:

- `ug_bulletin_2025-2026.pdf`, page 32
- `ug_bulletin_2025-2026.pdf`, page 33

**Result:** The expected answer structure and source references appeared in this manual test.

**Limitation:** This was a single test and does not establish citation accuracy across different questions or documents.

### 2. Observability Test

The same question was used to inspect the agent's run summary.


| Metric        | Observed result  |
| ------------- | ---------------- |
| Tool used     | `keyword_search` |
| Program Gate  | `ALLOW`          |
| Query Rewrite | 0.03 s           |
| Planner       | 0.10 s           |
| Program Gate  | 0.00 s           |
| Synthesis     | 3.36 s           |
| Total runtime | 3.49 s           |


**Result:** The logs exposed the rewritten query, selected tool, gate decision, stage timings, and total runtime.

**Limitation:** These timings represent one run and may vary depending on model-server load, hardware, and runtime conditions.

## Known Limitations and Future Work

- **ToolMemory compression:** The compression path is temporarily bypassed to avoid a runtime error involving dataclass serialization. Tool history is still recorded, but automatic compression is not currently exercised. The underlying issue should be fixed and tested before relying on long-running conversations.
- **Citation evaluation:** Test more questions to determine whether cited passages support the corresponding claims.
- **Performance evaluation:** Conduct repeated runs and evaluate a larger set of questions.
- **Retrieval evaluation:** Compare retrieval quality using a labeled set of questions and relevant evidence.

## Proposed Future Improvement: An Integrated Academic and Career Navigator

From the perspective of a first-year student, university information is often scattered across academic catalogs, department websites, research groups, campus organizations, and career platforms. I envision evolving ChatDKU beyond simple question answering into an **integrated academic and career navigator**. For example, when a student expresses an interest in computer science, the system could organize a personalized roadmap covering required and elective courses, prerequisites, timetable conflicts, undergraduate research opportunities, academic lectures, student organizations, foundational books and scholars, relevant internships, potential employers, and emerging industry trends. Rather than presenting disconnected search results, it could explain how these resources relate to one another and help students explore pathways aligned with their interests and goals.

This vision could build on ChatDKU's existing retrieval and agent architecture through multi-source retrieval, structured resource metadata, and goal-oriented planning. Reliable implementation would require authoritative and up-to-date university information, structured timetable data for conflict detection, and clear source citations and freshness indicators for time-sensitive opportunities. This is a **proposed future direction, not a currently implemented feature**. Its goal is to help students move from reactive information searching to proactive academic exploration and career planning, while supporting rather than replacing official academic advising.

## Running the Project

Follow the project's setup instructions and configure a compatible language-model endpoint and embedding provider before running the agent.

Example query:

```bash
uv run agent "What is ChatDKU?"

```

For detailed execution logs:

```bash
export CHATDKU_LOG_LEVEL=DEBUG
uv run agent "What is the tuition for Duke Kunshan University for 2025-2026?"

```

The language-model endpoint and model identifier depend on the local setup. Do not commit the actual `.env` file or API credentials.

## Reproducibility and Configuration

- Keep local credentials and machine-specific settings in `.env`.
- Use `.env.example` as a template for configuration variables.
- Keep generated local indexes, model caches, and unnecessary private documents out of version control unless the task explicitly requires them.
- Use consistent model and embedding configurations when comparing evaluation runs.

