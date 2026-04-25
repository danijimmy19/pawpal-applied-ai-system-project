# PawPal+ AI Assistant Reflection

## 1. Base project and extension

My base project was **PawPal+**, a Python pet-care scheduler with four core ideas:
- `Owner`, `Pet`, `Task`, and `Scheduler` classes
- multi-pet task management
- sorting, filtering, recurrence, and conflict checks
- a CLI and Streamlit interface

For the final project, I extended that scheduler into **PawPal+ AI Assistant**, which adds retrieval-augmented guidance, an observable agent workflow, guardrails, and an evaluation harness.

## 2. System design

The rebuilt system has five main layers:

1. **Data layer**  
   JSON files store the owner profile and a curated pet-care knowledge base.

2. **Retrieval layer**  
   TF-IDF retrieval ranks both knowledge-base chunks and schedule-derived context.

3. **Agent layer**  
   The agent classifies intent, chooses a tool, builds a trace, and performs self-critique.

4. **Guardrail layer**  
   Injection detection, medical emergency redirect, and task validation all run before risky actions.

5. **Interface layer**  
   The Streamlit app and CLI both call the same backend code.

## 3. AI collaboration

I used AI most effectively for:
- brainstorming the architecture
- identifying the right separation between retriever, agent, and schedule tools
- drafting initial regex and parsing strategies
- reviewing edge cases for guardrails and testing

A helpful AI suggestion was to make the agent output a visible action trace instead of opaque reasoning. That made the workflow easier to explain and grade.

A flawed AI suggestion was an earlier Streamlit design that mixed mutable cached objects with rerun-heavy UI code. That led to instability and blank-page failures, so I replaced it with a simpler pattern built around explicit forms, session state, and backend methods.

## 4. Reliability and evaluation

I added multiple reliability features:
- guardrails for emergency medical language
- prompt-injection detection
- structured validation for proposed tasks
- human approval before applying AI-generated tasks
- unit tests and a 12-case evaluation harness
- confidence scoring and self-critique

What surprised me most was how often ambiguous requests can still look plausible. Confidence scoring and approval gates helped prevent the system from acting too confidently on incomplete information.

## 5. Tradeoffs

### Tradeoff 1: deterministic agent vs. external LLM
I chose a deterministic agent instead of a hosted model because it is easier to reproduce and grade. The tradeoff is less language flexibility, but higher reliability.

### Tradeoff 2: local JSON persistence
Using JSON makes the system easy to inspect and run locally. The tradeoff is that it is not built for multi-user persistence or cloud deployment.

### Tradeoff 3: conservative medical behavior
The system deliberately redirects emergencies instead of trying to diagnose. This reduces risk, even though it limits how “smart” the assistant may appear.

## 6. Limitations and future improvements

Current limitations:
- intent detection is rule-based
- retrieval quality depends on the curated knowledge base
- the system does not integrate with real veterinary records
- schedule suggestions are local and single-user only

Future improvements:
- upgrade the retriever to hybrid retrieval with metadata filters
- support multiple owners and cloud persistence
- add calendar export
- extend the evaluation harness with expected-action labels
- add a stronger explanation layer comparing retrieved chunks against final answers

## 7. Ethical considerations

This assistant could be misused if someone treated it like a diagnostic system. To reduce that risk, I added:
- explicit emergency redirects
- confidence scoring
- conservative medical wording
- approval before task creation

The key lesson from this project is that an AI system is not only about generating an answer. It also needs structure, checks, observability, and safe failure behavior.