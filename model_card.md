# Model Card — PawPal+ AI Assistant

## 1. Model / System Overview

**Name:** PawPal+ AI Assistant  
**Version:** 1.0  
**Type:** Hybrid AI application with retrieval, agentic tool use, and configurable model backends  
**Primary task:** Pet-care scheduling assistance, grounded pet-care Q&A, and schedule-aware planning  
**Project context:** Final project extension of the original PawPal+ pet-care scheduler

PawPal+ AI Assistant is not a single standalone foundation model. It is a **hybrid AI system** that combines:
- a structured scheduling backend (`Owner`, `Pet`, `Task`, `Scheduler`)
- a retrieval layer over curated pet-care knowledge and live schedule context
- an agent/controller layer that interprets requests, selects tools, and produces structured outputs
- optional reasoning/generation backends:
  - deterministic local logic
  - Gemini API
  - Ollama local models

Because the system can run with multiple backends, this model card describes the **application-level AI behavior** rather than claiming characteristics of one proprietary model.

---

## 2. Intended Use

### Primary intended uses
PawPal+ AI Assistant is designed to help users:
- manage pet-care schedules across multiple pets
- create or review tasks from natural-language requests
- retrieve grounded pet-care information from a curated knowledge base
- check conflicts and find open time slots
- receive safer, structured assistance for routine planning tasks

### Example intended prompts
- “Schedule a 20 minute play session for Miso every day at 6:30 PM.”
- “What should I focus on today for Nova?”
- “Find the next available 30 minute slot for Clover on Sunday.”
- “Check whether any of today’s tasks conflict.”

### Out-of-scope uses
This system is **not** intended for:
- emergency veterinary diagnosis
- medication dose changes without veterinary approval
- replacing professional veterinary care
- autonomous decision-making without user review in high-stakes situations
- general open-domain medical advice

---

## 3. Model / Backend Options

PawPal+ AI Assistant supports multiple configurable backends.

### Retrieval backends
- **TF-IDF Retriever**
  - local, lightweight, reproducible
  - keyword/term-weighted retrieval over curated knowledge base and schedule context

- **Gemini Embeddings Retriever**
  - semantic retrieval using Gemini embeddings when API credentials are available
  - used to improve recall beyond lexical matching

### Generation / reasoning backends
- **Deterministic local backend**
  - rule-based and structured
  - strongest for reproducibility and offline demos

- **Gemini backend**
  - cloud-based generation / JSON-capable synthesis
  - useful when stronger language understanding is needed

- **Ollama backend**
  - locally hosted model inference
  - useful for privacy-preserving or offline-capable demos, subject to local model availability and performance

---

## 4. Inputs and Outputs

### Inputs
The system accepts:
- natural-language scheduling requests
- pet-care questions
- schedule review requests
- structured scheduler state (owners, pets, tasks, due dates, priorities, recurrence)
- optional backend/model selection from the user

### Outputs
The system may produce:
- natural-language grounded responses
- structured task proposals
- schedule-aware recommendations
- conflict warnings
- next-slot suggestions
- agent traces / tool-use summaries
- confidence scores or self-critique summaries
- safety redirects in medical/emergency scenarios

---

## 5. Training / Data Sources

This project does **not** train or fine-tune a new base model.

Instead, the system uses:
- a **curated pet-care knowledge base** created for the application
- live scheduler/task context from the user’s data
- optional third-party model APIs or local model runtimes for inference only

### Knowledge sources in the application
- routine pet-care guidance categories such as feeding, exercise, grooming, medication reminders, and warning signs
- current schedule/task data stored in JSON for the PawPal+ system
- model-generated reasoning constrained by prompt templates, tool routing, and guardrails

### Important note
When Gemini or Ollama is selected, the application depends partly on those external or local model capabilities. Their underlying training data is not controlled by this project.

---

## 6. System Architecture Summary

At a high level, the system follows this flow:

1. user enters a request through Streamlit UI or CLI  
2. input is checked by validation and safety guardrails  
3. the agent/controller decides whether to retrieve context, call tools, or both  
4. retrieval gathers relevant knowledge-base content and live schedule context  
5. the selected backend generates or structures a response  
6. the scheduler tools can apply or inspect task state  
7. the result is returned with structured output and, when enabled, trace/confidence information

---

## 7. Safety and Guardrails

PawPal+ AI Assistant includes several reliability features:

- input validation
- prompt-injection detection
- medical-advice safety redirect
- structured JSON output checks
- task proposal validation
- iteration caps / bounded agent steps
- self-critique and confidence scoring
- evaluation harness for predefined scenarios

### Medical safety behavior
The system is designed to redirect users toward veterinary care for emergency indicators such as:
- collapse
- trouble breathing
- seizures
- severe bleeding
- poisoning concerns
- other urgent warning signs

This is a deliberate safety constraint and not a failure mode.

---

## 8. Performance Characteristics

### Strengths
- integrates retrieval with scheduler actions rather than acting as a generic chatbot
- can operate with local-only configurations
- provides observable tool use and structured outputs
- supports multiple model/retrieval backends for flexibility
- includes automated tests and scenario-based evaluation

### Tradeoffs
- TF-IDF retrieval is lightweight but less semantically robust than embeddings
- Gemini improves semantic retrieval/generation but requires API access
- Ollama improves privacy/local control but may be slower and depends on local hardware/model availability
- deterministic mode is reproducible but less expressive than model-based generation

---

## 9. Evaluation

The system is evaluated using:
- unit tests for scheduler and assistant behavior
- scenario-based evaluation harness with predefined inputs and expected behaviors

### What evaluation checks
- end-to-end scheduling behavior
- grounded schedule-aware responses
- guardrail activation for safety-sensitive prompts
- structured output validity
- core scheduler functionality such as conflict checking and slot finding

### What evaluation does not fully prove
- clinical correctness for veterinary advice
- robustness against all prompt-injection variants
- fairness across all writing styles, dialects, or ambiguous inputs
- consistent semantic quality across all possible Ollama/Gemini model variants

---

## 10. Limitations

Known limitations include:
- not a substitute for licensed veterinary judgment
- depends on the quality and scope of the curated knowledge base
- retrieval may miss context if the phrasing is highly unusual
- local models may vary significantly in quality and speed
- confidence scores are heuristic and should not be interpreted as calibrated probabilities
- structured extraction can still fail on highly ambiguous user requests
- emergency detection is rule-based and may not cover every possible formulation

---

## 11. Ethical Considerations

This project was designed with several ethical considerations in mind:

- **Safety:** avoids presenting itself as a veterinarian or emergency authority
- **Privacy:** supports fully local usage through deterministic logic and Ollama
- **Transparency:** exposes traces, tool calls, and structured outputs to make behavior more observable
- **Human oversight:** keeps the user in control of scheduler actions and encourages escalation to professionals for high-stakes issues

Potential misuse includes:
- over-relying on the system for medical interpretation
- treating heuristic outputs as professional recommendations
- using incomplete or inaccurate pet/task data and assuming the output is still correct

Mitigations include guardrails, validation, and explicit scope limitations.

---

## 12. Recommended User Guidance

Users should:
- verify important schedule changes before applying them
- use the assistant for planning support, not veterinary diagnosis
- seek urgent veterinary care for emergency symptoms
- prefer deterministic/local modes for reproducibility or privacy-sensitive demos
- prefer retrieval-backed modes when grounded answers are important

---

## 13. Maintenance and Versioning

Because this is a hybrid application:
- behavior may change when swapping model backends
- Gemini behavior may vary with provider-side model changes
- Ollama behavior may vary by local model choice and version
- retrieval quality may improve or regress as the knowledge base changes