# 📡 3GPP RAG Assistant

> **An evidence-grounded Retrieval-Augmented Generation (RAG) assistant for querying 3GPP telecom standards using hybrid retrieval, reranking, and a local LLM.**

The **3GPP RAG Assistant** is a domain-specific AI assistant designed to answer questions from **3GPP technical specifications**.

The current knowledge base includes:

- **3GPP TS 38.300** — NR; Overall description; Stage-2
- **3GPP TS 38.331** — NR; Radio Resource Control (RRC) protocol specification

The system combines:

```text
Document Parsing
      ↓
Section Extraction
      ↓
Chunking
      ↓
Dense Embeddings ─────┐
                      ├── Hybrid Retrieval
BM25 Retrieval ───────┘
      ↓
Identifier-Aware Search
      ↓
Cross-Encoder Reranking
      ↓
Evidence Selection
      ↓
Local LLM
      ↓
Guardrails
      ↓
Grounded Answer
```

The main objective is to provide **accurate, evidence-grounded answers while reducing LLM hallucinations**.

---

# 🚀 Key Features

- 📚 3GPP technical-document knowledge base
- 📄 DOCX document parsing
- 🧩 Section-aware chunking
- 🔎 Dense semantic retrieval
- 🔤 BM25 keyword retrieval
- 🔀 Hybrid search
- 🎯 Telecom identifier detection
- 🧠 Domain relevance scoring
- 🔝 Cross-encoder reranking
- 🛡️ Evidence-based guardrails
- 🤖 Local LLM inference
- 💾 FAISS vector database
- 🧪 Retrieval evaluation
- ❌ Negative-query testing
- 📌 Evidence/source-aware generation
- 🔒 Local/offline-capable architecture

---

# 🎯 Problem Statement

3GPP specifications are extremely large and technically complex.

An engineer looking for information such as:

```text
What is T300 expiry?

What is T311 expiry?

What happens when T311 expires?

What is RRCSetupRequest?

What happens during RRC Re-establishment?

What is the purpose of RRCRelease?
```

may need to manually search through hundreds of pages.

A traditional LLM also has a problem:

```text
Question
   ↓
LLM
   ↓
Answer
```

The answer may be based on the model's pretrained knowledge and can potentially contain hallucinated or outdated information.

This project instead uses:

```text
Question
   ↓
Retrieve relevant 3GPP evidence
   ↓
Rerank evidence
   ↓
Give evidence to LLM
   ↓
Generate grounded answer
```

---

# 🧠 Why RAG?

RAG stands for:

> **Retrieval-Augmented Generation**

Instead of relying entirely on the LLM's internal knowledge, the system retrieves relevant information from the 3GPP knowledge base before generating the answer.

```text
                  User
                   │
                   ▼
             User Question
                   │
                   ▼
          Retrieval Pipeline
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
   FAISS Search            BM25 Search
       │                       │
       └───────────┬───────────┘
                   ▼
            Hybrid Candidates
                   │
                   ▼
              Reranker
                   │
                   ▼
            Relevant Evidence
                   │
                   ▼
               Local LLM
                   │
                   ▼
             Guardrails
                   │
                   ▼
             Final Answer
```

---

# 📚 Knowledge Base

The current project contains:

```text
data/
├── 38300-j10.docx
├── 38331-j10.docx
├── chunks.json
└── sections.json
```

The primary specification for RRC-related questions is:

```text
3GPP TS 38.331
NR; Radio Resource Control (RRC) protocol specification
```

The project also contains:

```text
3GPP TS 38.300
NR; Overall description; Stage-2
```

This allows the assistant to answer questions using information from multiple telecom specifications.

---

# 📊 Current Knowledge Base

The processed TS 38.331 document contains approximately:

| Component | Value |
|---|---:|
| Document | TS 38.331 |
| Sections | 636 |
| Total words | ~610,535 |
| Chunks | 1,639 |
| Vector DB | FAISS |
| Keyword Retrieval | BM25 |

These numbers depend on the exact 3GPP document revision and processing configuration.

---

# 🏗️ Project Architecture

```text
3GPP Documents
      │
      ▼
┌──────────────────────────┐
│       Ingestion          │
│                          │
│ parser.py                │
│ chunker.py               │
│ build_index.py           │
└────────────┬─────────────┘
             │
             ▼
      sections.json
      chunks.json
             │
      ┌──────┴───────┐
      │              │
      ▼              ▼
   FAISS            BM25
  Dense Search   Keyword Search
      │              │
      └──────┬───────┘
             ▼
       Hybrid Search
             │
             ▼
     Identifier Search
             │
             ▼
      Domain Scoring
             │
             ▼
        Reranker
             │
             ▼
      Evidence Selection
             │
             ▼
        Local LLM
             │
             ▼
       Answer Generator
             │
             ▼
        Guardrails
             │
             ▼
       Final Response
```

---

# 📁 Project Structure

The current repository structure is:

```text
3GPP-RAG/
│
├── data/
│   ├── 38300-j10.docx
│   ├── 38331-j10.docx
│   ├── chunks.json
│   └── sections.json
│
├── models/
│   └── <local-model-files>
│
├── src/
│   │
│   ├── evaluation/
│   │
│   ├── generation/
│   │   └── answer_generator.py
│   │
│   ├── guardrails/
│   │
│   ├── ingestion/
│   │   ├── build_index.py
│   │   ├── chunker.py
│   │   └── parser.py
│   │
│   ├── llm/
│   │   └── local_llm.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py
│   │   ├── custom_extractor.py
│   │   ├── dense_retriever.py
│   │   ├── domain_scorer.py
│   │   ├── hybrid_search.py
│   │   ├── identifier.py
│   │   ├── reranker.py
│   │   └── search.py
│   │
│   └── __init__.py
│
├── tests/
│
├── vectorstore/
│
├── .env
├── app.py
├── llm_generator.py
├── main.py
├── README.md
└── requirements.txt
```

---

# 🔄 Data Ingestion Pipeline

The ingestion pipeline converts the original 3GPP DOCX files into searchable data.

```text
3GPP DOCX
    ↓
parser.py
    ↓
sections.json
    ↓
chunker.py
    ↓
chunks.json
    ↓
build_index.py
    ↓
FAISS + Retrieval Index
```

---

# 📄 1. Document Parser

File:

```text
src/ingestion/parser.py
```

Responsibilities:

- Read DOCX files
- Extract headings
- Extract paragraphs
- Preserve section structure
- Clean document text
- Generate structured section data

Output:

```text
data/sections.json
```

Example conceptual structure:

```json
{
  "section_id": "5.3.3",
  "title": "RRC connection establishment",
  "text": "..."
}
```

---

# ✂️ 2. Chunker

File:

```text
src/ingestion/chunker.py
```

The document is too large to pass directly to an LLM.

Therefore, it is divided into smaller chunks.

Current development configuration:

```text
Chunk size: 300 words
Overlap: 50 words
```

Example:

```text
Chunk 1
Word 1 → Word 300

Chunk 2
Word 251 → Word 550

Chunk 3
Word 501 → Word 800
```

The overlap helps preserve context across chunk boundaries.

Output:

```text
data/chunks.json
```

---

# 🗃️ 3. Index Builder

File:

```text
src/ingestion/build_index.py
```

This step prepares the searchable indexes.

Conceptually:

```text
chunks.json
     │
     ├───────────────┐
     ▼               ▼
Embeddings          BM25
     │               │
     ▼               ▼
  FAISS          Keyword Index
```

---

# 🔎 Retrieval System

The retrieval system is located at:

```text
src/retrieval/
```

It contains several retrieval components.

---

# 🧠 Dense Retrieval

File:

```text
src/retrieval/dense_retriever.py
```

Dense retrieval converts the user question into an embedding and searches the FAISS vector database.

The current embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Example:

```text
"What is T311 expiry?"
          ↓
      Embedding
          ↓
      FAISS Search
          ↓
Relevant Chunks
```

---

# 🔤 BM25 Retrieval

File:

```text
src/retrieval/bm25_retriever.py
```

BM25 performs keyword-based retrieval.

This is especially useful for telecom identifiers.

Examples:

```text
T300
T301
T311
T319
RRCSetup
RRCRelease
RRCReconfiguration
```

For example:

```text
"What is T311 expiry?"
```

BM25 can directly match:

```text
T311
```

inside the specification.

---

# 🔀 Hybrid Search

File:

```text
src/retrieval/hybrid_search.py
```

The project combines:

```text
Dense Retrieval
       +
BM25 Retrieval
       ↓
Hybrid Candidate Set
```

This gives the system both:

### Semantic understanding

and

### Exact keyword matching

This is particularly useful for telecom standards.

---

# 🆔 Identifier Detection

File:

```text
src/retrieval/identifier.py
```

Telecom questions often contain identifiers.

Examples:

```text
T300
T311
RRCSetup
RRCReestablishment
RRCRelease
```

The system identifies these terms and can use them to improve retrieval.

Example:

```text
User Query:

What is T311 expiry?

Detected Identifier:

T311
```

This helps prevent confusion between similar timers.

---

# 🎯 Domain Scoring

File:

```text
src/retrieval/domain_scorer.py
```

The domain scorer helps prioritize chunks that are more relevant to telecom terminology and the specific query.

Conceptually:

```text
Candidate Chunk
      ↓
Domain Relevance
      ↓
Technical Relevance
      ↓
Final Retrieval Score
```

---

# 🔍 Custom Extraction

File:

```text
src/retrieval/custom_extractor.py
```

This component can be used for extracting specific technical information from retrieved results.

Examples:

```text
Timer values
Identifiers
Sections
Technical terms
Procedure names
```

---

# 🔝 Reranking

File:

```text
src/retrieval/reranker.py
```

Initial retrieval may return many candidates.

Example:

```text
Hybrid candidates: 15
```

The reranker evaluates the relationship between:

```text
Query
   +
Candidate Document
```

and assigns relevance scores.

```text
15 Candidates
      ↓
Cross Encoder
      ↓
Relevance Scores
      ↓
Top Relevant Evidence
```

---

# 🔎 Search Orchestration

File:

```text
src/retrieval/search.py
```

This component can coordinate the retrieval pipeline.

Conceptually:

```text
Query
 ↓
Identifier Detection
 ↓
Dense Retrieval
 ↓
BM25 Retrieval
 ↓
Hybrid Search
 ↓
Domain Scoring
 ↓
Reranking
 ↓
Final Context
```

---

# 🤖 Local LLM

File:

```text
src/llm/local_llm.py
```

The assistant is designed to support local LLM inference.

This means the architecture can work without sending the retrieved telecom content to an external LLM API.

Recommended lightweight model:

```text
Qwen2.5-1.5B-Instruct-GGUF
```

The official Qwen GGUF repository provides several quantizations including:

```text
Q2_K
Q3_K_M
Q4_0
Q4_K_M
Q5_0
Q5_K_M
Q6_K
Q8_0
```

The official Qwen repository provides local usage instructions for llama.cpp, Ollama and llama-cpp-python.

---

# 📥 Download Local Model

## Recommended Model

**Qwen2.5-1.5B-Instruct-GGUF**

[Download Qwen2.5-1.5B-Instruct-GGUF from Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF?utm_source=chatgpt.com)

For a balance between size and quality, use:

```text
Q4_K_M
```

If you want a slightly larger quantized file with potentially better quality:

```text
Q5_0
```

The official Q5_0 file is approximately **1.26 GB**.

---

# 📥 Download Model Using Hugging Face

Install:

```bash
pip install -U huggingface_hub
```

Then download the Q4_K_M model:

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
qwen2.5-1.5b-instruct-q4_k_m.gguf \
--local-dir models \
--local-dir-use-symlinks False
```

For Q5_0:

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
qwen2.5-1.5b-instruct-q5_0.gguf \
--local-dir models \
--local-dir-use-symlinks False
```

After downloading:

```text
models/
└── qwen2.5-1.5b-instruct-q4_k_m.gguf
```

or:

```text
models/
└── qwen2.5-1.5b-instruct-q5_0.gguf
```

---

# ⚠️ Important: Do NOT Upload the Model to GitHub

The model is large and should not normally be committed to the Git repository.

Your GitHub repository should contain:

```text
models/
```

but not the actual large model file.

For example:

```text
models/
└── .gitkeep
```

Then the user downloads the model separately.

Add to `.gitignore`:

```gitignore
models/*.gguf
models/*.bin
models/*.safetensors
```

This keeps the GitHub repository lightweight.

---

# 🖥️ Running the Local Model

If using llama.cpp, the official Qwen repository documents local GGUF execution.

For example:

```bash
llama-cli -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

The exact command depends on the llama.cpp installation and your `local_llm.py` implementation.

---

# 🦙 Ollama Alternative

If you prefer Ollama, Qwen's official GGUF repository also documents running the model through Ollama.

Example:

```bash
ollama run hf.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M
```

This provides another way to run the LLM locally.

---

# ✍️ Answer Generation

File:

```text
src/generation/answer_generator.py
```

The answer generator receives:

```text
User Question
+
Retrieved Evidence
+
System Instructions
```

and generates the final answer.

Example:

```text
Question:
What is T311 expiry?

Retrieved Evidence:
3GPP TS 38.331 relevant sections

        ↓

Local LLM

        ↓

Grounded Answer
```

---

# 🛡️ Guardrails

Directory:

```text
src/guardrails/
```

The guardrail layer is responsible for reducing unsupported answers.

The principle is:

```text
No Evidence
     ↓
Do NOT confidently answer
```

Instead, the assistant should be able to respond:

```text
I could not find sufficient evidence in the
provided 3GPP specifications to answer this reliably.
```

This is particularly important for telecom standards.

---

# 🚫 Negative Query Testing

A major feature of the project is testing questions for which the knowledge base does not contain sufficient evidence.

Example:

```text
What is T999 expiry?
```

If T999 does not exist in the loaded specification, the assistant should not invent a definition.

Expected behavior:

```text
Insufficient evidence found in the provided
3GPP knowledge base.
```

This tests the hallucination resistance of the RAG system.

---

# 🧪 Evaluation

Evaluation code is located under:

```text
src/evaluation/
```

and additional tests are located under:

```text
tests/
```

Recommended test categories:

| Test Type | Example |
|---|---|
| Timer | What is T300 expiry? |
| Timer | What is T311 expiry? |
| RRC | What is RRCSetup? |
| Procedure | Explain RRC establishment |
| Identifier | What does T311 mean? |
| Negative | What is T999 expiry? |
| Cross-section | Explain RRC re-establishment |

---

# 📈 Evaluation Metrics

The system can be evaluated using:

### Retrieval Recall@K

Whether the correct evidence appears in the top K results.

### Precision@K

How many retrieved chunks are actually relevant.

### MRR

Mean Reciprocal Rank of the correct evidence.

### Answer Faithfulness

Whether the final answer is supported by the retrieved context.

### Hallucination Rate

How frequently the model generates unsupported information.

---

# 🧪 Example Query

```text
User:

What is T311 expiry?
```

Pipeline:

```text
                T311 expiry
                     │
                     ▼
             Identifier Detection
                     │
                     ▼
             ┌───────────────┐
             │ Dense Search  │
             └───────┬───────┘
                     │
             ┌───────▼───────┐
             │  BM25 Search  │
             └───────┬───────┘
                     │
                     ▼
             Hybrid Candidates
                     │
                     ▼
                Reranking
                     │
                     ▼
             Relevant Evidence
                     │
                     ▼
                Local LLM
                     │
                     ▼
                Guardrails
                     │
                     ▼
                Final Answer
```

---

# 💻 Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/3GPP-RAG.git
```

```bash
cd 3GPP-RAG
```

---

# 2. Create Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

---

# 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Add Local Model

Download the Qwen GGUF model using the instructions above.

Place it inside:

```text
models/
```

Example:

```text
models/
└── qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Then configure the model path according to your `src/llm/local_llm.py`.

---

# 5. Prepare Documents

Place the 3GPP documents inside:

```text
data/
```

Example:

```text
data/
├── 38300-j10.docx
└── 38331-j10.docx
```

---

# 6. Run Data Ingestion

If another user clones this repository and wants to rebuild the knowledge base, they can run the ingestion pipeline.

### Parse documents

```bash
python src/ingestion/parser.py
```

### Create chunks

```bash
python src/ingestion/chunker.py
```

### Build indexes

```bash
python src/ingestion/build_index.py
```

The generated files include:

```text
data/
├── sections.json
└── chunks.json
```

and the vector database is stored under:

```text
vectorstore/
```

---

# ▶️ Running the Application

Depending on the configured application entry point:

```bash
python app.py
```

or:

```bash
python main.py
```

The project also contains:

```text
llm_generator.py
```

for the LLM generation workflow.

Use the entry point configured in the current implementation.

---

# 🔐 Environment Variables

If your implementation uses environment variables, create:

```text
.env
```

Example:

```env
MODEL_PATH=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
VECTORSTORE_PATH=vectorstore
CHUNKS_PATH=data/chunks.json
SECTIONS_PATH=data/sections.json
```

Never commit API keys or passwords.

---

# 🚫 `.gitignore`

Recommended `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environment
.venv/
venv/
env/

# Environment variables
.env

# Local models
models/*.gguf
models/*.bin
models/*.safetensors
models/*.pt
models/*.pth

# Vector database
# Uncomment if vectorstore is generated locally
# vectorstore/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary files
*.tmp
*.temp

# Jupyter
.ipynb_checkpoints/

# Test/cache
.pytest_cache/
.coverage
htmlcov/

# Build
build/
dist/
*.egg-info/
```

---

# 📦 What Should Be Committed to GitHub?

Recommended:

```text
src/
tests/
app.py
main.py
llm_generator.py
requirements.txt
README.md
.gitignore
```

Potentially commit:

```text
data/chunks.json
data/sections.json
vectorstore/
```

if their size is acceptable and redistribution of the underlying source-derived data is permitted.

Do **not** commit:

```text
.venv/
.env
large GGUF model files
private credentials
temporary files
```

---

# 🌐 Deployment Architecture

For deployment, the recommended architecture is:

```text
                 GitHub Repository
                        │
                        ▼
                 Application Code
                        │
                        ▼
                Deployment Server
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
      RAG Pipeline                Local LLM
          │                           │
          ▼                           ▼
    FAISS + BM25                 GGUF Model
          │                           │
          └─────────────┬─────────────┘
                        ▼
                    API / UI
```

---

# 🔒 Local AI Architecture

The major advantage of the local setup is:

```text
User
 │
 ▼
RAG Application
 │
 ├── FAISS
 ├── BM25
 ├── Reranker
 ├── Evidence Guard
 │
 ▼
Local LLM
 │
 ▼
Answer
```

The LLM does not need to be a cloud API.

This makes the architecture suitable for environments where telecom documentation should remain within the local environment.

---

# ⚡ Performance Considerations

The main components affecting performance are:

```text
Embedding Model
        ↓
FAISS Search
        ↓
BM25 Search
        ↓
Reranker
        ↓
LLM Generation
```

The local LLM is generally the most computationally expensive generation component.

Using a quantized GGUF model reduces memory requirements compared with loading the original full-precision model.

---

# 🔮 Future Improvements

Planned improvements include:

- [ ] Multi-version 3GPP support
- [ ] Automatic specification version detection
- [ ] Better section hierarchy preservation
- [ ] Parent-child retrieval
- [ ] Query rewriting
- [ ] Query expansion
- [ ] Better identifier extraction
- [ ] Improved reranking
- [ ] RAGAS evaluation
- [ ] Automated hallucination evaluation
- [ ] Answer confidence score
- [ ] Source citation in every response
- [ ] FastAPI backend
- [ ] Web-based chat interface
- [ ] Docker deployment
- [ ] GPU inference
- [ ] Multi-user support
- [ ] Authentication
- [ ] Telecom troubleshooting mode
- [ ] Network log analysis
- [ ] Support for additional 3GPP specifications

---

# 🎓 Interview Explanation

If asked:

> **"Explain your 3GPP RAG project."**

A good answer is:

> I developed a domain-specific RAG assistant for 3GPP telecom standards. I used TS 38.300 and TS 38.331 as the knowledge sources. The documents are first parsed into structured sections and then divided into overlapping chunks. For retrieval, I use both dense semantic search with FAISS and lexical search with BM25. This is important because telecom documents contain exact identifiers such as T300, T311 and RRC message names. I then combine the retrieval results and use identifier detection, domain scoring and a reranker to select the most relevant evidence. The selected evidence is passed to a local Qwen-based LLM for answer generation. Finally, guardrails check whether the answer is supported by the retrieved evidence, reducing hallucinations. I also test negative questions to make sure the system does not invent information that is not present in the 3GPP knowledge base.

---

# 💼 Resume Description

### 3GPP RAG Telecom Assistant

> Developed a domain-specific RAG assistant for 3GPP TS 38.300/38.331 using DOCX parsing, structure-aware chunking, FAISS dense retrieval, BM25 lexical search, hybrid retrieval, telecom identifier detection, domain scoring, cross-encoder reranking, local LLM inference, and evidence-based guardrails to reduce hallucinations and improve technical answer reliability.

---

# 🧰 Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Document Processing | python-docx |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Keyword Search | BM25 |
| Reranking | Cross Encoder |
| LLM | Qwen2.5-1.5B-Instruct GGUF |
| Local Inference | llama.cpp / compatible runtime |
| Data | JSON |
| Testing | Pytest |
| Interface | Python application |
| Version Control | Git/GitHub |

---

# ⚠️ Limitations

### 1. Specification Version

Answers depend on the exact version of the 3GPP documents used during ingestion.

### 2. Retrieval Quality

If the correct evidence is not retrieved, the LLM may not be able to answer correctly.

### 3. LLM Limitations

A local LLM can still generate unsupported information, which is why the project includes evidence guardrails.

### 4. Hardware

Local LLM performance depends on available RAM, CPU and GPU resources.

### 5. 3GPP Document Rights

Users are responsible for ensuring that their use, storage and redistribution of 3GPP documents complies with applicable 3GPP terms.

---

# 📜 License

The software code in this repository can be licensed separately from the 3GPP documents.

The 3GPP specifications are subject to their applicable terms and conditions.

Do not assume that making this repository public automatically grants redistribution rights for the underlying 3GPP documents.

---

# 👨‍💻 Author

**Prashant Darshanwar**

AI & Data Science Engineer

Interests:

```text
Artificial Intelligence
Machine Learning
Generative AI
RAG
LLMs
Computer Vision
Telecom AI
Data Science
```

---

# ⭐ Project Highlights

```text
636 Sections
      ↓
~610K Words
      ↓
1,639 Chunks
      ↓
Dense + BM25
      ↓
Hybrid Retrieval
      ↓
Identifier Detection
      ↓
Domain Scoring
      ↓
Reranking
      ↓
Local LLM
      ↓
Evidence Guardrails
      ↓
Grounded Answer
```

---

# 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/3GPP-RAG.git

# 2. Enter project
cd 3GPP-RAG

# 3. Create environment
python -m venv .venv

# 4. Activate - Windows
.venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Download the local Qwen GGUF model
pip install -U huggingface_hub

huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
qwen2.5-1.5b-instruct-q4_k_m.gguf \
--local-dir models \
--local-dir-use-symlinks False

# 7. Add 3GPP documents to data/

# 8. Rebuild knowledge base if required
python src/ingestion/parser.py
python src/ingestion/chunker.py
python src/ingestion/build_index.py

# 9. Start application
python app.py
```

---

# 📌 Local Model Download

**Official model:**

[Qwen2.5-1.5B-Instruct-GGUF — Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF?utm_source=chatgpt.com)

**Recommended:** `Q4_K_M` for a smaller local deployment.

**Alternative:** `Q5_0` for a larger quantized model; the official Qwen file is approximately 1.26 GB.

**Do not upload the GGUF model to GitHub.** Keep it in `.gitignore` and let users download it separately.

---

# 📡 Final Architecture

```text
             ┌─────────────────────────┐
             │     3GPP TS 38.300      │
             │     3GPP TS 38.331      │
             └────────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    INGESTION    │
                 │                 │
                 │ Parser          │
                 │ Chunker         │
                 │ Index Builder   │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   KNOWLEDGE     │
                 │      BASE       │
                 │                 │
                 │ JSON + FAISS    │
                 │ BM25            │
                 └────────┬────────┘
                          │
                          ▼
                   User Question
                          │
                          ▼
                 ┌─────────────────┐
                 │    RETRIEVAL    │
                 │                 │
                 │ Dense           │
                 │ BM25            │
                 │ Identifier      │
                 │ Domain Score    │
                 │ Hybrid Search   │
                 └────────┬────────┘
                          │
                          ▼
                    ┌───────────┐
                    │ Reranker  │
                    └─────┬─────┘
                          │
                          ▼
                   Relevant Evidence
                          │
                          ▼
                 ┌─────────────────┐
                 │    LOCAL LLM    │
                 │                 │
                 │ Qwen2.5 1.5B   │
                 │ GGUF            │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   GUARDRAILS    │
                 │                 │
                 │ Evidence Check  │
                 │ Hallucination   │
                 │ Protection      │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  FINAL ANSWER   │
                 │                 │
                 │ Grounded in     │
                 │ 3GPP Evidence   │
                 └─────────────────┘
```

**Retrieve → Rerank → Ground → Generate → Validate**

This is the core architecture of the **3GPP RAG Assistant**.#   R A G _ A s s i s t a n t  
 