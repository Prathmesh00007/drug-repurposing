# 🧬 Drug Repurposing Platform

A hybrid route-based drug repurposing platform that converts structured user intent into ranked shortlists and decision-ready reports using multi-agent reasoning pipelines.

📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Value Proposition](#value-proposition)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Available Routes](#available-routes)
- [Data Sources](#data-sources)
- [Contributing](#contributing)
- [License](#license)

***

## 🎯 Overview

Traditional drug repurposing workflows suffer from **low reproducibility**, **data fragmentation**, and **overwhelming outputs**. Our platform addresses these challenges by providing:

- **Fixed, repeatable workflows** that mirror real pharma decision gates (science, clinical, legal, business)
- **Hybrid reasoning** combining deterministic retrieval (auditability) with LLM synthesis (speed)
- **Decision-ready outputs** including ranked shortlists + saved decision reports
- **Cross-functional alignment** with evidence traced across domains

### Target Users

- Pharma/biotech translational scientists
- Computational biologists
- Therapeutic area leads
- Portfolio & business development teams
- Cross-functional review committees

***

## ✨ Key Features

### 🔄 Route-Based Workflows

| Route | Intent Type | Description |
|-------|-------------|-------------|
| **Route A** | Market-Back | Start with indication → find best drug candidates |
| **Route B** | Asset-Forward | Start with drug asset → find new indications |
| **Route C** | Debate Mode | Generate pro/con cases for hypotheses |
| **Route E** | White-Space | Identify underexplored therapeutic areas |
| **Route LLM** | Natural Language | Freeform queries auto-routed via Cerebras LLM |

### 🤖 Multi-Agent Architecture

- **Literature Agent**: PubMed E-utilities + BioBERT embeddings
- **Target Discovery Agent**: OpenTargets + UniProt + STRING-DB
- **Drug Candidate Agent**: ChEMBL + DrugBank + PubChem
- **Clinical Trials Agent**: ClinicalTrials.gov + EU CTR
- **Safety/Tox Agent**: FDA FAERS + AEOLUS + ToxCast
- **Patent Landscape Agent**: Lens.org + Google Patents
- **Regulatory Agent**: FDA Orange Book + EMA + DailyMed
- **IQVIA Market Agent**: Commercial data & market sizing

### 🎛️ Customization & Cost Control

- **Reasoning modes**: Low/Moderate/High (temperature control)
- **Database selection**: Choose which sources to query per run
- **Result limits**: Configure max targets/drugs to fetch
- **Safe defaults**: Moderate reasoning + recommended databases for general users
- **Power controls**: Advanced users can tune depth and budget

### 📊 Decision-Ready Outputs

- Ranked shortlist with evidence scores
- Saved decision report (PDF/JSON)
- Traceable reasoning steps
- Cross-functional evidence capture (science + clinical + IP + business)
- Comparable outputs across teams and time

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI 0.104+
- **Orchestration**: LangGraph (stateful multi-agent workflows)
- **LLMs**: Cerebras Llama 3.1, Google Gemini, Claude (Anthropic), OpenAI GPT, Ollama (local)
- **Embeddings**: BioBERT, PubMedBERT, SBERT
- **Async**: HTTPX, Asyncio, Tenacity (retry logic)
- **Validation**: Pydantic

### Frontend
- **Framework**: React 18+, TypeScript
- **Styling**: TailwindCSS, Framer Motion
- **State Management**: React Hooks
- **HTTP Client**: Fetch API

### Databases
- **Graph DB**: Neo4j 5.0+ (knowledge graph for drug-target-disease relationships)
- **Relational DB**: PostgreSQL (run state, user configs, reports)
- **Vector DB**: Pinecone/Qdrant (semantic search over biomedical literature)
- **Cache**: Redis (API rate limiting + result caching)

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **Reverse Proxy**: Nginx
- **Cloud**: AWS/GCP/Azure (flexible deployment)
- **CI/CD**: GitHub Actions

### External APIs (90+ data sources)

#### Literature & Publications
PubMed, PubMed Central, Semantic Scholar, Europe PMC, Dimensions, Cochrane

#### Drug & Compound Databases
DrugBank, ChEMBL, PubChem, DrugCentral, ZINC, BindingDB, SIDER, Repurposing Hub

#### Clinical Trials
ClinicalTrials.gov, EU Clinical Trials, WHO ICTRP

#### Targets & Proteins
OpenTargets, UniProt, STRING, Protein Data Bank, AlphaFold DB, BioGRID, IntAct, TTD

#### Disease & Phenotype
DisGeNET, OMIM, Orphanet, MONDO, Human Phenotype Ontology

#### Genomics & Expression
ClinVar, gnomAD, COSMIC, TCGA, GTEx, GEO, Expression Atlas, LINCS

#### Pathways & Systems
KEGG, Reactome, WikiPathways, Pathway Commons

#### Safety & Toxicology
FDA FAERS, AEOLUS, OFFSIDES, ToxCast, Tox21, CompTox

#### Regulatory & Approval
RxNorm, DailyMed, FDA Orange Book, FDA Purple Book, EMA Database

#### Patents & IP
Lens.org, Google Patents, SureChEMBL

#### Pharmacology
PharmGKB, Drugs.com, PubChem BioAssay, STITCH

***

## 💡 Value Proposition

### For Pharma R&D Teams

1. **Lower time-to-shortlist**: Compress weeks of manual literature/data stitching into hours
2. **Repeatable & comparable**: Same intent run by different analysts produces consistent outputs
3. **Cross-functional alignment**: Single artifact for debate across science/clinical/IP/strategy
4. **Audit trail**: Saved configs + reasoning steps for governance and institutional memory

### For Computational Biologists

1. **Hybrid trust + speed**: Deterministic retrieval (traceable) + LLM synthesis (fast hypothesis generation)
2. **Customizable depth**: Tune reasoning strength, database selection, and result limits
3. **Cost control**: Pay only for evidence depth needed; avoid "query everything" waste

### For Portfolio & BD Teams

1. **Decision-ready outputs**: Ranked shortlist + decision report ready for review committees
2. **Business feasibility built in**: Market sizing, IP/FTO signals, and competitive landscape integrated
3. **Risk-adjusted recommendations**: Safety flags, regulatory precedent, and clinical feasibility gated early

***

## 📦 Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Neo4j 5.0+
- PostgreSQL 15+

### 1. Clone Repository

```bash
git clone https://github.com/your-org/drug-repurposing-platform.git
cd drug-repurposing-platform
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env

# Edit .env with your API keys
nano .env
```

**Required environment variables:**

```bash
# LLM APIs
CEREBRAS_API_KEY=csk-xxxxx
OPENAI_API_KEY=sk-xxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxx
GOOGLE_API_KEY=AIza-xxxxx

# Databases
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
POSTGRES_URI=postgresql://user:pass@localhost:5432/repurposing

# External APIs (optional, most are free/public)
NCBI_API_KEY=xxxxx  # For higher PubMed rate limits
IQVIA_API_KEY=xxxxx  # Commercial data (paid)
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env

# Edit API endpoint
nano .env
```

**Frontend .env:**

```bash
VITE_API_URL=http://localhost:8000
```

### 4. Database Setup

```bash
# Start Neo4j + PostgreSQL with Docker
docker-compose up -d neo4j postgres redis

# Run migrations
cd backend
alembic upgrade head

# (Optional) Seed knowledge graph with base ontologies
python scripts/seed_knowledge_graph.py
```

### 5. Start Services

**Backend:**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm run dev
```

**Access:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474

***

## 🚀 Usage

### 1. Natural Language Query (LLM Route)

```bash
curl -X POST http://localhost:8000/api/v1/route-llm/parse \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find repurposing candidates for Type 2 Diabetes in South Asia",
    "include_reasoning": true
  }'
```

**Response:**
```json
{
  "indication": "Type 2 Diabetes Mellitus",
  "geography": "South Asia",
  "confidence": "high",
  "reasoning": "Query explicitly mentions Type 2 Diabetes and South Asia region"
}
```

### 2. Market-Back (Route A)

```bash
curl -X POST http://localhost:8000/api/v1/route-a/run \
  -H "Content-Type: application/json" \
  -d '{
    "indication": "Inflammatory Bowel Disease",
    "geography": "United States",
    "max_targets": 5,
    "max_drugs_per_target": 3,
    "reasoning_mode": "moderate",
    "selected_databases": ["pubmed", "chembl", "opentargets", "clinical_trials", "drugbank"]
  }'
```

**Response:**
```json
{
  "run_id": "run-abc123",
  "status": "running"
}
```

**Poll for completion:**
```bash
curl http://localhost:8000/api/v1/route-a/run/run-abc123
```

**Fetch final results:**
```bash
curl http://localhost:8000/api/v1/route-a/run/run-abc123/state
```

### 3. Asset-Forward (Route B)

```bash
curl -X POST http://localhost:8000/api/v1/route-b/run \
  -H "Content-Type: application/json" \
  -d '{
    "drug_name": "Metformin",
    "known_indications": ["Type 2 Diabetes"],
    "max_new_indications": 10,
    "reasoning_mode": "high"
  }'
```

***

## 📚 API Documentation

### Interactive Docs
Visit http://localhost:8000/docs for Swagger UI with live testing.

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/route-llm/parse` | Parse natural language query with LLM |
| `POST` | `/api/v1/route-a/run` | Execute market-back workflow |
| `GET` | `/api/v1/route-a/run/{run_id}` | Poll run status |
| `GET` | `/api/v1/route-a/run/{run_id}/state` | Fetch final results |
| `POST` | `/api/v1/route-b/run` | Execute asset-forward workflow |
| `POST` | `/api/v1/route-c/run` | Execute debate mode |
| `GET` | `/api/v1/datasets` | List available data sources |
| `GET` | `/health` | Health check |

***

## 🗺️ Available Routes

### Route A: Market-Back
**Start with indication → find best drugs**

- Literature analysis (unmet needs, current therapies)
- Target discovery (disease-associated proteins/pathways)
- Drug candidate ranking (approved drugs, investigational, preclinical)
- Clinical trial precedent
- Safety/toxicity screening
- Patent landscape & regulatory feasibility
- Market sizing & competitive analysis

### Route B: Asset-Forward
**Start with drug → find new indications**

- Drug mechanism analysis
- Target repurposing (alternative disease associations)
- Phenotype matching (similar diseases)
- Clinical trial opportunities
- Patent white-space analysis

### Route C: Debate Mode
**Test hypothesis with pro/con cases**

- Generate evidence for both sides
- Confidence scoring
- Risk assessment
- Alternative scenarios

### Route E: White-Space
**Identify underexplored opportunities**

- Low-competition therapeutic areas
- Orphan disease mapping
- Patent gap analysis
- Market opportunity sizing

### Route LLM: Natural Language
**Freeform queries auto-routed**

- Cerebras LLM extracts intent
- Auto-maps to appropriate route
- Guided parameter extraction

***

## 🗄️ Data Sources

### Free & Open Sources (90% of platform)
- PubMed, PubMed Central, Europe PMC
- ChEMBL, PubChem, DrugBank (academic license)
- OpenTargets, UniProt, STRING, Reactome
- ClinicalTrials.gov, EU Clinical Trials, WHO ICTRP
- FDA FAERS, AEOLUS, ToxCast, Orange Book
- Lens.org (patents), Google Patents
- DisGeNET, OMIM, Orphanet, GTEx
- And 60+ more...

### Commercial/Paid (Optional)
- IQVIA (market data & analytics)
- Cortellis (drug pipeline intelligence)
- Clarivate Integrity (R&D database)
- Dimensions (research with grants/trials)
- COSMIC (cancer mutations)

**Cost optimization:** Users can toggle databases per run to control API spend.

***

## 📊 Example Outputs

### Ranked Drug Shortlist
```json
{
  "indication": "Inflammatory Bowel Disease",
  "geography": "United States",
  "candidates": [
    {
      "rank": 1,
      "drug_name": "Infliximab",
      "mechanism": "TNF-alpha inhibitor",
      "approval_status": "Approved (Crohn's, UC)",
      "evidence_score": 0.92,
      "clinical_precedent": "15 Phase III trials",
      "safety_flags": "Low (established safety profile)",
      "patent_status": "Off-patent, biosimilars available",
      "market_potential": "$2.3B (US market)",
      "reasoning": "Strong clinical evidence, approved indication, proven efficacy..."
    },
    {
      "rank": 2,
      "drug_name": "Tofacitinib",
      "mechanism": "JAK inhibitor",
      "approval_status": "Approved (UC), Investigational (Crohn's)",
      "evidence_score": 0.87,
      "clinical_precedent": "8 Phase III trials",
      "safety_flags": "Moderate (boxed warning for infections)",
      "patent_status": "Patent expires 2028",
      "market_potential": "$1.8B (projected)",
      "reasoning": "Emerging evidence for Crohn's, oral administration advantage..."
    }
  ],
  "report_url": "/reports/run-abc123.pdf",
  "reasoning_steps": [...]
}
```

***

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes with tests
4. Run tests: `pytest` (backend), `npm test` (frontend)
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Standards

- **Backend**: Black (formatting), Ruff (linting), MyPy (type checking)
- **Frontend**: ESLint, Prettier
- **Commits**: Conventional Commits format
- **Tests**: >80% coverage required

***

## 🐛 Known Issues & Roadmap

### Current Limitations
- Route A polling can take 5-10 minutes for complex queries
- Some commercial APIs require paid subscriptions
- Knowledge graph seeding takes ~2 hours for full ontologies

### Roadmap
- [ ] Real-time streaming results (SSE/WebSockets)
- [ ] Saved query templates & reusable workflows
- [ ] Team collaboration features (shared reports, comments)
- [ ] Fine-tuned domain LLMs for better extraction
- [ ] Mobile-responsive UI
- [ ] Multi-language support (currently English only)
- [ ] GraphQL API alongside REST

***

## 📄 License

This project is licensed under the MIT License.

***

## 🙏 Acknowledgments

- **Data Sources**: NCBI, EBI, Broad Institute, FDA, EMA, and 90+ public databases
- **LLM Providers**: Cerebras, OpenAI, Anthropic, Google
- **Open Source**: LangGraph, FastAPI, React, Neo4j community

***

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-org/drug-repurposing-platform&type=DateBuilt with ❤️ for accelerating drug discovery and improving patient outcomes.**
