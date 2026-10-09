Domain-Specific Generative AI Chatbot
A professional Streamlit chatbot that answers questions using a curated knowledge base. It implements a retrieval-augmented generation (RAG) workflow with source references, optional semantic embeddings, a lightweight TF-IDF fallback, and an OpenAI-compatible LLM interface.

Demo knowledge domain: AI & Data Science. Upload documents from another domain to adapt it to your coursework, organization, or project.

Features
Polished, responsive Streamlit interface with chat history and sidebar controls.
Upload PDF, DOCX, TXT, Markdown, CSV, JSON, Python, HTML, and YAML documents.
Extract, chunk, and index source documents; PDF citations retain page numbers.
Semantic retrieval with sentence-transformers/all-MiniLM-L6-v2, with TF-IDF fallback if the model cannot load.
Optional answer generation using OpenAI or an OpenAI-compatible endpoint (including supported local endpoints).
Grounded prompts that ask the model to cite passages and admit when evidence is missing.
Source passages and relevance scores shown for each answer.
Fine-tuning dataset preparation utility that converts question/answer CSV rows into chat-format JSONL and validates the output.
Tests, Dockerfile, GitHub Actions workflow, and deployment instructions.
Project structure
domain_specific_genai_chatbot/
├── app.py
├── requirements.txt
├── README.md
├── Dockerfile
├── runtime.txt
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── src/
│   ├── ingestion.py
│   ├── rag.py
│   ├── llm.py
│   └── fine_tuning.py
├── data/
│   ├── sample_knowledge_base.md
│   └── fine_tuning_examples.csv
├── scripts/
│   ├── prepare_finetuning_data.py
│   └── start_openai_finetuning.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_rag.py
│   └── test_fine_tuning.py
└── .github/workflows/tests.yml
Requirements
Python 3.12 recommended.
Internet access the first time the semantic embedding model is downloaded.
An API key for synthesized LLM answers. Without one, the app still indexes and retrieves source passages. A compatible local or hosted OpenAI-style endpoint can also be configured.
Run locally on Windows
Open Command Prompt or PowerShell in this project directory and run:

py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
If py -3.12 is unavailable, install Python 3.12 first or use a compatible Python 3.11/3.12 environment. Streamlit will print a local URL, usually http://localhost:8501.

Configure the LLM
The app works in retrieval-only mode without a key. To enable generated answers, set environment variables before launching Streamlit or enter the settings in the sidebar:

PowerShell example

$env:OPENAI_API_KEY = "YOUR_API_KEY"
$env:OPENAI_MODEL = "gpt-4o-mini"
streamlit run app.py
Optional variables:

OPENAI_BASE_URL: custom OpenAI-compatible endpoint, including its /v1 path where required.
OPENAI_MODEL: model identifier accepted by that provider.
For Streamlit Community Cloud, add OPENAI_API_KEY, OPENAI_MODEL, and optionally OPENAI_BASE_URL to the app's Secrets settings. Do not upload a real .streamlit/secrets.toml file or hard-code keys in source code. The file .streamlit/secrets.toml.example is only a template.

Local endpoint note: a URL such as http://localhost:11434/v1 refers to the machine running the Streamlit process. It will not refer to your own PC when the app is deployed to a cloud service.

Use the chatbot
Launch the app and choose a knowledge domain in the sidebar.
Keep the included sample knowledge base for a quick test, or upload your domain-specific reference files.
Click Add documents to knowledge base.
Select semantic embeddings for meaning-based retrieval or TF-IDF for a lightweight index.
Ask a question in the chat box and inspect the citations and source passages below the response.
Add an LLM API key and compatible model settings to generate a synthesized response from the retrieved passages.
Supported files: PDF, DOCX, TXT, MD, CSV, JSON, PY, HTML, YAML, and YML. Each upload is limited to 15 MB. Scanned image-only PDFs may have no selectable text and should be OCR-processed before uploading.

Prepare fine-tuning data
The repository includes a small example CSV. Replace it with examples reviewed for your own domain before using any training service.

python scripts/prepare_finetuning_data.py data/fine_tuning_examples.csv data/fine_tuning_examples.jsonl
The CSV should contain question and answer columns; prompt/completion or input/response are also supported. The script outputs one JSON object per line with chat messages and checks the JSONL format.

To optionally submit a training job through the OpenAI API, first check the provider's current model eligibility, format requirements, and pricing. A dry run is the default and does not submit anything:

python scripts/start_openai_finetuning.py data/fine_tuning_examples.jsonl --model YOUR_SUPPORTED_BASE_MODEL
Only add --submit when you deliberately want to upload the file and create a potentially billable job. Set OPENAI_API_KEY in your environment first. The exact model identifier and minimum data requirements depend on the provider and can change.

Important: preparing a dataset is not itself model fine-tuning. The training script starts a job only when explicitly passed --submit; after training, configure the resulting model ID in the chatbot's model field if the provider supports that workflow. The app uses RAG regardless of whether a separately trained model is configured.

Run tests
Install the runtime requirements, then run:

python -m unittest discover -s tests -v
python -m compileall -q app.py src scripts tests
The tests cover text chunking, supported-file validation, TF-IDF retrieval, citations, fallback behavior, and fine-tuning JSONL validation. Semantic model downloading and third-party API calls are not required by the unit tests.

Deploy to Streamlit Community Cloud
Create a GitHub repository and upload all project files.
Open Streamlit Community Cloud and create a new app connected to your repository.
Set the main file path to app.py and use Python 3.12 if asked.
Add API credentials under the app's Secrets settings if generated answers are needed.
Deploy, then verify document upload and a sample question in the deployed app.
Current deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

Docker
docker build -t domain-ai-assistant .
docker run --rm -p 8501:8501 -e OPENAI_API_KEY="YOUR_API_KEY" domain-ai-assistant
Do not bake credentials into the Docker image. Pass them at runtime through environment variables or your deployment platform's secret manager.

Design and safety notes
This is a demo/reference implementation, not an audited production service.
Uploaded documents are processed in the running app session; this starter project does not persist a shared remote vector database.
Do not upload confidential or regulated documents to an app deployment you do not control.
Retrieval relevance is not a guarantee of factual correctness. Evaluate with domain-specific queries and inspect source citations.
A model may still make mistakes. Use explicit review and escalation rules for consequential domains.
