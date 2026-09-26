# 📘 HR Policy Assistant — RAG with Streamlit

An HR Policy Assistant that lets a user upload an HR Policy PDF and ask questions about it.

The application uses **Retrieval-Augmented Generation (RAG)**:

1. Upload an HR Policy PDF.
2. Extract text with **PyMuPDF**.
3. Split the text into overlapping chunks.
4. Convert chunks into embeddings with **Sentence Transformers**.
5. Store embeddings in a **FAISS** vector index.
6. Retrieve the most relevant chunks for each question.
7. Send only the retrieved policy context to **Groq `openai/gpt-oss-20b`**.
8. Display the answer together with the PDF pages/chunks used as sources.

## 🧱 Tech Stack

- **Frontend / App:** Streamlit
- **PDF extraction:** PyMuPDF
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector database:** FAISS
- **LLM:** Groq — `openai/gpt-oss-20b`
- **Deployment:** Streamlit Community Cloud
- **Source control:** GitHub

## 📁 Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🔄 RAG Architecture

```text
             HR Policy PDF
                   │
                   ▼
            PyMuPDF extraction
                   │
                   ▼
            Text + page numbers
                   │
                   ▼
          Overlapping text chunks
                   │
                   ▼
       Sentence Transformer embeddings
                   │
                   ▼
              FAISS Index
                   │
       User question
             │
             ▼
      Question embedding
             │
             ▼
       FAISS similarity search
             │
             ▼
      Top relevant policy chunks
             │
             ▼
       Groq GPT-OSS 20B
             │
             ▼
       Grounded HR answer
             │
             ▼
       Answer + source pages
```

## 🔐 Groq API Key

Do **not** put your Groq API key inside `app.py`.

The app reads:

```text
GROQ_API_KEY
```

On Streamlit Community Cloud, add it through the app's **Secrets** settings.

Use this TOML format:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Never commit your real API key to GitHub.

## 🌐 Deploy Without VS Code or Terminal

You can create and deploy the complete project using only your browser.

### 1. Create a GitHub repository

Go to GitHub and create a new repository.

Suggested name:

```text
hr-policy-assistant
```

You can make it public or private. Streamlit Community Cloud supports both when the GitHub connection has the required permissions.

### 2. Add the files

Inside the repository, choose **Add file → Create new file**.

Create these four files exactly:

```text
app.py
requirements.txt
README.md
.gitignore
```

Copy the code from this project into the corresponding files and commit each file.

The repository root should look like:

```text
hr-policy-assistant/
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### 3. Create your Groq API key

Create a Groq API key from the Groq developer console.

Keep the key private. Do not put it in GitHub.

### 4. Open Streamlit Community Cloud

Open:

https://share.streamlit.io/

Sign in with GitHub and connect your GitHub account.

### 5. Deploy the repository

Choose:

**Create app → Yup, I have an app**

Then select:

- Repository: `YOUR_USERNAME/hr-policy-assistant`
- Branch: `main`
- Main file path: `app.py`

Then open **Advanced settings**.

### 6. Add the Groq secret

In the **Secrets** field, paste:

```toml
GROQ_API_KEY = "your-groq-api-key"
```

Save the settings and deploy.

Do not put the API key in the repository.

### 7. Wait for deployment

Streamlit Cloud will install the packages from `requirements.txt` and start `app.py`.

The first startup can take longer because the Sentence Transformer model needs to be downloaded.

### 8. Test the app

After deployment:

1. Upload an HR Policy PDF.
2. Wait for the indexing message.
3. Ask a question about the policy.
4. Check the answer.
5. Open **Sources used** to see the retrieved pages/chunks.

## 💬 Example Questions

Try questions such as:

```text
How many annual leave days are employees entitled to?
```

```text
What is the policy for working from home?
```

```text
How does an employee request sick leave?
```

```text
What are the working hours?
```

```text
What is the resignation notice period?
```

The assistant is instructed to say that it could not find the information when the uploaded policy does not contain enough evidence.

## ⚙️ How Retrieval Works

The app uses:

```text
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
```

The chunks are embedded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

FAISS uses cosine-style similarity through normalized embeddings and an inner-product index:

```text
IndexFlatIP
```

The top 5 relevant chunks are sent to the LLM as context.

## 🛡️ Important RAG Behavior

The prompt tells the model:

- use only the uploaded policy context;
- do not invent HR rules;
- mention relevant pages when possible;
- include conditions and exceptions;
- clearly say when the information is not present.

This makes the application more suitable for policy-document Q&A than a normal chatbot.

## ⚠️ Current Limitations

### Scanned PDFs

This version extracts selectable text from PDFs. A scanned PDF containing only images may produce little or no text.

For scanned documents, OCR should be added later.

### Temporary document storage

The PDF is processed in memory. The application does not permanently save the uploaded HR policy to a database.

The FAISS index is also created in memory for the current Streamlit session.

### Multiple users

For a production HR system, consider adding:

- authentication;
- per-user document isolation;
- persistent vector storage;
- document deletion;
- audit logging;
- access control;
- encryption;
- OCR;
- better chunking;
- hybrid keyword + vector search.

## 🔄 Updating the App

After deployment, edit files directly on GitHub.

Commit your changes.

Streamlit Community Cloud automatically detects repository changes and redeploys the app.

## 🧪 Suggested Future Improvements

1. Multi-PDF knowledge base
2. Department-specific HR policies
3. Policy version management
4. Conversation memory
5. Better source highlighting
6. OCR for scanned PDFs
7. Hybrid BM25 + FAISS retrieval
8. Reranking
9. User authentication
10. Admin dashboard
11. Persistent vector database
12. Policy comparison between versions

## 📜 License

This project is provided as a learning/demo project. Add your preferred license before publishing it as an open-source project.
