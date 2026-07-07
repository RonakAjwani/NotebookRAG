# NotebookRAG - Multimodal Document Intelligence

A powerful multimodal Graph RAG system for document analysis and intelligent querying, inspired by Google's NotebookLM.

## Features

- **Multi-document Upload**: Upload multiple PDFs, DOCX, TXT, and other document formats
- **Graph-based RAG**: Utilizes Graph RAG for enhanced document understanding and relationship mapping
- **Vector Search**: FAISS-powered vector storage for semantic search
- **Agentic Workflow**: LangGraph-based intelligent agents for complex query handling
- **Modern UI**: Built with React, TypeScript, and shadcn/ui components

## Tech Stack

### Frontend
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **UI Components**: shadcn/ui (Radix UI primitives)
- **Styling**: Tailwind CSS
- **Routing**: React Router v6
- **State Management**: TanStack Query
- **Theme**: next-themes for dark/light mode support

### Backend
- **Framework**: FastAPI & Uvicorn
- **Graph Storage**: NetworkX
- **Vector Storage**: FAISS
- **LLM API**: Cerebras
- **Workflow**: LangGraph for agentic workflows

## Getting Started

### Prerequisites

- Node.js v22.20.0
- npm 11.6.2
- Python 3.14.0

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:8080`

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

## Project Structure

```
NLP-Project/
├── frontend/           # React frontend application
│   ├── src/
│   │   ├── components/ # Reusable UI components
│   │   ├── pages/      # Page components
│   │   ├── lib/        # Utility functions
│   │   └── hooks/      # Custom React hooks
│   └── public/         # Static assets
├── backend/            # FastAPI backend application
│   ├── venv/          # Virtual environment (not tracked)
│   ├── api/           # API routes
│   ├── services/      # Business logic
│   ├── models/        # Data models
│   ├── storage/       # Graph & Vector storage
│   └── agents/        # LangGraph agents
└── README.md
```

## Development

### Available Scripts (Frontend)

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI).

## Environment Variables

Create a `.env` file in the backend directory:

```env
CEREBRAS_API_KEY=your_api_key_here
ENVIRONMENT=development
```

## Contributing

This is an MVP project. Contributions and suggestions are welcome!

## License

MIT License
