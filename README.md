# PromptCritic 🎯

An AI-powered prompt evaluation tool that analyzes and scores prompts using a comprehensive 35-criteria rubric. Get actionable insights to improve your prompts with professional-grade evaluation powered by OpenAI, Claude, or Gemini.

## 📋 Overview

PromptCritic evaluates prompts across 35 expert criteria including:
- Clarity & Specificity
- Context & Task Definition
- Output Format Requirements
- Role/Persona Usage
- Hallucination Minimization
- Ethical Alignment
- And 29 more dimensions...

Each prompt receives:
- ✅ Total score out of 175 points
- 📊 Individual criterion scores (1-5 scale)
- 💡 Strengths and improvement areas
- 🎯 Actionable refinement suggestions
- 📄 Exportable reports (PDF/JSON)

## 🏗️ Architecture

**Backend**: FastAPI + Python + MongoDB  
**Frontend**: React 19 + shadcn/ui + Tailwind CSS  
**LLM Integration**: OpenAI GPT-4, Claude Sonnet, Google Gemini

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**
- **Node.js 16+** and **Yarn**
- **MongoDB** - See [MongoDB Setup](#mongodb-setup) section below
- **API Key** for at least one LLM provider:
  - [OpenAI API Key](https://platform.openai.com/api-keys)
  - [Anthropic API Key](https://console.anthropic.com/)
  - [Google AI Studio API Key](https://makersuite.google.com/app/apikey)

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
cd /Users/sharajrewoo/DemoReposQA/PromptCritic
```

### Step 2: MongoDB Setup

#### Option 1: Local MongoDB (Recommended for Development)

**macOS (using Homebrew):**
```bash
# Install MongoDB
brew tap mongodb/brew
brew install mongodb-community@8.0

# Start MongoDB service
brew services start mongodb/brew/mongodb-community@8.0

# Verify MongoDB is running
brew services list | grep mongodb
```

**Windows:**
1. Download MongoDB from [MongoDB Download Center](https://www.mongodb.com/try/download/community)
2. Run the installer and follow the setup wizard
3. MongoDB will start automatically as a Windows Service

**Linux (Ubuntu/Debian):**
```bash
# Import MongoDB public key
wget -qO - https://www.mongodb.org/static/pgp/server-8.0.asc | sudo apt-key add -

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list

# Install MongoDB
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod
sudo systemctl enable mongod
```

#### Option 2: MongoDB Atlas (Cloud)

1. Create a free account at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create a new cluster (free tier available)
3. Get your connection string
4. Update `MONGO_URL` in `.env` with your Atlas connection string

#### Option 3: Docker

```bash
# Run MongoDB in a Docker container
docker run -d -p 27017:27017 --name mongodb mongo:8.0

# Stop MongoDB
docker stop mongodb

# Start MongoDB again
docker start mongodb
```

### Step 3: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR on Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Create .env file with your configuration
cat > .env << EOF
MONGO_URL=mongodb://localhost:27017
DB_NAME=promptcritic
CORS_ORIGINS=http://localhost:3000
EOF
```

**Important**: 
- For local MongoDB: Use `MONGO_URL=mongodb://localhost:27017`
- For MongoDB Atlas: Use your Atlas connection string (e.g., `mongodb+srv://username:password@cluster.mongodb.net/`)
- For Docker: Use `MONGO_URL=mongodb://localhost:27017`

### Step 4: Frontend Setup

Open a **new terminal window** and run:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
yarn install
```

## ▶️ Running the Application

### Start Backend Server

In the **backend terminal**:

```bash
cd backend
source venv/bin/activate  # Activate venv if not already active
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

✅ Backend will be available at: `http://localhost:8000`  
✅ API docs available at: `http://localhost:8000/docs`

### Start Frontend Server

In a **separate terminal**:

```bash
cd frontend
yarn start
```

✅ Frontend will open automatically at: `http://localhost:3000`

## 🎮 Using the Application

### First-Time Configuration

1. **Open the app** in your browser at `http://localhost:3000`
2. **Click the Settings button** (⚙️ icon in the top-right)
3. **Configure your LLM provider**:
   - Select provider (OpenAI, Claude, or Gemini)
   - Enter your API key
   - (Optional) Specify custom model name
4. **Click "Save Configuration"**

### Evaluating a Prompt

1. **Enter your prompt** in the text area on the Dashboard
2. **Click "Evaluate Prompt"** button
3. **View results**:
   - Total score out of 175
   - Quality rating (Excellent/Good/Fair/Needs Improvement)
   - Top 5 refinement suggestions
4. **Click "View Full Details"** to see all 35 criteria scores
5. **Export** results as PDF or JSON if needed

### Additional Features

- **📜 History**: View all past evaluations
- **🔄 Compare**: Compare multiple evaluations side-by-side
- **📥 Export**: Download evaluation reports as PDF or JSON

## 📁 Project Structure

```
PromptCritic/
├── backend/
│   ├── server.py              # FastAPI application & routes
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment configuration
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.js      # Main evaluation interface
│   │   │   ├── History.js        # Past evaluations
│   │   │   ├── Compare.js        # Comparison view
│   │   │   └── EvaluationDetail.js
│   │   └── components/ui/        # shadcn/ui components
│   ├── package.json           # Node dependencies
│   └── tailwind.config.js     # Styling configuration
└── README.md                  # This file
```

## 🔑 Environment Variables

### Backend (.env)

| Variable | Description | Example |
|----------|-------------|---------|
| `MONGO_URL` | MongoDB connection string | `mongodb://localhost:27017` |
| `DB_NAME` | Database name | `promptcritic` |
| `CORS_ORIGINS` | Allowed frontend origins | `http://localhost:3000` |

## 🛠️ Troubleshooting

### MongoDB Connection Issues

**Check if MongoDB is running:**
```bash
# macOS (Homebrew)
brew services list | grep mongodb

# Linux
sudo systemctl status mongod

# Windows
services.msc  # Look for "MongoDB Server"
```

**Start MongoDB if not running:**
```bash
# macOS (Homebrew)
brew services start mongodb/brew/mongodb-community@8.0

# Linux
sudo systemctl start mongod

# Docker
docker start mongodb
```

**Common issues:**
- Verify `MONGO_URL` in `.env` is correct
- For Atlas: Check connection string and whitelist your IP
- For local: Ensure port 27017 is not in use by another process
- Check MongoDB logs for errors:
  - macOS: `/opt/homebrew/var/log/mongodb/mongo.log`
  - Linux: `/var/log/mongodb/mongod.log`

### Backend Won't Start
- Verify Python virtual environment is activated
- Check all dependencies installed: `pip list`
- Review logs for specific error messages

### Frontend Issues
- Clear node_modules and reinstall: `rm -rf node_modules && yarn install`
- Check that backend is running and accessible
- Verify API endpoint in `frontend/src/App.js`

### API Key Issues
- Ensure API key is valid and has sufficient credits
- Check API key permissions for the selected model
- Verify correct provider selected in settings

## 📦 Tech Stack Details

### Backend
- **FastAPI** - Modern Python web framework
- **Motor** - Async MongoDB driver
- **Pydantic** - Data validation
- **ReportLab** - PDF generation
- **OpenAI/Anthropic/Google SDKs** - LLM provider integration

### Frontend
- **React 19** - UI framework
- **shadcn/ui** - Component library
- **Tailwind CSS** - Utility-first CSS
- **Axios** - HTTP client
- **React Router** - Navigation
- **Lucide React** - Icons

## 🤝 Development

### Backend Development
```bash
cd backend
source venv/bin/activate
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend
yarn start
```

## 📝 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/settings` | Get LLM configuration |
| POST | `/api/settings` | Save LLM configuration |
| POST | `/api/evaluate` | Evaluate a prompt |
| GET | `/api/evaluations` | Get all evaluations |
| GET | `/api/evaluations/{id}` | Get specific evaluation |
| DELETE | `/api/evaluations/{id}` | Delete evaluation |
| POST | `/api/compare` | Compare evaluations |
| GET | `/api/export/json/{id}` | Export as JSON |
| GET | `/api/export/pdf/{id}` | Export as PDF |

## 🎯 Default Models

| Provider | Default Model |
|----------|---------------|
| OpenAI | `gpt-4o` |
| Claude | `claude-3-7-sonnet-20250219` |
| Gemini | `gemini-2.0-flash-exp` |

You can override these by specifying a custom model name in Settings.

## 📄 License

This project is for educational and development purposes.

## 🙋‍♂️ Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review API documentation at `http://localhost:8000/docs`
3. Check console logs for detailed error messages
