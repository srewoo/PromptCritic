# 🔧 Troubleshooting Guide

## Issue: Cannot Save LLM Configuration

### Quick Diagnosis

Run the MongoDB test script:
```bash
cd backend
python test_mongodb.py
```

This will check:
- ✅ MongoDB connection
- ✅ Environment variables
- ✅ Database read/write operations

---

## Common Issues & Solutions

### 1. MongoDB Not Running

**Symptoms:**
- Settings don't save
- Backend shows connection errors
- Test script fails to connect

**Solution:**

**For macOS (Homebrew):**
```bash
# Check if MongoDB is running
brew services list | grep mongodb

# Start MongoDB
brew services start mongodb-community

# Or start manually
mongod --config /usr/local/etc/mongod.conf
```

**For Linux:**
```bash
# Check status
sudo systemctl status mongod

# Start MongoDB
sudo systemctl start mongod

# Enable on boot
sudo systemctl enable mongod
```

**For Windows:**
```bash
# Start MongoDB service
net start MongoDB
```

**For Docker:**
```bash
# Run MongoDB in Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Check if running
docker ps | grep mongodb
```

---

### 2. Missing .env File

**Symptoms:**
- Backend crashes on startup
- "MONGO_URL not set" error

**Solution:**

Create `.env` file in the `backend/` directory:

```bash
cd backend
cp env.example .env
```

Edit `.env` with your values:
```bash
MONGO_URL=mongodb://localhost:27017
DB_NAME=promptcritic
CORS_ORIGINS=http://localhost:3000
```

---

### 3. Wrong MongoDB Connection String

**Symptoms:**
- Connection timeout errors
- "Failed to connect to server" errors

**Common Connection Strings:**

**Local MongoDB:**
```
MONGO_URL=mongodb://localhost:27017
```

**MongoDB with Authentication:**
```
MONGO_URL=mongodb://username:password@localhost:27017/promptcritic?authSource=admin
```

**MongoDB Atlas (Cloud):**
```
MONGO_URL=mongodb+srv://username:password@cluster.mongodb.net/promptcritic?retryWrites=true&w=majority
```

**Docker MongoDB:**
```
MONGO_URL=mongodb://host.docker.internal:27017
```

---

### 4. Port Already in Use

**Symptoms:**
- Backend fails to start
- "Address already in use" error

**Solution:**

**Find process using port 8000:**
```bash
# macOS/Linux
lsof -i :8000

# Kill the process
kill -9 <PID>
```

**Or use a different port:**
```bash
# In backend directory
uvicorn server:app --reload --port 8001
```

Then update frontend API URL in `frontend/src/App.js`:
```javascript
export const API = "http://localhost:8001/api";
```

---

### 5. CORS Errors

**Symptoms:**
- Frontend can't connect to backend
- "CORS policy" errors in browser console

**Solution:**

Add your frontend URL to `.env`:
```bash
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

Or temporarily allow all origins (development only):
```bash
CORS_ORIGINS=*
```

---

### 6. Database Permissions

**Symptoms:**
- "not authorized" errors
- "authentication failed" errors

**Solution:**

**Create MongoDB user:**
```bash
mongosh

use admin
db.createUser({
  user: "promptcritic_user",
  pwd: "your_password",
  roles: [{ role: "readWrite", db: "promptcritic" }]
})
```

Update `.env`:
```bash
MONGO_URL=mongodb://promptcritic_user:your_password@localhost:27017/promptcritic?authSource=admin
```

---

### 7. Frontend Can't Reach Backend

**Symptoms:**
- Network errors in browser console
- "Failed to fetch" errors

**Checklist:**

1. **Backend is running:**
   ```bash
   cd backend
   uvicorn server:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Check backend URL in browser:**
   ```
   http://localhost:8000/api/settings
   ```
   Should return `null` or settings JSON

3. **Check API constant in frontend:**
   ```javascript
   // frontend/src/App.js or Dashboard.js
   const API = 'http://localhost:8000/api';
   ```

4. **Clear browser cache:**
   - Chrome: Cmd+Shift+Delete (Mac) or Ctrl+Shift+Delete (Windows)
   - Or use Incognito/Private mode

---

### 8. Settings Save But Don't Persist

**Symptoms:**
- Settings save successfully
- But disappear after page refresh

**Solution:**

Check if `useEffect` is loading settings:

```javascript
// In Dashboard.js
useEffect(() => {
  const loadSettings = async () => {
    try {
      const response = await axios.get(`${API}/settings`);
      if (response.data) {
        setSettings(response.data);
        setLlmProvider(response.data.llm_provider);
        setApiKey(response.data.api_key);
        setModelName(response.data.model_name || '');
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };
  loadSettings();
}, []);
```

---

## Verification Steps

### 1. Check MongoDB is Running
```bash
mongosh
show dbs
use promptcritic
show collections
db.settings.find()
```

### 2. Check Backend Logs
```bash
cd backend
uvicorn server:app --reload --log-level debug
```

Look for:
- ✅ "Connected to MongoDB"
- ✅ "Application startup complete"
- ❌ Connection errors
- ❌ Authentication errors

### 3. Check Browser Console
Open DevTools (F12) → Console tab

Look for:
- ❌ Network errors (red)
- ❌ CORS errors
- ❌ 400/500 status codes

### 4. Test API Directly

**Save settings:**
```bash
curl -X POST http://localhost:8000/api/settings \
  -H "Content-Type: application/json" \
  -d '{
    "llm_provider": "openai",
    "api_key": "test_key",
    "model_name": "gpt-4o"
  }'
```

**Get settings:**
```bash
curl http://localhost:8000/api/settings
```

---

## Still Having Issues?

### Enable Debug Mode

**Backend:**
```python
# In server.py, add at the top
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Frontend:**
```javascript
// In Dashboard.js saveSettings function
console.log('Saving settings:', { llmProvider, apiKey, modelName });
console.log('Response:', response.data);
```

### Check Dependencies

**Backend:**
```bash
cd backend
pip list | grep -E "motor|pymongo|fastapi"
```

Should show:
- motor >= 3.0.0
- pymongo >= 4.0.0
- fastapi >= 0.100.0

**Frontend:**
```bash
cd frontend
npm list axios
```

Should show axios installed

---

## Clean Restart

If all else fails, try a clean restart:

```bash
# Stop everything
# Kill backend (Ctrl+C)
# Kill frontend (Ctrl+C)
# Stop MongoDB
brew services stop mongodb-community

# Clean start
brew services start mongodb-community
cd backend && uvicorn server:app --reload
cd frontend && npm start
```

---

## Get Help

If you're still stuck, gather this info:

1. **MongoDB status:**
   ```bash
   brew services list | grep mongodb
   mongosh --eval "db.version()"
   ```

2. **Backend logs:**
   ```bash
   # Last 20 lines of backend output
   ```

3. **Browser console errors:**
   ```
   # Screenshot or copy errors from DevTools
   ```

4. **Environment:**
   - OS: macOS/Linux/Windows
   - MongoDB version
   - Python version
   - Node version
