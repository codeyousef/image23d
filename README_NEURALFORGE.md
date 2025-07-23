# NeuralForge Studio

A comprehensive AI creative suite with desktop and web applications for generating high-quality images, 3D models, and videos using state-of-the-art AI models.

![NeuralForge Studio](./docs/banner.png)

## 🚀 Features

### Core Capabilities
- **Image Generation**: Create stunning images using FLUX models (schnell, dev, pro)
- **3D Model Generation**: Generate 3D models with HunyuanVideo 3D (text-to-3D and image-to-3D)
- **Prompt Enhancement**: Enhance prompts using local LLM (Mistral 7B via Ollama)
- **Multi-Platform**: Desktop app (NiceGUI) and Web app (React + FastAPI)
- **Serverless GPU**: RunPod integration for cloud GPU execution
- **Real-time Updates**: WebSocket support for live progress tracking

### Advanced Features
- **LoRA Support**: Multi-LoRA stacking and merging
- **Face Swap**: FaceFusion integration
- **Batch Processing**: Queue system with priority handling
- **Export Options**: Multiple 3D formats (GLB, OBJ, FBX, USDZ, etc.)
- **Credit System**: Usage-based billing with flexible pricing
- **Arabic Support**: Full RTL layout and Arabic localization

## 🏗️ Architecture

```
neuralforge-studio/
├── core/                # Shared business logic
├── desktop/             # NiceGUI desktop application
├── backend/             # FastAPI web backend
├── frontend/            # React web frontend
├── deployment/          # Docker and deployment configs
└── models/              # Model storage directory
```

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- Node.js 18+ (for web frontend)
- Docker (optional, for containerized deployment)

### Desktop Application

```bash
# Clone the repository
git clone https://github.com/yourusername/neuralforge-studio.git
cd neuralforge-studio

# Install dependencies
pip install -r requirements.txt

# Install Ollama for prompt enhancement
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral:latest

# Run the desktop app
python desktop/main.py
```

### Web Application (Development)

```bash
# Start all services with Docker Compose
docker-compose up -d

# Or run services separately:

# Backend API
cd backend
pip install -r requirements.txt
python main.py

# Frontend
cd frontend
npm install
npm run dev
```

## 📖 Usage

### Desktop App
1. Launch the application: `python desktop/main.py`
2. Navigate to the **Create** tab
3. Enter your prompt and adjust settings
4. Click **Generate** to create content
5. View results in the **Library** tab

### Web App
1. Access the app at `http://localhost:3000`
2. Register/login to your account
3. Purchase credits or use free tier
4. Generate content through the web interface
5. Download or share your creations

### API Usage

```python
import httpx

# Authenticate
response = httpx.post("http://localhost:8000/api/auth/token", data={
    "username": "your_username",
    "password": "your_password"
})
token = response.json()["access_token"]

# Generate image
headers = {"Authorization": f"Bearer {token}"}
response = httpx.post(
    "http://localhost:8000/api/generate/image",
    headers=headers,
    json={
        "prompt": "a beautiful sunset over mountains",
        "model": "flux-schnell",
        "width": 1024,
        "height": 1024
    }
)
job_id = response.json()["job_id"]
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# API Keys
RUNPOD_API_KEY=your_runpod_key
SECRET_KEY=your_secret_key

# Redis (optional)
REDIS_URL=redis://localhost:6379

# Supabase (optional)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Model Configuration

Models are configured in `core/config.py`. Available models:

**Image Models:**
- FLUX.1-schnell (fast, 4 steps)
- FLUX.1-dev (quality, 20-50 steps)
- FLUX.1-pro (premium, 50+ steps)

**3D Models:**
- HunyuanVideo 3D v2.1 (latest)
- HunyuanVideo 3D v2.0
- HunyuanVideo 3D Mini (low VRAM)

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build individual images
docker build -f deployment/Dockerfile.backend -t neuralforge-backend .
docker build -f deployment/Dockerfile.frontend -t neuralforge-frontend .
```

### Google Cloud Run

```bash
# Deploy backend to Cloud Run
./deployment/deploy.sh
```

### RunPod Serverless

1. Get RunPod API key from https://runpod.io
2. Deploy handler: `python deployment/deploy_runpod.py`
3. Configure endpoint in `.env`

## 💰 Pricing

### Credit System
- 1 Credit = $0.01 USD
- Image Generation: 5-20 credits
- 3D Generation: 50-200 credits
- Prompt Enhancement: 1 credit

### Subscription Tiers
- **Free**: 50 credits/month
- **Starter**: $25/month (500 credits)
- **Professional**: $69/month (2,000 credits)
- **Studio**: $179/month (6,000 credits)

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run integration tests
python tests/test_integration.py

# Test API endpoints
python test_backend_api.py
```

## 📚 API Documentation

- Interactive docs: http://localhost:8000/docs
- OpenAPI schema: http://localhost:8000/openapi.json

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Tencent for HunyuanVideo 3D models
- Black Forest Labs for FLUX models
- Mistral AI for prompt enhancement
- RunPod for serverless GPU infrastructure

## 📞 Support

- Documentation: [docs.neuralforge.studio](https://docs.neuralforge.studio)
- Discord: [discord.gg/neuralforge](https://discord.gg/neuralforge)
- Email: support@neuralforge.studio

---

Built with ❤️ by the NeuralForge Team