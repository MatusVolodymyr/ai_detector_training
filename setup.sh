#!/bin/bash

# AI Detector Training Pipeline Setup Script

echo "=================================================="
echo "AI Detector Training Pipeline Setup"
echo "=================================================="
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Install Python requirements
echo ""
echo "📦 Installing Python requirements..."
pip install -r requirements.txt

# Download spaCy models
echo ""
echo "📥 Downloading spaCy models..."
echo "   This may take a few minutes..."

python3 -m spacy download en_core_web_sm
python3 -m spacy download xx_sent_ud_sm

echo ""
echo "❓ Do you need multilingual support? (y/n)"
read -r multilingual

if [ "$multilingual" = "y" ]; then
    echo "   Downloading additional language models..."
    python3 -m spacy download de_core_news_sm
    python3 -m spacy download fr_core_news_sm
    python3 -m spacy download es_core_news_sm
    python3 -m spacy download pt_core_news_sm
fi

# Download NLTK data
echo ""
echo "📥 Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Create directories
echo ""
echo "📁 Creating output directories..."
mkdir -p trained_models
mkdir -p checkpoints
mkdir -p logs
mkdir -p feature_extraction

# Test GPU availability
echo ""
echo "🖥️  Testing GPU availability..."
python3 -c "import torch; print(f'   GPU available: {torch.cuda.is_available()}'); print(f'   Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Prepare your dataset (or create a sample with: python create_sample_dataset.py)"
echo "2. Edit config.yaml to set your dataset path"
echo "3. Run training: python main.py --config config.yaml"
echo ""
echo "For more information, see README.md"
echo ""
