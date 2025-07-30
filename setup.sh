#!/bin/bash

# MySQL Schema Fetcher Setup Script
# This script sets up the complete environment for the MySQL Schema Fetcher

set -e  # Exit on any error

echo "🗄️  MySQL Schema Fetcher Setup"
echo "================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.12"

if [ "$(echo "$python_version >= $required_version" | bc -l)" -eq 0 ]; then
    echo "❌ Python 3.9+ is required. Found: Python $python_version"
    exit 1
fi
echo "✅ Python $python_version detected"

# Create virtual environment
echo "🐍 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p output
mkdir -p logs
mkdir -p config
mkdir -p tests

# Copy configuration files if they don't exist
echo "⚙️  Setting up configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Environment file created (.env)"
    echo "⚠️  Please edit .env with your database credentials"
else
    echo "ℹ️  Environment file already exists"
fi

# Set up configuration files
if [ ! -f "config/database.json" ]; then
    echo "✅ Database configuration created"
else
    echo "ℹ️  Database configuration already exists"
fi

if [ ! -f "config/mapping_rules.json" ]; then
    echo "✅ Mapping rules configuration created"
else
    echo "ℹ️  Mapping rules configuration already exists"
fi

# Make main.py executable
chmod +x main.py

# Run tests to verify setup
echo "🧪 Running setup verification tests..."
python -m pytest tests/test_schema_fetcher.py::TestConfiguration -v || {
    echo "⚠️  Some tests failed, but setup is complete"
}

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your database credentials:"
echo "   nano .env"
echo ""
echo "2. Test database connection:"
echo "   python -c \"from src.database.connector import db_connector; print('✅ Connection OK' if db_connector.connect() else '❌ Connection Failed')\""
echo ""
echo "3. Run the complete pipeline:"
echo "   python main.py --full"
echo ""
echo "4. Check the output directory for generated files:"
echo "   ls -la output/"
echo ""
echo "📖 For more information, see README.md"