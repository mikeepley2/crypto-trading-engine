#!/bin/bash
# Quick Repository Setup Script
# Run this after creating the GitHub repository

echo "🚀 Setting up crypto-trading-engine repository..."

cd e:/git/crypto-trading-engine

echo "📊 Current repository status:"
git status

echo "🔗 Remote configuration:"
git remote -v

echo "📤 Pushing to GitHub..."
git push -u origin master

if [ $? -eq 0 ]; then
    echo "✅ SUCCESS: Trading engine repository pushed to GitHub!"
    echo "🌐 Repository URL: https://github.com/mikeepley2/crypto-trading-engine"
else
    echo "❌ FAILED: Please create the repository on GitHub first"
    echo "📋 Steps:"
    echo "1. Go to https://github.com/new"
    echo "2. Repository name: crypto-trading-engine"
    echo "3. Make it Private"
    echo "4. DO NOT initialize with README"
    echo "5. Click 'Create repository'"
    echo "6. Run this script again"
fi