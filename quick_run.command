#!/bin/bash

# 快速启动脚本 - 使用系统Python（不创建虚拟环境）
# Quick run script - Uses system Python (no virtual environment)

cd "$(dirname "$0")"

echo "=================================================="
echo "  快速启动 RAG 系统 / Quick Launch RAG System"
echo "=================================================="
echo ""
echo "⚠️  警告：使用系统Python，不创建虚拟环境"
echo "⚠️  Warning: Using system Python, no virtual environment"
echo ""

# 检查是否已安装streamlit
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit未安装。请先运行："
    echo "   pip3 install -r requirements.txt"
    echo ""
    echo "或者使用 run.command 创建虚拟环境"
    read -p "按回车键退出..."
    exit 1
fi

echo "🚀 启动应用..."
echo ""

# 直接运行，不使用虚拟环境
python3 -m streamlit run app.py

