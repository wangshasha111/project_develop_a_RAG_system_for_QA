#!/usr/bin/env python3
"""
部署前检查脚本
检查所有必需的文件和配置是否就绪
"""

import os
import sys
from pathlib import Path
import json

def check_file_exists(filepath, required=True):
    """检查文件是否存在"""
    exists = Path(filepath).exists()
    status = "✅" if exists else ("❌" if required else "⚠️")
    print(f"{status} {filepath}")
    return exists

def check_gitignore():
    """检查 .gitignore 是否正确配置"""
    print("\n📝 检查 .gitignore 配置...")
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        print("❌ .gitignore 文件不存在")
        return False
    
    with open(gitignore_path) as f:
        content = f.read()
    
    critical_items = [".env", "venv/", "__pycache__/", "*.pyc"]
    missing = []
    
    for item in critical_items:
        if item not in content:
            missing.append(item)
    
    if missing:
        print(f"⚠️  .gitignore 缺少以下项: {', '.join(missing)}")
        return False
    
    print("✅ .gitignore 配置正确")
    return True

def check_env_not_in_git():
    """确保 .env 文件不会被提交到 Git"""
    print("\n🔒 检查敏感文件...")
    
    if Path(".env").exists():
        # 检查 .env 是否在 git 中
        import subprocess
        try:
            result = subprocess.run(
                ["git", "check-ignore", ".env"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print("✅ .env 文件已被 Git 忽略")
                return True
            else:
                print("❌ 警告: .env 文件可能会被提交到 Git!")
                print("   请确保 .env 在 .gitignore 中")
                return False
        except:
            print("⚠️  无法验证 Git 状态")
            return True
    else:
        print("✅ 没有 .env 文件（将在 Streamlit Secrets 中配置）")
        return True

def check_requirements():
    """检查 requirements.txt"""
    print("\n📦 检查依赖配置...")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt 不存在")
        return False
    
    with open("requirements.txt") as f:
        content = f.read()
    
    required_packages = [
        "streamlit",
        "langchain",
        "chromadb",
        "openai",
        "unstructured"
    ]
    
    missing = [pkg for pkg in required_packages if pkg not in content.lower()]
    
    if missing:
        print(f"⚠️  requirements.txt 可能缺少: {', '.join(missing)}")
    else:
        print("✅ requirements.txt 包含所有关键依赖")
    
    return True

def check_project_structure():
    """检查项目结构"""
    print("\n📁 检查项目结构...")
    
    required_files = {
        "app.py": True,
        "config.py": True,
        "document_processor.py": True,
        "retriever.py": True,
        "rag_chain.py": True,
        "utils.py": True,
        "requirements.txt": True,
        "packages.txt": True,
        ".streamlit/config.toml": True,
        ".gitignore": True,
    }
    
    optional_files = {
        "preprocess.py": False,
        "README.md": False,
        "DEPLOYMENT_GUIDE.md": False,
    }
    
    all_ok = True
    
    print("\n必需文件:")
    for filepath, required in required_files.items():
        if not check_file_exists(filepath, required) and required:
            all_ok = False
    
    print("\n可选文件:")
    for filepath, required in optional_files.items():
        check_file_exists(filepath, required)
    
    return all_ok

def check_dataset():
    """检查数据集"""
    print("\n📚 检查数据集...")
    
    dataset_dir = Path("RAG Project Dataset")
    if not dataset_dir.exists():
        print("⚠️  RAG Project Dataset 目录不存在")
        print("   你可以在部署后上传数据，或使用其他数据源")
        return True
    
    pdf_files = list(dataset_dir.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  没有找到 PDF 文件")
        return True
    
    total_size = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)  # MB
    
    print(f"✅ 找到 {len(pdf_files)} 个 PDF 文件")
    print(f"   总大小: {total_size:.2f} MB")
    
    if total_size > 100:
        print("⚠️  警告: 数据集较大，可能影响部署速度")
        print("   考虑减少 PDF 数量或使用外部存储")
    
    return True

def check_git_status():
    """检查 Git 状态"""
    print("\n🔄 检查 Git 状态...")
    
    import subprocess
    
    try:
        # 检查是否是 git 仓库
        result = subprocess.run(
            ["git", "status"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("⚠️  不是 Git 仓库")
            print("   运行: git init")
            return False
        
        # 检查是否有未提交的更改
        if "nothing to commit" in result.stdout:
            print("✅ 所有更改已提交")
        else:
            print("⚠️  有未提交的更改")
            print("   运行: git add . && git commit -m 'message'")
        
        # 检查是否有远程仓库
        result = subprocess.run(
            ["git", "remote", "-v"],
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            print("✅ 已配置远程仓库")
            print(result.stdout)
        else:
            print("⚠️  未配置远程仓库")
            print("   运行: git remote add origin <your-repo-url>")
            return False
        
        return True
        
    except FileNotFoundError:
        print("❌ Git 未安装")
        return False
    except Exception as e:
        print(f"⚠️  检查 Git 状态时出错: {e}")
        return True

def generate_secrets_template():
    """生成 Streamlit Secrets 模板"""
    print("\n🔐 生成 Streamlit Secrets 模板...")
    
    template = """# Streamlit Secrets 配置
# 复制以下内容到 Streamlit Cloud 的 Secrets 管理中

# API Keys
OPENAI_API_KEY = "sk-your-openai-api-key-here"
GOOGLE_API_KEY = "AI-your-google-api-key-here"

# Application Settings
DEFAULT_PROVIDER = "OpenAI"
DEFAULT_MODEL = "gpt-4o-mini"
DEBUG_MODE = "false"

# Redis Configuration (可选)
REDIS_HOST = "localhost"
REDIS_PORT = "6379"
"""
    
    secrets_file = Path(".streamlit/secrets.toml.template")
    secrets_file.parent.mkdir(exist_ok=True)
    
    with open(secrets_file, "w") as f:
        f.write(template)
    
    print(f"✅ Secrets 模板已生成: {secrets_file}")
    print("   请将此内容复制到 Streamlit Cloud 的 Secrets 配置中")
    
    return True

def main():
    """主函数"""
    print("=" * 80)
    print("🚀 Streamlit Cloud 部署前检查")
    print("=" * 80)
    
    checks = [
        ("项目结构", check_project_structure),
        ("Git 配置", check_gitignore),
        ("敏感文件", check_env_not_in_git),
        ("依赖配置", check_requirements),
        ("数据集", check_dataset),
        ("Git 状态", check_git_status),
        ("Secrets 模板", generate_secrets_template),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} 检查失败: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 检查总结")
    print("=" * 80)
    
    for name, result in results:
        status = "✅" if result else "⚠️"
        print(f"{status} {name}")
    
    all_critical_passed = all(result for name, result in results 
                              if name in ["项目结构", "Git 配置", "依赖配置"])
    
    if all_critical_passed:
        print("\n✅ 关键检查全部通过！")
        print("\n下一步:")
        print("1. 推送代码到 GitHub: git push")
        print("2. 访问 https://share.streamlit.io/")
        print("3. 连接你的 GitHub 仓库")
        print("4. 配置 Secrets（使用 .streamlit/secrets.toml.template）")
        print("5. 点击 Deploy!")
        print("\n详细步骤请参考: DEPLOYMENT_GUIDE.md")
    else:
        print("\n⚠️  请修复上述问题后再部署")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
