#!/bin/bash

# Claude Code AI协作开发项目快速搭建脚本
# 使用方法: ./setup_project.sh <project_name> <project_type> [target_directory]

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    cat << EOF
Claude Code AI协作开发项目快速搭建脚本

使用方法:
    ./setup_project.sh <project_name> <project_type> [target_directory]

参数说明:
    project_name      项目名称 (必需)
    project_type      项目类型 (必需)
    target_directory  目标目录 (可选，默认为当前目录)

支持的项目类型:
    pytorch          PyTorch深度学习项目
    web              Web开发项目  
    data-science     数据科学项目
    research         研究型项目
    general          通用软件项目

示例:
    ./setup_project.sh my-classifier pytorch
    ./setup_project.sh web-app web ~/projects/
    ./setup_project.sh data-analysis data-science

EOF
}

# 检查参数
check_arguments() {
    if [[ $# -lt 2 ]]; then
        log_error "参数不足"
        show_help
        exit 1
    fi

    PROJECT_NAME="$1"
    PROJECT_TYPE="$2"
    TARGET_DIR="${3:-$(pwd)}"

    # 验证项目名称
    if [[ ! "$PROJECT_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        log_error "项目名称只能包含字母、数字、下划线和连字符"
        exit 1
    fi

    # 验证项目类型
    case "$PROJECT_TYPE" in
        pytorch|web|data-science|research|general)
            ;;
        *)
            log_error "不支持的项目类型: $PROJECT_TYPE"
            log_info "支持的类型: pytorch, web, data-science, research, general"
            exit 1
            ;;
    esac

    log_info "项目名称: $PROJECT_NAME"
    log_info "项目类型: $PROJECT_TYPE" 
    log_info "目标目录: $TARGET_DIR"
}

# 获取脚本所在目录
get_script_dir() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SOURCE_REPO="$SCRIPT_DIR"
    log_info "源仓库路径: $SOURCE_REPO"
}

# 创建项目目录
create_project_directory() {
    PROJECT_PATH="$TARGET_DIR/$PROJECT_NAME"
    
    if [[ -d "$PROJECT_PATH" ]]; then
        log_error "目录已存在: $PROJECT_PATH"
        exit 1
    fi
    
    mkdir -p "$PROJECT_PATH"
    cd "$PROJECT_PATH"
    log_success "创建项目目录: $PROJECT_PATH"
}

# 复制核心文档和配置
copy_core_files() {
    log_info "复制核心文档和配置..."
    
    # 创建目录结构
    mkdir -p {docs/{workflows,standards,templates,knowledge/{best_practices,error_cases}}}
    mkdir -p .claude/agents
    
    # 复制CLAUDE.md配置文件
    if [[ -f "$SOURCE_REPO/CLAUDE.md" ]]; then
        cp "$SOURCE_REPO/CLAUDE.md" .
        log_success "复制 CLAUDE.md"
    fi
    
    # 复制工作流文档
    if [[ -f "$SOURCE_REPO/pytorch_project/workflow/extract_rewrite/workflow.md" ]]; then
        cp "$SOURCE_REPO/pytorch_project/workflow/extract_rewrite/workflow.md" \
           docs/workflows/
        log_success "复制工作流文档"
    fi
    
    # 复制标准规范
    if [[ -d "$SOURCE_REPO/pytorch_project/standards" ]]; then
        cp -r "$SOURCE_REPO/pytorch_project/standards"/* docs/standards/
        log_success "复制标准规范"
    fi
    
    # 复制文档模板
    if [[ -d "$SOURCE_REPO/pytorch_project/documentation" ]]; then
        cp -r "$SOURCE_REPO/pytorch_project/documentation"/* docs/templates/
        log_success "复制文档模板"
    fi
    
    # 复制知识库模板
    if [[ -d "$SOURCE_REPO/pytorch_project/knowledge" ]]; then
        cp -r "$SOURCE_REPO/pytorch_project/knowledge"/* docs/knowledge/
        log_success "复制知识库"
    fi
    
    # 复制Agent配置
    if [[ -d "$SOURCE_REPO/pytorch_project/agents" ]]; then
        cp -r "$SOURCE_REPO/pytorch_project/agents"/* .claude/agents/
        log_success "复制Agent配置"
    fi
}

# 根据项目类型创建对应结构
setup_project_structure() {
    log_info "为项目类型 '$PROJECT_TYPE' 创建目录结构..."
    
    case "$PROJECT_TYPE" in
        pytorch)
            mkdir -p {src,tests/{unit,integration,e2e},configs,data,models,notebooks,scripts}
            echo "torch>=2.0.0" > requirements.txt
            echo "pytorch-lightning>=2.0.0" >> requirements.txt
            echo "pytest>=7.0.0" >> requirements.txt
            echo "pytest-cov>=4.0.0" >> requirements.txt
            ;;
        web)
            mkdir -p {src,tests,public,docs}
            cat > package.json << EOF
{
  "name": "$PROJECT_NAME",
  "version": "0.1.0",
  "description": "Web应用项目",
  "main": "index.js",
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "test": "jest"
  }
}
EOF
            ;;
        data-science)
            mkdir -p {data/{raw,processed,external},notebooks,src,reports,models}
            echo "pandas>=1.5.0" > requirements.txt
            echo "numpy>=1.24.0" >> requirements.txt
            echo "scikit-learn>=1.2.0" >> requirements.txt
            echo "jupyter>=1.0.0" >> requirements.txt
            ;;
        research)
            mkdir -p {papers,experiments,data,analysis,presentations}
            echo "jupyter>=1.0.0" > requirements.txt
            echo "matplotlib>=3.6.0" >> requirements.txt
            echo "numpy>=1.24.0" >> requirements.txt
            ;;
        general)
            mkdir -p {src,tests,docs}
            echo "pytest>=7.0.0" > requirements.txt
            ;;
    esac
    
    log_success "项目结构创建完成"
}

# 创建项目README
create_project_readme() {
    log_info "创建项目README..."
    
    cat > README.md << EOF
# $PROJECT_NAME

$PROJECT_TYPE 项目，使用Claude Code AI协作开发工作流

## 项目概述

<!-- 项目描述 -->

## AI协作开发工作流

本项目采用多Agent协作开发模式，包含以下Agent：

- **agent-product-manager**: 需求分析、PRD编写、功能验收
- **agent-tech-lead**: 技术方案设计、架构决策、项目协调
- **agent-researcher**: 论文调研、技术可行性分析、理论验证
- **agent-algorithm-engineer**: 算法实现、模型设计、核心开发
- **agent-code-reviewer**: 代码质量审核、标准检查
- **agent-qa-engineer**: 测试用例编写、质量保证
- **agent-docs-writer**: 技术文档、项目文档编写

## 快速开始

### 环境设置

\`\`\`bash
# 安装依赖
pip install -r requirements.txt

# 激活Claude Code
claude-code --version
\`\`\`

### 开发流程

1. **需求分析阶段**
   \`\`\`bash
   # 创建PRD文档
   cp docs/templates/PRD/prd_template.md docs/PRD.md
   \`\`\`

2. **技术设计阶段**  
   \`\`\`bash
   # 创建TECH_SPEC文档
   cp docs/templates/TECH_SPEC/TECH_SPEC_template.md docs/TECH_SPEC.md
   \`\`\`

3. **原型开发阶段**
   \`\`\`bash
   # 创建原型文档
   cp docs/templates/PROTOTYPE/PROTOTYPE_template.md docs/PROTOTYPE.md
   \`\`\`

## 文档结构

- \`docs/\`: 项目文档
  - \`docs/templates/\`: 文档模板
  - \`docs/standards/\`: 代码规范和测试标准
  - \`docs/knowledge/\`: 最佳实践和错误案例库
  - \`docs/workflows/\`: AI协作工作流文档
- \`.claude/agents/\`: Agent配置文件

## 代码规范

- Python代码: 遵循 \`docs/standards/pycode_standards.md\`
- 测试代码: 遵循 \`docs/standards/pytest_stands.md\`
- Git提交: 遵循 \`docs/standards/git_commit_std.md\`

## Agent协作

- Agent配置: \`.claude/agents/\` 目录下的markdown文件
- 工作流程: \`docs/workflows/workflow.md\`

## 许可证

<!-- 添加许可证信息 -->
EOF

    log_success "项目README创建完成"
}

# 显示完成信息
show_completion_info() {
    log_success "项目设置完成!"
    echo
    log_info "项目路径: $PROJECT_PATH"
    log_info "项目类型: $PROJECT_TYPE"
    echo
    echo -e "${GREEN}下一步操作:${NC}"
    echo "  1. cd $PROJECT_PATH"
    echo "  2. 阅读 README.md 了解项目结构"
    echo "  3. 阅读 docs/workflows/workflow.md 了解AI协作流程"
    echo "  4. 开始需求分析: cp docs/templates/PRD/prd_template.md docs/PRD.md"
    echo
    echo -e "${BLUE}重要文档:${NC}"
    echo "  - 工作流程: docs/workflows/workflow.md"
    echo "  - 代码规范: docs/standards/pycode_standards.md"
    echo "  - 测试规范: docs/standards/pytest_stands.md"
    echo "  - Git规范: docs/standards/git_commit_std.md"
    echo "  - 文档模板: docs/templates/"
    echo "  - 知识库: docs/knowledge/"
    echo "  - Agent配置: .claude/agents/"
    echo
    log_success "Happy coding with Claude Code AI! 🚀"
}

# 主函数
main() {
    echo -e "${BLUE}"
    cat << "EOF"
   _____ _                 _        _____          _      
  / ____| |               | |      / ____|        | |     
 | |    | | __ _ _   _  __| | ___ | |     ___   __| | ___ 
 | |    | |/ _` | | | |/ _` |/ _ \| |    / _ \ / _` |/ _ \
 | |____| | (_| | |_| | (_| |  __/| |___| (_) | (_| |  __/
  \_____|_|\__,_|\__,_|\__,_|\___| \_____\___/ \__,_|\___|
                                                          
     AI协作开发项目快速搭建脚本
EOF
    echo -e "${NC}"
    
    # 检查帮助参数
    if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    # 执行设置流程
    check_arguments "$@"
    get_script_dir
    create_project_directory
    copy_core_files
    setup_project_structure
    create_project_readme
    show_completion_info
}

# 如果脚本被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi