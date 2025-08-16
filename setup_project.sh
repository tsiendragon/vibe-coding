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
    backend          FastAPI后端项目
    android          Flutter移动应用项目
    web              Web开发项目  
    data-science     数据科学项目
    research         研究型项目
    general          通用软件项目

示例:
    ./setup_project.sh my-classifier pytorch
    ./setup_project.sh my-api backend
    ./setup_project.sh my-app android ~/projects/

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
        pytorch|backend|android|web|data-science|research|general)
            ;;
        *)
            log_error "不支持的项目类型: $PROJECT_TYPE"
            log_info "支持的类型: pytorch, backend, android, web, data-science, research, general"
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
    
    # 创建目录结构 - 只创建.claude和docs
    mkdir -p docs/workflows
    mkdir -p docs/standards
    mkdir -p docs/templates
    mkdir -p docs/knowledge/best_practices
    mkdir -p docs/knowledge/error_cases
    mkdir -p .claude/agents
    
    # 复制CLAUDE.md配置文件
    if [[ -f "$SOURCE_REPO/CLAUDE.md" ]]; then
        cp "$SOURCE_REPO/CLAUDE.md" .
        log_success "复制 CLAUDE.md"
    fi
    
    # 复制工作流文档
    WORKFLOW_COPIED=false
    for workflow_path in \
        "$SOURCE_REPO/${PROJECT_TYPE}_project/workflow/extract_rewrite/workflow.md" \
        "$SOURCE_REPO/pytorch_project/workflow/extract_rewrite/workflow.md"; do
        if [[ -f "$workflow_path" ]]; then
            cp "$workflow_path" docs/workflows/
            log_success "复制工作流文档"
            WORKFLOW_COPIED=true
            break
        fi
    done
    
    if [[ "$WORKFLOW_COPIED" == "false" ]]; then
        log_error "警告: 未找到工作流文档"
    fi
    
    # 复制标准规范
    STANDARDS_COPIED=false
    for standards_path in \
        "$SOURCE_REPO/${PROJECT_TYPE}_project/standards" \
        "$SOURCE_REPO/pytorch_project/standards"; do
        if [[ -d "$standards_path" ]]; then
            cp -r "$standards_path"/* docs/standards/
            log_success "复制标准规范"
            STANDARDS_COPIED=true
            break
        fi
    done
    
    if [[ "$STANDARDS_COPIED" == "false" ]]; then
        log_error "警告: 未找到标准规范"
    fi
    
    # 复制文档模板
    TEMPLATES_COPIED=false
    for templates_path in \
        "$SOURCE_REPO/${PROJECT_TYPE}_project/documentation" \
        "$SOURCE_REPO/pytorch_project/documentation"; do
        if [[ -d "$templates_path" ]]; then
            cp -r "$templates_path"/* docs/templates/
            log_success "复制文档模板"
            TEMPLATES_COPIED=true
            break
        fi
    done
    
    if [[ "$TEMPLATES_COPIED" == "false" ]]; then
        log_error "警告: 未找到文档模板"
    fi
    
    # 复制知识库
    KNOWLEDGE_COPIED=false
    for knowledge_path in \
        "$SOURCE_REPO/${PROJECT_TYPE}_project/knowledge" \
        "$SOURCE_REPO/pytorch_project/knowledge"; do
        if [[ -d "$knowledge_path" ]]; then
            cp -r "$knowledge_path"/* docs/knowledge/
            log_success "复制知识库"
            KNOWLEDGE_COPIED=true
            break
        fi
    done
    
    if [[ "$KNOWLEDGE_COPIED" == "false" ]]; then
        log_error "警告: 未找到知识库"
    fi
    
    # 复制Agent配置
    AGENTS_COPIED=false
    for agents_path in \
        "$SOURCE_REPO/${PROJECT_TYPE}_project/agents" \
        "$SOURCE_REPO/pytorch_project/agents"; do
        if [[ -d "$agents_path" ]]; then
            cp -r "$agents_path"/* .claude/agents/
            log_success "复制Agent配置"
            AGENTS_COPIED=true
            break
        fi
    done
    
    if [[ "$AGENTS_COPIED" == "false" ]]; then
        log_error "警告: 未找到Agent配置"
    fi
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
    echo "  2. 阅读 docs/workflows/workflow.md 了解AI协作流程"
    echo "  3. 开始需求分析: cp docs/templates/PRD/prd_template.md docs/PRD.md"
    echo
    echo -e "${BLUE}重要文档:${NC}"
    echo "  - 工作流程: docs/workflows/workflow.md"
    echo "  - 代码规范: docs/standards/"
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
    show_completion_info
}

# 如果脚本被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi