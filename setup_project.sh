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
    方式1 (传统方式): ./setup_project.sh <project_name> <project_type> [target_directory] [--existing]
    方式2 (智能方式): ./setup_project.sh <project_path>

参数说明:
    project_path      项目完整路径 (智能模式，自动判断新项目或已有项目)
    project_name      项目名称 (传统模式必需)
    project_type      项目类型 (传统模式必需)
    target_directory  目标目录 (传统模式可选，默认为当前目录)
    --existing        在已有项目中设置Claude配置 (传统模式可选)

支持的项目类型:
    1) pytorch          PyTorch深度学习项目
    2) backend          FastAPI后端项目
    3) android          Flutter移动应用项目
    4) web              Web开发项目  
    5) data-science     数据科学项目
    6) research         研究型项目
    7) general          通用软件项目

示例:
    智能模式:
      ./setup_project.sh /home/user/repos/my-project
      ./setup_project.sh ~/projects/new-app
      ./setup_project.sh .
    
    传统模式:
      ./setup_project.sh my-classifier pytorch
      ./setup_project.sh my-api backend
      ./setup_project.sh existing-project pytorch . --existing

EOF
}

# 显示项目类型选择菜单
show_project_type_menu() {
    echo
    echo -e "${BLUE}请选择项目类型:${NC}"
    echo "1) pytorch      - PyTorch深度学习项目"
    echo "2) backend      - FastAPI后端项目" 
    echo "3) android      - Flutter移动应用项目"
    echo "4) web          - Web开发项目"
    echo "5) data-science - 数据科学项目"
    echo "6) research     - 研究型项目"
    echo "7) general      - 通用软件项目"
    echo
}

# 获取用户选择的项目类型
get_project_type_choice() {
    local choice
    while true; do
        read -p "请输入选择 (1-7): " choice
        case $choice in
            1) PROJECT_TYPE="pytorch"; break ;;
            2) PROJECT_TYPE="backend"; break ;;
            3) PROJECT_TYPE="android"; break ;;
            4) PROJECT_TYPE="web"; break ;;
            5) PROJECT_TYPE="data-science"; break ;;
            6) PROJECT_TYPE="research"; break ;;
            7) PROJECT_TYPE="general"; break ;;
            *) 
                log_error "无效选择，请输入 1-7"
                ;;
        esac
    done
}

# 智能检测项目状态
detect_project_status() {
    local project_path="$1"
    
    if [[ ! -d "$project_path" ]]; then
        echo "new_project"
    elif [[ -z "$(ls -A "$project_path" 2>/dev/null)" ]]; then
        echo "empty_project"
    else
        echo "existing_project"
    fi
}

# 智能模式参数处理
handle_smart_mode() {
    local project_path="$1"
    
    # 转换相对路径为绝对路径
    project_path="$(cd "$(dirname "$project_path")" 2>/dev/null && pwd)/$(basename "$project_path")" || project_path="$1"
    
    PROJECT_NAME="$(basename "$project_path")"
    
    local project_status=$(detect_project_status "$project_path")
    
    log_info "项目路径: $project_path"
    log_info "项目名称: $PROJECT_NAME"
    
    case $project_status in
        "new_project")
            log_info "检测结果: 新项目 (目录不存在)"
            EXISTING_PROJECT=false
            TARGET_DIR="$(dirname "$project_path")"
            ;;
        "empty_project")
            log_info "检测结果: 新项目 (空目录)"
            EXISTING_PROJECT=false
            PROJECT_PATH="$project_path"
            TARGET_DIR="$(dirname "$project_path")"
            ;;
        "existing_project")
            log_info "检测结果: 已有项目 (包含文件)"
            EXISTING_PROJECT=true
            PROJECT_PATH="$project_path"
            TARGET_DIR="$project_path"
            ;;
    esac
    
    show_project_type_menu
    get_project_type_choice
    
    log_success "已选择项目类型: $PROJECT_TYPE"
}

# 传统模式参数处理
handle_traditional_mode() {
    PROJECT_NAME="$1"
    PROJECT_TYPE="$2"
    
    # 检查是否有--existing参数
    EXISTING_PROJECT=false
    if [[ "$3" == "--existing" ]] || [[ "$4" == "--existing" ]]; then
        EXISTING_PROJECT=true
        # 如果第三个参数是--existing，那么目标目录就是当前目录
        if [[ "$3" == "--existing" ]]; then
            TARGET_DIR="$(pwd)"
        else
            TARGET_DIR="$3"
        fi
    else
        TARGET_DIR="${3:-$(pwd)}"
    fi

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
    if [[ "$EXISTING_PROJECT" == "true" ]]; then
        log_info "模式: 在已有项目中设置Claude配置"
    fi
}

# 检查参数
check_arguments() {
    if [[ $# -lt 1 ]]; then
        log_error "参数不足"
        show_help
        exit 1
    fi
    
    # 判断是智能模式还是传统模式
    if [[ $# -eq 1 ]] && [[ "$1" =~ ^[/~.]|^[a-zA-Z]: ]]; then
        # 智能模式: 单个参数且看起来像路径
        log_info "使用智能模式"
        handle_smart_mode "$1"
    elif [[ $# -ge 2 ]] && [[ "$2" =~ ^(pytorch|backend|android|web|data-science|research|general)$ ]]; then
        # 传统模式: 至少两个参数且第二个参数是有效的项目类型
        log_info "使用传统模式"
        handle_traditional_mode "$@"
    else
        log_error "参数格式错误"
        echo
        log_info "智能模式: ./setup_project.sh <project_path>"
        log_info "传统模式: ./setup_project.sh <project_name> <project_type> [target_directory] [--existing]"
        echo
        show_help
        exit 1
    fi
}

# 获取脚本所在目录
get_script_dir() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SOURCE_REPO="$SCRIPT_DIR"
    log_info "源仓库路径: $SOURCE_REPO"
}

# 创建项目目录
create_project_directory() {
    if [[ "$EXISTING_PROJECT" == "true" ]]; then
        # 在已有项目模式下，直接使用目标目录
        PROJECT_PATH="$TARGET_DIR"
        cd "$PROJECT_PATH"
        log_success "在已有项目中设置Claude配置: $PROJECT_PATH"
    else
        # 新项目模式
        if [[ -z "$PROJECT_PATH" ]]; then
            # 传统模式：创建新目录
            PROJECT_PATH="$TARGET_DIR/$PROJECT_NAME"
            
            if [[ -d "$PROJECT_PATH" ]]; then
                log_error "目录已存在: $PROJECT_PATH"
                exit 1
            fi
            
            mkdir -p "$PROJECT_PATH"
        else
            # 智能模式：PROJECT_PATH已经设定
            mkdir -p "$PROJECT_PATH"
        fi
        
        cd "$PROJECT_PATH"
        log_success "创建项目目录: $PROJECT_PATH"
    fi
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