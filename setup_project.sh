#!/bin/bash

# Claude Code AI协作开发项目快速搭建脚本
# 使用方法: ./setup_project.sh <project_name> <project_type> [target_directory]

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
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

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    local deps=("git" "python3" "pip3")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "缺少依赖: $dep"
            exit 1
        fi
    done
    
    log_success "系统依赖检查完成"
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
        log_warning "目录已存在: $PROJECT_PATH"
        read -p "是否覆盖现有目录? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "操作已取消"
            exit 0
        fi
        rm -rf "$PROJECT_PATH"
    fi
    
    mkdir -p "$PROJECT_PATH"
    cd "$PROJECT_PATH"
    log_success "创建项目目录: $PROJECT_PATH"
}

# 初始化Git仓库
init_git_repo() {
    log_info "初始化Git仓库..."
    git init
    
    # 创建.gitignore
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt
checkpoints/
runs/
wandb/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Environment
.env
.venv
env/
venv/

# Jupyter
.ipynb_checkpoints

# Claude Code
.claude_session
EOF

    log_success "Git仓库初始化完成"
}

# 复制核心文档和模板
copy_core_documents() {
    log_info "复制核心文档和模板..."
    
    # 创建目录结构
    mkdir -p {docs,standards,knowledge/{best_practices,error_cases},documentation}
    mkdir -p workflow/extract_rewrite
    
    # 复制工作流文档
    if [[ -f "$SOURCE_REPO/pytorch_project/workflow/extract_rewrite/extract_rewrite_workflow.md" ]]; then
        cp "$SOURCE_REPO/pytorch_project/workflow/extract_rewrite/extract_rewrite_workflow.md" \
           workflow/extract_rewrite/
        log_success "复制工作流文档"
    fi
    
    # 复制标准规范
    if [[ -d "$SOURCE_REPO/pytorch_project/standards" ]]; then
        cp -r "$SOURCE_REPO/pytorch_project/standards"/* standards/
        log_success "复制标准规范"
    fi
    
    # 复制文档模板
    if [[ -d "$SOURCE_REPO/pytorch_project/documentation" ]]; then
        cp -r "$SOURCE_REPO/pytorch_project/documentation"/* documentation/
        log_success "复制文档模板"
    fi
    
    # 复制知识库模板
    if [[ -d "$SOURCE_REPO/pytorch_project/knowledge" ]]; then
        cp -r "$SOURCE_REPO/pytorch_project/knowledge"/* knowledge/
        log_success "复制知识库模板"
    fi
    
    # 复制CLAUDE.md配置文件
    if [[ -f "$SOURCE_REPO/CLAUDE.md" ]]; then
        cp "$SOURCE_REPO/CLAUDE.md" .
        log_success "复制CLAUDE.md配置文件"
    fi
}

# 根据项目类型定制配置
customize_for_project_type() {
    log_info "为项目类型 '$PROJECT_TYPE' 定制配置..."
    
    case "$PROJECT_TYPE" in
        pytorch)
            setup_pytorch_project
            ;;
        web)
            setup_web_project
            ;;
        data-science)
            setup_data_science_project
            ;;
        research)
            setup_research_project
            ;;
        general)
            setup_general_project
            ;;
    esac
}

# PyTorch项目设置
setup_pytorch_project() {
    log_info "设置PyTorch项目..."
    
    # 创建PyTorch特定目录
    mkdir -p {src,tests/{unit,integration,e2e},configs,data,models,notebooks,scripts}
    
    # 创建requirements.txt
    cat > requirements.txt << EOF
torch>=2.0.0
torchvision>=0.15.0
pytorch-lightning>=2.0.0
transformers>=4.20.0
datasets>=2.10.0
wandb>=0.15.0
tensorboard>=2.13.0
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
ruff>=0.0.270
mypy>=1.0.0
timm>=0.9.0
omegaconf>=2.3.0
hydra-core>=1.3.0
EOF

    # 创建PyTorch Lightning模板
    cat > src/lightning_module_template.py << 'EOF'
"""PyTorch Lightning模块模板"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Dict, Tuple

class LightningModuleTemplate(pl.LightningModule):
    """标准Lightning模块模板"""
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }
EOF

    # 创建配置文件模板
    cat > configs/config.yaml << EOF
# 项目配置文件
project:
  name: "$PROJECT_NAME"
  version: "0.1.0"
  description: "PyTorch深度学习项目"

model:
  name: "resnet50"
  num_classes: 10
  pretrained: true

training:
  batch_size: 32
  learning_rate: 1e-3
  weight_decay: 0.01
  max_epochs: 100
  
data:
  root: "./data"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

logging:
  project_name: "$PROJECT_NAME"
  entity: "your_wandb_entity"
EOF

    log_success "PyTorch项目设置完成"
}

# Web项目设置
setup_web_project() {
    log_info "设置Web项目..."
    
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
    "test": "jest",
    "test:coverage": "jest --coverage"
  },
  "dependencies": {
    "next": "^13.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@types/node": "^18.0.0",
    "@types/react": "^18.0.0",
    "eslint": "^8.0.0",
    "eslint-config-next": "^13.0.0",
    "jest": "^29.0.0",
    "typescript": "^4.9.0"
  }
}
EOF

    log_success "Web项目设置完成"
}

# 数据科学项目设置
setup_data_science_project() {
    log_info "设置数据科学项目..."
    
    mkdir -p {data/{raw,processed,external},notebooks,src,reports,models}
    
    cat > requirements.txt << EOF
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
jupyter>=1.0.0
ipykernel>=6.20.0
plotly>=5.12.0
streamlit>=1.20.0
pytest>=7.0.0
black>=23.0.0
ruff>=0.0.270
EOF

    # 创建数据科学项目结构说明
    cat > README.md << EOF
# $PROJECT_NAME

数据科学项目

## 项目结构

\`\`\`
├── data/
│   ├── raw/          # 原始数据
│   ├── processed/    # 处理后的数据
│   └── external/     # 外部数据源
├── notebooks/        # Jupyter notebooks
├── src/             # 源代码
├── reports/         # 分析报告
├── models/          # 训练好的模型
└── docs/            # 文档
\`\`\`

## 快速开始

1. 安装依赖: \`pip install -r requirements.txt\`
2. 启动Jupyter: \`jupyter lab\`
3. 开始数据探索和分析
EOF

    log_success "数据科学项目设置完成"
}

# 研究项目设置
setup_research_project() {
    log_info "设置研究项目..."
    
    mkdir -p {papers,experiments,data,analysis,presentations}
    
    cat > requirements.txt << EOF
jupyter>=1.0.0
matplotlib>=3.6.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=1.5.0
seaborn>=0.12.0
plotly>=5.12.0
EOF

    log_success "研究项目设置完成"
}

# 通用项目设置
setup_general_project() {
    log_info "设置通用项目..."
    
    mkdir -p {src,tests,docs}
    
    cat > requirements.txt << EOF
pytest>=7.0.0
black>=23.0.0
ruff>=0.0.270
mypy>=1.0.0
EOF

    log_success "通用项目设置完成"
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

## 工作流阶段

1. **需求分析**: PRD编写和需求确认
2. **技术设计**: TECH_SPEC设计和多Agent评审
3. **原型开发**: 核心算法原型实现和验证
4. **完整开发**: 模块化开发和持续质量保证
5. **测试验证**: 集成测试、性能测试、鲁棒性测试
6. **项目验收**: 多维度验收和交付决策

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
   cp documentation/PRD/prd_template.md docs/PRD.md
   \`\`\`

2. **技术设计阶段**  
   \`\`\`bash
   # 创建TECH_SPEC文档
   cp documentation/TECH_SPEC/TECH_SPEC_template.md docs/TECH_SPEC.md
   \`\`\`

3. **原型开发阶段**
   \`\`\`bash
   # 创建原型文档
   cp documentation/PROTOTYPE/PROTOTYPE_template.md docs/PROTOTYPE.md
   \`\`\`

## 文档结构

- \`docs/\`: 项目文档
- \`documentation/\`: 文档模板
- \`standards/\`: 代码规范和测试标准
- \`knowledge/\`: 最佳实践和错误案例库
- \`workflow/\`: AI协作工作流文档

## 代码规范

- Python代码: 遵循 \`standards/pycode_standards.md\`
- 测试代码: 遵循 \`standards/pytest_stands.md\`
- Git提交: 遵循 \`standards/git_commit_std.md\`

## 贡献指南

1. 阅读工作流文档: \`workflow/extract_rewrite/extract_rewrite_workflow.md\`
2. 遵循Agent协作模式
3. 及时进行知识沉淀
4. 维护文档和测试

## 许可证

<!-- 添加许可证信息 -->
EOF

    log_success "项目README创建完成"
}

# 安装Python依赖
install_dependencies() {
    if [[ -f "requirements.txt" ]]; then
        log_info "安装Python依赖..."
        
        # 检查是否在虚拟环境中
        if [[ -z "$VIRTUAL_ENV" ]]; then
            log_warning "建议在虚拟环境中安装依赖"
            read -p "是否创建虚拟环境? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                python3 -m venv venv
                source venv/bin/activate
                log_success "虚拟环境创建完成"
            fi
        fi
        
        pip3 install -r requirements.txt
        log_success "Python依赖安装完成"
    fi
}

# 创建初始提交
create_initial_commit() {
    log_info "创建初始Git提交..."
    
    git add .
    git commit -m "feat: initialize $PROJECT_TYPE project with Claude Code AI collaboration workflow

- Add AI collaboration workflow documentation
- Add coding standards and testing guidelines  
- Add document templates for PRD, TECH_SPEC, PROTOTYPE
- Add knowledge base templates for best practices
- Configure project structure for $PROJECT_TYPE development

🤖 Generated with Claude Code AI Collaboration Setup

Co-Authored-By: Claude <noreply@anthropic.com>"

    log_success "初始提交创建完成"
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
    
    if [[ -f "$PROJECT_PATH/venv/bin/activate" ]]; then
        echo "  2. source venv/bin/activate"
    fi
    
    echo "  3. 阅读 README.md 了解项目结构"
    echo "  4. 阅读 workflow/extract_rewrite/extract_rewrite_workflow.md 了解AI协作流程"
    echo "  5. 开始需求分析: cp documentation/PRD/prd_template.md docs/PRD.md"
    echo
    echo -e "${BLUE}重要文档:${NC}"
    echo "  - 工作流程: workflow/extract_rewrite/extract_rewrite_workflow.md"
    echo "  - 代码规范: standards/pycode_standards.md"
    echo "  - 测试规范: standards/pytest_stands.md"
    echo "  - Git规范: standards/git_commit_std.md"
    echo "  - 文档模板: documentation/"
    echo "  - 知识库: knowledge/"
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
    check_dependencies
    get_script_dir
    create_project_directory
    init_git_repo
    copy_core_documents
    customize_for_project_type
    create_project_readme
    install_dependencies
    create_initial_commit
    show_completion_info
}

# 如果脚本被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi