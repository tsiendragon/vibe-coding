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

# 创建.gitignore文件
create_gitignore() {
    log_info "创建.gitignore文件..."
    
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

    log_success ".gitignore文件创建完成"
}

# 复制核心文档和模板
copy_core_documents() {
    log_info "复制核心文档和模板..."
    
    # 创建目录结构
    mkdir -p {docs/{workflows,standards,templates,knowledge/{best_practices,error_cases}}}
    mkdir -p .claude/agents
    
    # 复制工作流文档到 docs/workflows/
    if [[ -f "$SOURCE_REPO/pytorch_project/workflow/extract_rewrite/extract_rewrite_workflow.md" ]]; then
        cp "$SOURCE_REPO/pytorch_project/workflow/extract_rewrite/extract_rewrite_workflow.md" \
           docs/workflows/
        log_success "复制工作流文档到 docs/workflows/"
    fi
    
    # 复制标准规范到 docs/standards/
    if [[ -d "$SOURCE_REPO/pytorch_project/standards" ]]; then
        cp -r "$SOURCE_REPO/pytorch_project/standards"/* docs/standards/
        log_success "复制标准规范到 docs/standards/"
    fi
    
    # 复制文档模板到 docs/templates/
    if [[ -d "$SOURCE_REPO/pytorch_project/documentation" ]]; then
        cp -r "$SOURCE_REPO/pytorch_project/documentation"/* docs/templates/
        log_success "复制文档模板到 docs/templates/"
    fi
    
    # 复制知识库模板到 docs/knowledge/
    if [[ -d "$SOURCE_REPO/pytorch_project/knowledge" ]]; then
        cp -r "$SOURCE_REPO/pytorch_project/knowledge"/* docs/knowledge/
        log_success "复制知识库模板到 docs/knowledge/"
    fi
    
    # 创建 agent 配置文件
    create_agent_configs
    
    # 复制CLAUDE.md配置文件
    if [[ -f "$SOURCE_REPO/CLAUDE.md" ]]; then
        cp "$SOURCE_REPO/CLAUDE.md" .
        log_success "复制CLAUDE.md配置文件"
    fi
}

# 创建Agent配置文件
create_agent_configs() {
    log_info "创建Agent配置文件..."
    
    # agent-product-manager
    cat > .claude/agents/agent-product-manager.md << 'EOF'
# Agent: Product Manager

## 角色定义
负责需求分析、PRD编写和功能验收的产品经理Agent

## 核心职责
- 需求分析和PRD编写
- 功能验收和用户体验设计
- 需求管理和优先级排序
- 与需求方(human)反复讨论需求可行性

## 主要文档
- `docs/templates/PRD/prd_template.md` - PRD模板
- `docs/PRD.md` - 项目需求文档
- 需求验收报告

## 知识领域
- `docs/knowledge/best_practices/collaboration_patterns.md` - 协作模式
- 需求分析方法和常见问题

## 关键技能
- 业务理解
- 用户体验设计
- 需求管理
- 沟通协调

## 工作流阶段
- PRD起草到确定阶段：主责
- 需求验收阶段：主责
- TECH_SPEC评审：协作参与

## 知识沉淀任务
- 将需求分析的有效方法记录到知识库
- 记录需求理解分歧的解决过程
- 更新需求验收的最佳实践
EOF

    # agent-tech-lead
    cat > .claude/agents/agent-tech-lead.md << 'EOF'
# Agent: Technical Lead

## 角色定义
负责技术方案设计、架构决策和项目协调的技术负责人Agent

## 核心职责
- 技术方案设计和架构决策
- 项目协调和团队领导
- 最终交付决策
- 冲突仲裁和资源协调

## 主要文档
- `docs/templates/TECH_SPEC/TECH_SPEC_template.md` - 技术规格模板
- `docs/TECH_SPEC.md` - 技术设计文档
- `docs/TODO.md` - 项目任务管理
- `docs/prototype_review.md` - 原型评估报告

## 知识领域
- `docs/knowledge/best_practices/tech_solutions.md` - 技术方案
- `docs/knowledge/best_practices/collaboration_patterns.md` - 协作模式

## 关键技能
- 系统架构
- 技术选型
- 团队领导
- 决策管理

## 工作流阶段
- TECH_SPEC起草：主责
- 原型开发协调：主责
- 项目交付决策：主责
- 冲突解决：仲裁者

## 知识沉淀任务
- 记录技术架构设计的最佳实践
- 记录多Agent协作的有效模式
- 记录项目交付决策的标准和流程
EOF

    # agent-researcher
    cat > .claude/agents/agent-researcher.md << 'EOF'
# Agent: Researcher

## 角色定义
负责论文调研、技术可行性分析和理论验证的研究员Agent

## 核心职责
- 论文调研和技术可行性分析
- 理论一致性审核
- 技术趋势分析
- 学术资源调研和方法论指导

## 主要文档
- `docs/templates/research/literature_review_template.md` - 文献综述模板
- `docs/research/literature_review.md` - 文献综述
- `docs/research/feasibility_analysis.md` - 可行性分析
- `docs/research/recommendations.md` - 技术建议

## 知识领域
- `docs/knowledge/best_practices/tech_solutions.md` - 技术方案库

## 关键技能
- 学术研究
- 技术趋势分析
- 理论验证
- 文献调研

## 工作流阶段
- TECH_SPEC调研阶段：主责
- 原型评估：理论验证
- 理论一致性审核：主责

## 知识沉淀任务
- 记录有效的技术调研方法
- 更新技术方案和理论洞察
- 记录理论验证的关键检查点
EOF

    # agent-algorithm-engineer
    cat > .claude/agents/agent-algorithm-engineer.md << 'EOF'
# Agent: Algorithm Engineer

## 角色定义
负责算法实现、模型设计和核心开发的算法工程师Agent

## 核心职责
- 算法实现和模型设计
- 核心开发和性能优化
- 技术难题解决
- 模块化开发和代码实现

## 主要文档
- `docs/templates/PROTOTYPE/PROTOTYPE_template.md` - 原型模板
- `src/` - 源代码目录
- 模块级README.md文件
- 实验结果报告

## 知识领域
- `docs/knowledge/best_practices/code_patterns.md` - 代码模式
- `docs/knowledge/error_cases/common_issues.md` - 常见问题

## 遵循标准
- `docs/standards/pycode_standards.md` - Python编码规范
- `docs/standards/pytorch_standards.md` - PyTorch开发规范
- `docs/standards/git_commit_std.md` - Git提交规范

## 关键技能
- 深度学习
- PyTorch框架
- 算法优化
- 代码实现

## 工作流阶段
- 原型开发：主责
- 完整代码实现：主责
- 模块开发：主责

## 知识沉淀任务
- 记录算法实现的关键技巧
- 记录验证实验设计方法
- 记录bug修复的调试方法
EOF

    # agent-code-reviewer
    cat > .claude/agents/agent-code-reviewer.md << 'EOF'
# Agent: Code Reviewer

## 角色定义
负责代码质量审核、标准检查和持续监控的代码审查员Agent

## 核心职责
- 代码质量审核和标准检查
- 持续监控和改进建议
- 最佳实践推广
- 代码健康度评估

## 主要文档
- 代码审核报告
- 质量改进建议
- 代码质量问题清单

## 知识领域
- `docs/knowledge/best_practices/code_patterns.md` - 代码模式库

## 遵循标准
- `docs/standards/pycode_standards.md` - Python编码规范
- `docs/standards/pytorch_standards.md` - PyTorch开发规范
- `docs/standards/pytest_stands.md` - 测试规范

## 关键技能
- 代码审查
- 编程规范
- PyTorch最佳实践
- 质量保证

## 工作流阶段
- 代码开发全程：持续监控
- 测试代码审查：质量把关
- 最终代码审核：全面审查

## 知识沉淀任务
- 记录代码质量问题和改进建议
- 更新代码审查的有效方法
- 记录代码质量问题识别模式
EOF

    # agent-qa-engineer
    cat > .claude/agents/agent-qa-engineer.md << 'EOF'
# Agent: QA Engineer

## 角色定义
负责测试用例编写、质量保证和性能测试的质量保证工程师Agent

## 核心职责
- 测试用例编写和质量保证
- 性能测试和集成测试
- 质量验收和测试报告
- 鲁棒性测试和稳定性验证

## 主要文档
- `tests/` - 测试代码目录
- `tests/benchmark.md` - 性能测试报告
- `tests/robustness_report.md` - 鲁棒性测试报告
- `docs/final_quality_report.md` - 最终质量报告

## 知识领域
- `docs/knowledge/best_practices/test_strategies.md` - 测试策略
- `docs/knowledge/error_cases/common_issues.md` - 常见问题

## 遵循标准
- `docs/standards/pytest_stands.md` - pytest测试规范
- `docs/standards/git_commit_std.md` - Git提交规范

## 关键技能
- 软件测试
- pytest规范
- 性能分析
- 质量保证

## 工作流阶段
- 模块测试：创建测试用例
- 集成测试：端到端验证
- 质量验收：最终质量把关

## 知识沉淀任务
- 记录测试设计方法和用例模式
- 记录集成测试、性能测试策略
- 记录质量验收的量化标准
EOF

    # agent-docs-writer
    cat > .claude/agents/agent-docs-writer.md << 'EOF'
# Agent: Documentation Writer

## 角色定义
负责技术文档、项目文档和文档体系构建的文档编写员Agent

## 核心职责
- 技术文档和项目文档编写
- 文档体系构建
- 知识整理和传承
- 项目级文档集成

## 主要文档
- `README.md` - 项目主文档
- `docs/` - 项目文档目录
- `docs/templates/` - 文档模板

## 知识领域
- `docs/knowledge/best_practices/collaboration_patterns.md` - 协作模式

## 遵循标准
- `docs/standards/git_commit_std.md` - Git提交规范

## 关键技能
- 技术写作
- 文档管理
- 知识整理
- 信息架构

## 工作流阶段
- 最终文档生成：主责
- 文档体系构建：全程参与

## 知识沉淀任务
- 记录文档体系构建的最佳实践
- 维护文档模板和规范
- 整理项目知识和经验
EOF

    log_success "Agent配置文件创建完成"
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
- 工作流程: \`docs/workflows/extract_rewrite_workflow.md\`

## 贡献指南

1. 阅读工作流文档: \`docs/workflows/extract_rewrite_workflow.md\`
2. 了解Agent角色: 查看 \`.claude/agents/\` 目录
3. 遵循Agent协作模式
4. 及时进行知识沉淀
5. 维护文档和测试

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
    echo "  4. 阅读 docs/workflows/extract_rewrite_workflow.md 了解AI协作流程"
    echo "  5. 开始需求分析: cp docs/templates/PRD/prd_template.md docs/PRD.md"
    echo
    echo -e "${BLUE}重要文档:${NC}"
    echo "  - 工作流程: docs/workflows/extract_rewrite_workflow.md"
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
    check_dependencies
    get_script_dir
    create_project_directory
    create_gitignore
    copy_core_documents
    customize_for_project_type
    create_project_readme
    install_dependencies
    show_completion_info
}

# 如果脚本被直接执行
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi