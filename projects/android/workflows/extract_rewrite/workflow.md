# Flutter App 重写型工作流设计

这个工作流基于现有的移动应用需求或参考项目，提取核心用户交互和业务逻辑，然后设计清晰的Flutter应用架构，重新实现更加用户友好、易维护的移动端应用。需要设计不同的agent赋予他们的角色定位和特长，以及项目不同阶段不同agent之间的协作方式。

## Agent 列表

| Agent | 核心职责 | 主要产出 | 关键技能 | 关键工作阶段 |
|-------|----------|----------|----------|-------------|
| **agent-product-manager** | 需求分析、用户体验设计、功能验收 | `docs/PRD/PRD.md`、用户体验验收报告 | 移动产品设计、用户体验、需求管理 | 需求映射、功能验收 |
| **agent-tech-lead** | 移动架构设计、技术决策、项目协调、最终交付决策 | `docs/TECH_SPEC/TECH_SPEC.md`、`docs/TODO/TODO.md`、架构评估、交付决策 | Flutter架构、状态管理、技术选型、团队领导 | 架构设计、TECH_SPEC评审、UI评估、应用质量评估、项目交付决策 |
| **agent-researcher** | 技术调研、Flutter最佳实践研究、性能优化研究 | `docs/research/flutter_review.md`、`docs/research/recommendations.md`、技术审核报告 | Flutter生态研究、移动开发趋势、UI/UX模式 | 技术调研、TECH_SPEC评审、架构评估、技术一致性审核 |
| **agent-flutter-developer** | UI实现、状态管理、原生功能集成 | 核心UI代码、功能模块代码、模块README | Flutter开发、Dart编程、状态管理、UI优化 | UI原型实现、功能实现、所有应用模块开发 |
| **agent-code-reviewer** | 代码质量审核、标准检查、持续监控 | 代码审核报告、质量改进建议 | 代码审查、编程规范、Flutter最佳实践 | 代码开发全程、测试代码审查、最终代码审核 |
| **agent-qa-engineer** | 测试用例编写、UI测试、性能测试 | 测试代码、Widget测试报告、性能评估报告 | Widget测试、集成测试、性能分析、用户体验测试 | 模块测试、集成测试、性能测试、用户体验验收、质量验收 |
| **agent-docs-writer** | 用户指南、技术文档、文档体系构建 | 用户指南、项目README.md、部署指南 | 技术写作、用户文档、知识整理 | 最终文档生成 |

## Flutter项目特有的工作阶段

### UI原型开发

**前置条件**: TECH_SPEC审核通过，UI设计方案确定

1. **agent-tech-lead** 与 agent-flutter-developer 协作制定UI原型开发计划：
   - 核心界面设计：登录界面、主要功能界面、导航结构
   - 状态管理架构：Provider/Riverpod/Bloc选择和配置
   - 验证目标：UI渲染正确、基本交互可用、状态更新正常

2. **agent-flutter-developer** 实现核心UI原型：
   - 实现主要页面和组件
   - 建立基本的状态管理
   - 配置基本的路由导航
   - 实现基础的用户交互

3. **agent-qa-engineer** 进行UI原型测试：
   - Widget渲染测试
   - 用户交互测试
   - 基本性能测试

### 状态管理实现阶段

**前置条件**: UI原型验证通过

1. **agent-flutter-developer** 进行完整状态管理设计：
   - 设计应用状态结构
   - 实现状态管理模式(Provider/Riverpod/Bloc)
   - 建立数据流和事件处理
   - 实现状态持久化

2. **agent-code-reviewer** 审核状态管理设计：
   - 状态结构合理性
   - 数据流清晰度
   - 性能优化考虑

### 功能模块开发阶段

**前置条件**: 状态管理架构完成

1. **agent-flutter-developer** 实现应用功能模块：
   - 用户模块：登录、注册、个人中心
   - 核心功能模块：主要业务功能实现
   - 数据模块：本地存储、网络请求、缓存
   - 通用模块：工具函数、通用组件

2. **agent-qa-engineer** 进行功能模块测试：
   - Widget测试：每个组件测试
   - 集成测试：页面间交互测试
   - 用户体验测试：完整用户流程验证

### 原生功能集成阶段

**前置条件**: 核心功能模块完成

1. **agent-flutter-developer** 集成原生功能：
   - 设备功能：摄像头、定位、传感器
   - 系统集成：推送通知、分享功能
   - 第三方服务：支付、地图、社交登录
   - 平台适配：Android/iOS特定功能

2. **agent-qa-engineer** 进行原生功能测试：
   - 设备功能测试
   - 平台兼容性测试
   - 性能影响评估

## Flutter特有的质量标准

### UI/UX标准
- Material Design或Cupertino设计规范
- 响应式布局适配不同屏幕
- 流畅的动画和过渡效果
- 一致的用户体验
- 无障碍功能支持

### 性能标准
- 应用启动时间 < 3秒
- 页面切换流畅无卡顿
- 内存使用合理
- 电池消耗优化
- 网络请求优化

### 代码质量标准
- 遵循Dart/Flutter编码规范
- 组件复用和模块化
- 状态管理清晰合理
- 错误处理完善
- 代码可读性和维护性

### 测试标准
- Widget测试覆盖率≥85%
- 集成测试覆盖关键用户流程
- 性能测试验证响应时间
- 多设备兼容性测试
- 用户体验测试

## 工具和技术栈

### 开发工具
- Flutter SDK进行应用开发
- Dart语言编程
- Flutter DevTools进行调试和性能分析
- Android Studio/VS Code开发环境

### 状态管理
- Riverpod (推荐)：类型安全、测试友好
- Bloc：复杂状态管理
- Provider：简单状态管理

### 网络和数据
- Dio进行HTTP请求
- Hive/SharedPreferences本地存储
- Drift(Moor)进行SQLite数据库操作
- JSON序列化和反序列化

### UI组件库
- Material Design组件
- Cupertino组件
- 自定义组件库
- 动画和过渡效果

### 测试工具
- Flutter Test进行Widget测试
- Integration Test进行集成测试
- Mockito进行Mock测试
- Golden Test进行UI回归测试

### 部署和分发
- Android APK/AAB打包
- iOS IPA打包
- App Store/Google Play发布
- Firebase分发测试

### 开发规范
- 遵循 `docs/standards/flutter_standards.md` - Flutter开发规范
- 遵循 `docs/standards/dart_standards.md` - Dart编码标准  
- 遵循 `docs/standards/dart_test_standards.md` - Flutter测试标准

### Git提交规范
- UI实现完成: `feat(ui): implement [page_name] user interface`
- 功能实现完成: `feat(feature): implement [feature_name] functionality`
- 状态管理完成: `feat(state): implement [state_name] state management`
- 测试完成: `test(widget): add comprehensive tests for [feature_name]`
- 性能优化: `perf(app): optimize [specific_optimization]`

## 移动端特有的考虑

### 用户体验重点
- 直观的导航和操作
- 快速响应的交互
- 离线功能支持
- 数据同步机制
- 错误恢复和重试

### 性能优化重点
- 懒加载和分页
- 图片压缩和缓存
- 网络请求优化
- 内存管理
- 电池续航优化

### 安全考虑
- 本地数据加密
- 网络传输安全
- 用户隐私保护
- 防止逆向工程
- 安全的用户认证