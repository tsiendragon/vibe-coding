# UI/UX Design Workflow

专门针对移动应用用户界面和用户体验设计的工作流，强调以用户为中心的设计过程。

## 工作流概述

从用户研究到视觉设计的完整UX流程，确保应用提供优秀的用户体验和直观的界面设计。

## 专属Agent团队

| Agent | 角色定位 | 核心输出 | 关键阶段 |
|-------|---------|----------|----------|
| **agent-user-researcher** | 用户研究员 | 用户画像、需求分析 | 用户研究、需求洞察 |
| **agent-interaction-designer** | 交互设计师 | 交互原型、用户流程 | 交互设计、流程优化 |
| **agent-visual-designer** | 视觉设计师 | 视觉规范、UI界面 | 视觉设计、品牌传达 |
| **agent-design-system-architect** | 设计系统架构师 | 设计组件库、规范 | 设计系统、组件标准 |
| **agent-accessibility-specialist** | 无障碍专家 | 可访问性方案 | 无障碍设计、包容性 |
| **agent-usability-tester** | 可用性测试员 | 测试报告、优化建议 | 可用性验证、迭代优化 |

## 工作流阶段

### Phase 1: 用户研究与洞察
**负责**: agent-user-researcher  
**产出**: `docs/USER_RESEARCH.md`

#### 用户画像定义
```markdown
## 主要用户画像

### 用户A: 年轻专业人士
- 年龄: 25-35岁
- 职业: 白领、创业者
- 痛点: 时间紧张，需要高效工具
- 行为: 重度手机用户，追求效率
- 目标: 快速完成任务，简化流程

### 用户B: 技术爱好者
- 年龄: 20-30岁
- 职业: 程序员、设计师
- 痛点: 需要专业功能，定制化需求
- 行为: 关注新技术，喜欢探索
- 目标: 功能丰富，操作灵活
```

#### 用户需求分析
```markdown
## 核心需求
1. **效率需求**: 快速完成日常任务
2. **个性化需求**: 自定义界面和功能
3. **社交需求**: 分享和协作功能
4. **安全需求**: 数据隐私保护

## 使用场景
- 场景1: 通勤路上使用
- 场景2: 办公室工作时使用
- 场景3: 家庭休闲时使用
```

### Phase 2: 信息架构设计
**负责**: agent-interaction-designer  
**协作**: agent-user-researcher  
**产出**: `docs/INFORMATION_ARCHITECTURE.md`

#### 应用结构图
```
应用首页
├── 核心功能区
│   ├── 功能A
│   ├── 功能B
│   └── 功能C
├── 个人中心
│   ├── 个人资料
│   ├── 设置
│   └── 帮助
└── 发现页面
    ├── 推荐内容
    ├── 分类浏览
    └── 搜索功能
```

#### 用户流程设计
```markdown
## 核心用户流程

### 新用户注册流程
欢迎页 → 注册方式选择 → 填写信息 → 验证邮箱 → 完善资料 → 引导教程 → 进入主页

### 核心功能使用流程
主页 → 功能入口 → 参数设置 → 执行操作 → 结果展示 → 保存/分享

### 用户设置流程
个人中心 → 设置入口 → 选择设置项 → 修改配置 → 确认保存 → 返回
```

### Phase 3: 交互原型设计
**负责**: agent-interaction-designer  
**产出**: `docs/INTERACTION_PROTOTYPE.md`

#### 交互规范
```dart
// 手势交互定义
class InteractionGuidelines {
  // 点击反馈
  static const Duration tapFeedbackDuration = Duration(milliseconds: 150);
  
  // 滑动阈值
  static const double swipeThreshold = 50.0;
  
  // 长按时间
  static const Duration longPressDuration = Duration(milliseconds: 500);
  
  // 动画时长
  static const Duration standardAnimation = Duration(milliseconds: 300);
  static const Duration quickAnimation = Duration(milliseconds: 150);
}
```

#### 页面跳转逻辑
```dart
// 导航流程定义
class NavigationFlow {
  static const Map<String, List<String>> flowMap = {
    'login': ['welcome', 'register', 'forgot_password'],
    'main': ['home', 'profile', 'settings'],
    'feature': ['list', 'detail', 'edit'],
  };
  
  static bool canNavigateTo(String from, String to) {
    return flowMap[from]?.contains(to) ?? false;
  }
}
```

### Phase 4: 视觉设计系统
**负责**: agent-visual-designer  
**协作**: agent-design-system-architect  
**产出**: `docs/VISUAL_DESIGN_SYSTEM.md`

#### 色彩系统
```dart
class AppColorPalette {
  // 主色调
  static const Color primary = Color(0xFF2196F3);
  static const Color primaryVariant = Color(0xFF1976D2);
  
  // 辅助色
  static const Color secondary = Color(0xFFFF9800);
  static const Color secondaryVariant = Color(0xFFF57C00);
  
  // 功能色
  static const Color success = Color(0xFF4CAF50);
  static const Color warning = Color(0xFFFF9800);
  static const Color error = Color(0xFFF44336);
  static const Color info = Color(0xFF2196F3);
  
  // 中性色
  static const Color neutral100 = Color(0xFFF5F5F5);
  static const Color neutral200 = Color(0xFFEEEEEE);
  static const Color neutral300 = Color(0xFFE0E0E0);
  static const Color neutral400 = Color(0xFFBDBDBD);
  static const Color neutral500 = Color(0xFF9E9E9E);
  static const Color neutral600 = Color(0xFF757575);
  static const Color neutral700 = Color(0xFF616161);
  static const Color neutral800 = Color(0xFF424242);
  static const Color neutral900 = Color(0xFF212121);
}
```

#### 字体系统
```dart
class AppTypography {
  static const String fontFamily = 'Roboto';
  
  // 标题字体
  static const TextStyle h1 = TextStyle(
    fontSize: 32,
    fontWeight: FontWeight.w300,
    letterSpacing: -0.5,
    height: 1.2,
  );
  
  static const TextStyle h2 = TextStyle(
    fontSize: 28,
    fontWeight: FontWeight.w400,
    letterSpacing: 0,
    height: 1.3,
  );
  
  // 正文字体
  static const TextStyle body1 = TextStyle(
    fontSize: 16,
    fontWeight: FontWeight.w400,
    letterSpacing: 0.15,
    height: 1.5,
  );
  
  static const TextStyle body2 = TextStyle(
    fontSize: 14,
    fontWeight: FontWeight.w400,
    letterSpacing: 0.25,
    height: 1.4,
  );
  
  // 功能字体
  static const TextStyle button = TextStyle(
    fontSize: 14,
    fontWeight: FontWeight.w500,
    letterSpacing: 1.25,
  );
  
  static const TextStyle caption = TextStyle(
    fontSize: 12,
    fontWeight: FontWeight.w400,
    letterSpacing: 0.4,
    height: 1.3,
  );
}
```

### Phase 5: 组件库设计
**负责**: agent-design-system-architect  
**产出**: `docs/COMPONENT_LIBRARY.md`

#### 基础组件
```dart
// 按钮组件规范
class AppButtonStyles {
  static ButtonStyle primary = ElevatedButton.styleFrom(
    backgroundColor: AppColorPalette.primary,
    foregroundColor: Colors.white,
    elevation: 2,
    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(8),
    ),
    textStyle: AppTypography.button,
  );
  
  static ButtonStyle secondary = OutlinedButton.styleFrom(
    foregroundColor: AppColorPalette.primary,
    side: BorderSide(color: AppColorPalette.primary),
    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(8),
    ),
    textStyle: AppTypography.button,
  );
}

// 输入框组件规范
class AppInputDecoration {
  static InputDecoration standard({
    required String labelText,
    String? hintText,
    Widget? prefixIcon,
    Widget? suffixIcon,
  }) {
    return InputDecoration(
      labelText: labelText,
      hintText: hintText,
      prefixIcon: prefixIcon,
      suffixIcon: suffixIcon,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: AppColorPalette.neutral300),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: AppColorPalette.neutral300),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: AppColorPalette.primary, width: 2),
      ),
      errorBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(8),
        borderSide: BorderSide(color: AppColorPalette.error),
      ),
      filled: true,
      fillColor: AppColorPalette.neutral100,
    );
  }
}
```

#### 复合组件
```dart
// 卡片组件
class AppCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final VoidCallback? onTap;
  
  const AppCard({
    Key? key,
    required this.child,
    this.padding,
    this.onTap,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    Widget cardWidget = Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: padding ?? const EdgeInsets.all(16),
        child: child,
      ),
    );
    
    if (onTap != null) {
      cardWidget = InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: cardWidget,
      );
    }
    
    return cardWidget;
  }
}
```

### Phase 6: 无障碍设计
**负责**: agent-accessibility-specialist  
**产出**: `docs/ACCESSIBILITY_GUIDELINES.md`

#### 无障碍规范
```dart
// 语义标签规范
class AccessibilityLabels {
  static Widget addSemantics({
    required Widget child,
    required String label,
    String? hint,
    bool? button,
    bool? header,
  }) {
    return Semantics(
      label: label,
      hint: hint,
      button: button ?? false,
      header: header ?? false,
      child: child,
    );
  }
}

// 色彩对比度检查
class ContrastChecker {
  static bool checkContrast(Color foreground, Color background) {
    final ratio = _calculateContrastRatio(foreground, background);
    return ratio >= 4.5; // WCAG AA标准
  }
  
  static double _calculateContrastRatio(Color color1, Color color2) {
    final l1 = _luminance(color1);
    final l2 = _luminance(color2);
    final lighter = math.max(l1, l2);
    final darker = math.min(l1, l2);
    return (lighter + 0.05) / (darker + 0.05);
  }
}
```

#### 触摸目标规范
```dart
class TouchTargetGuidelines {
  static const double minimumSize = 48.0; // 最小触摸目标
  static const double recommendedSize = 56.0; // 推荐触摸目标
  static const double minimumSpacing = 8.0; // 最小间距
  
  static Widget ensureMinimumTouchTarget({
    required Widget child,
    VoidCallback? onTap,
  }) {
    return SizedBox(
      width: minimumSize,
      height: minimumSize,
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          child: Center(child: child),
        ),
      ),
    );
  }
}
```

### Phase 7: 动效设计
**负责**: agent-interaction-designer  
**协作**: agent-visual-designer  
**产出**: `docs/ANIMATION_GUIDELINES.md`

#### 动画规范
```dart
class AnimationGuidelines {
  // 标准动画时长
  static const Duration quick = Duration(milliseconds: 150);
  static const Duration standard = Duration(milliseconds: 300);
  static const Duration slow = Duration(milliseconds: 500);
  
  // 标准缓动曲线
  static const Curve easeIn = Curves.easeIn;
  static const Curve easeOut = Curves.easeOut;
  static const Curve easeInOut = Curves.easeInOut;
  
  // 弹性动画
  static const Curve bounce = Curves.bounceOut;
  static const Curve elastic = Curves.elasticOut;
}

// 页面转场动画
class PageTransitions {
  static Widget slideTransition(
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
    Widget child,
  ) {
    return SlideTransition(
      position: Tween<Offset>(
        begin: const Offset(1.0, 0.0),
        end: Offset.zero,
      ).animate(CurvedAnimation(
        parent: animation,
        curve: Curves.easeInOut,
      )),
      child: child,
    );
  }
  
  static Widget fadeTransition(
    BuildContext context,
    Animation<double> animation,
    Animation<double> secondaryAnimation,
    Widget child,
  ) {
    return FadeTransition(
      opacity: animation,
      child: child,
    );
  }
}
```

### Phase 8: 可用性测试
**负责**: agent-usability-tester  
**产出**: `docs/USABILITY_TEST_REPORT.md`

#### 测试计划
```markdown
## 可用性测试计划

### 测试目标
1. 验证核心功能的易用性
2. 发现用户操作障碍点
3. 评估界面理解度
4. 测试新用户学习曲线

### 测试方法
- 任务导向测试
- 思维大声说话法
- 眼动追踪测试
- A/B对比测试

### 测试任务
1. 新用户注册和设置
2. 完成核心功能操作
3. 查找特定信息
4. 个性化设置配置

### 成功指标
- 任务完成率: >90%
- 任务完成时间: <预期时间的120%
- 用户满意度: >4.0/5.0
- 错误率: <5%
```

#### 测试实施
```dart
// 用户行为跟踪
class UsabilityTracker {
  static void trackUserAction(String action, Map<String, dynamic> data) {
    Analytics.logEvent(action, parameters: {
      'timestamp': DateTime.now().millisecondsSinceEpoch,
      'user_id': UserService.currentUserId,
      'session_id': SessionService.currentSessionId,
      ...data,
    });
  }
  
  static void trackTaskCompletion(String taskId, bool successful) {
    trackUserAction('task_completion', {
      'task_id': taskId,
      'successful': successful,
      'completion_time': TaskTimer.getElapsedTime(taskId),
    });
  }
}
```

## 设计评审与迭代

### 设计评审检查点
1. **用户研究评审**: 用户画像准确性、需求覆盖完整性
2. **交互设计评审**: 用户流程合理性、交互一致性
3. **视觉设计评审**: 品牌一致性、视觉层次清晰度
4. **技术可行性评审**: 实现难度、性能影响
5. **可用性测试评审**: 测试结果分析、改进方案

### 迭代优化流程
```
设计初版 → 内部评审 → 修改优化 → 用户测试 → 数据分析 → 设计迭代
```

## 设计交付物

### 最终交付清单
- [ ] 用户研究报告
- [ ] 信息架构图
- [ ] 交互原型(Figma/Sketch)
- [ ] 视觉设计稿
- [ ] 设计规范文档
- [ ] 组件库(Figma Components)
- [ ] 切图资源(适配多分辨率)
- [ ] 动效说明文档
- [ ] 可用性测试报告
- [ ] 开发对接文档