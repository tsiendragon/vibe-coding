# Mobile App Development Workflow

专门针对Flutter移动应用开发的工作流，从用户体验设计到应用发布的完整流程。

## 工作流概述

构建原生体验的移动应用，强调用户体验、性能优化、平台适配和应用生命周期管理。

## 专属Agent团队

| Agent | 角色定位 | 核心输出 | 关键阶段 |
|-------|---------|----------|----------|
| **agent-ux-designer** | 用户体验设计师 | UI设计、交互原型 | UX设计、用户研究 |
| **agent-flutter-architect** | Flutter架构师 | 应用架构、技术选型 | 架构设计、技术规划 |
| **agent-mobile-developer** | 移动端开发工程师 | UI实现、功能开发 | 开发实现、功能集成 |
| **agent-state-manager** | 状态管理专家 | 状态架构、数据流 | 状态管理、数据绑定 |
| **agent-platform-integrator** | 平台集成工程师 | 原生功能、第三方集成 | 平台适配、功能集成 |
| **agent-mobile-tester** | 移动端测试工程师 | 测试用例、兼容性测试 | 测试验证、性能优化 |
| **agent-release-manager** | 发布管理员 | 打包发布、应用商店 | 发布管理、版本控制 |

## 详细工作流阶段

### Phase 1: 用户体验设计
**负责**: agent-ux-designer  
**起始文档**: `docs/templates/PRD/prd_template.md` → 复制为 `docs/PRD.md`  
**产出文档**: `docs/UX_DESIGN.md`, `docs/UI_PROTOTYPE.md`

**详细步骤**:
1. **需求分析**: 从`docs/PRD.md`提取用户需求和功能点
2. **用户研究**: 创建用户画像(personas)和使用场景
3. **信息架构**: 设计应用信息结构和导航流程
4. **线框图**: 绘制低保真线框图和页面布局
5. **视觉设计**: 制作高保真UI设计稿
6. **交互原型**: 使用Figma/Adobe XD制作可交互原型
7. **设计系统**: 建立颜色、字体、组件规范

**Git提交**:
```bash
git add docs/UX_DESIGN.md docs/UI_PROTOTYPE.md assets/designs/
git commit -m "feat(ux): complete user experience design and interactive prototype"
```

**成功标准**: 设计原型完整、用户流程清晰、符合移动端交互规范  
**失败处理**: 
- 用户流程不清楚 → 重新梳理信息架构
- 交互设计不符合平台规范 → 参考Material Design/Human Interface指南
**通知下游**: 完成后通知 `agent-flutter-architect` 开始架构设计

---

### Phase 2: Flutter应用架构
**负责**: agent-flutter-architect  
**协作**: agent-ux-designer  
**起始文档**: `docs/UX_DESIGN.md`, `docs/PRD.md`  
**产出文档**: `docs/APP_ARCHITECTURE.md`

**详细步骤**:
1. **架构选型**: 基于需求复杂度选择Clean Architecture或简化架构
2. **目录结构**: 设计Flutter项目目录结构
3. **状态管理**: 选择状态管理方案(Riverpod/Bloc/Provider)
4. **路由设计**: 设计应用路由和导航结构
5. **插件规划**: 选择必要的Flutter插件和第三方库
6. **数据存储**: 设计本地数据存储方案(SQLite/Hive/SharedPreferences)
7. **网络层**: 设计API调用和数据同步架构
8. **主题系统**: 设计应用主题和多语言支持

**Git提交**:
```bash
git add docs/APP_ARCHITECTURE.md pubspec.yaml lib/core/
git commit -m "feat(arch): define Flutter app architecture and project structure"
```

**成功标准**: 架构设计清晰、技术选型合理、可扩展性好  
**失败处理**: 
- 架构过于复杂 → 简化设计，选择更轻量的方案
- 插件兼容性问题 → 重新选择插件或寻找替代方案
**通知下游**: 完成后通知 `agent-state-manager` 设计状态管理

---

### Phase 3: 状态管理设计
**负责**: agent-state-manager  
**起始文档**: `docs/APP_ARCHITECTURE.md`  
**产出文档**: `docs/STATE_MANAGEMENT.md`

**详细步骤**:
1. **状态分析**: 分析应用中的状态类型(UI状态、业务状态、全局状态)
2. **状态结构**: 设计状态类和数据模型
3. **数据流设计**: 规划状态更新和数据流向
4. **Provider设计**: 基于选定方案(Riverpod)设计Provider结构
5. **异步处理**: 设计异步操作和错误处理机制
6. **状态持久化**: 设计状态持久化策略
7. **性能优化**: 设计状态监听优化和rebuild控制
8. **调试工具**: 集成状态调试工具

**Git提交**:
```bash
git add docs/STATE_MANAGEMENT.md lib/core/providers/
git commit -m "feat(state): design comprehensive state management architecture"
```

**成功标准**: 状态架构清晰、数据流合理、性能优化到位  
**失败处理**: 
- 状态结构混乱 → 重新分析状态依赖关系
- 性能问题 → 优化Provider设计和监听机制
**通知下游**: 完成后通知 `agent-mobile-developer` 开始UI开发

---

### Phase 4: UI组件开发
**负责**: agent-mobile-developer  
**协作**: agent-ux-designer  
**起始文档**: `docs/UX_DESIGN.md`, `docs/STATE_MANAGEMENT.md`  
**产出代码**: `lib/presentation/widgets/` 组件库

**详细步骤**:
1. **基础组件**: 实现按钮、输入框、卡片等基础UI组件
2. **主题集成**: 集成Material Design主题系统
3. **响应式布局**: 实现多屏幕尺寸适配
4. **动画效果**: 实现页面切换和微交互动画
5. **组件测试**: 为每个组件编写Widget测试
6. **样式系统**: 建立统一的样式和间距系统
7. **图标字体**: 集成自定义图标和字体
8. **可访问性**: 添加语义标签和无障碍支持

**Git提交** (按组件提交):
```bash
# 基础组件
git add lib/presentation/widgets/common/
git commit -m "feat(ui): implement basic UI components with Material Design"

# 主题系统
git add lib/core/theme/
git commit -m "feat(ui): implement app theme system and responsive layout"

# 动画效果
git add lib/presentation/animations/
git commit -m "feat(ui): add page transitions and micro-interactions"
```

**成功标准**: UI组件完整、样式一致、动画流畅、可访问性好  
**失败处理**: 
- 性能问题 → 优化Widget构建和动画性能
- 适配问题 → 调整响应式布局和屏幕适配
- 可访问性不达标 → 补充语义标签和读屏器支持
**通知下游**: 完成后通知开发功能模块

---

### Phase 5: 功能模块开发
**负责**: agent-mobile-developer  
**协作**: agent-state-manager  
**起始代码**: UI组件库和状态管理架构  
**产出代码**: `lib/features/` 功能模块

**详细步骤**:
1. **模块架构**: 按功能创建独立模块(用户、内容、设置等)
2. **数据模型**: 实现业务数据模型和序列化
3. **页面开发**: 开发各功能页面和导航
4. **状态集成**: 集成状态管理和数据绑定
5. **网络请求**: 实现API调用和数据同步
6. **本地存储**: 实现数据缓存和离线功能
7. **表单处理**: 实现表单验证和提交
8. **错误处理**: 实现全局错误处理和用户反馈

**Git提交** (按功能模块提交):
```bash
# 用户模块
git add lib/features/user/
git commit -m "feat(user): implement user authentication and profile management"

# 内容模块  
git add lib/features/content/
git commit -m "feat(content): implement content display and interaction features"

# 设置模块
git add lib/features/settings/
git commit -m "feat(settings): implement app settings and preferences"
```

**成功标准**: 功能完整、数据流畅、用户体验良好  
**失败处理**: 
- 数据同步问题 → 检查网络层和状态管理
- 页面性能差 → 优化Widget构建和数据加载
- 用户体验问题 → 优化交互流程和错误提示
**通知下游**: 完成后通知 `agent-platform-integrator` 集成平台功能

---

### Phase 6: 平台功能集成
**负责**: agent-platform-integrator  
**起始代码**: 完成的功能模块  
**产出文档**: `docs/PLATFORM_INTEGRATION.md`

**详细步骤**:
1. **权限管理**: 实现相机、位置、存储等权限申请
2. **原生功能**: 集成相机、GPS、传感器等设备功能
3. **推送通知**: 集成Firebase Cloud Messaging推送
4. **社交登录**: 集成Google、Facebook、Apple登录
5. **支付集成**: 集成应用内购买和支付功能
6. **地图服务**: 集成Google Maps或其他地图服务
7. **分享功能**: 实现内容分享到社交平台
8. **深度链接**: 实现应用深度链接和Universal Links

**Git提交**:
```bash
git add lib/core/services/ android/ ios/
git commit -m "feat(platform): integrate camera, location and push notification services"

git add lib/features/auth/social/
git commit -m "feat(platform): add social login integration (Google, Facebook, Apple)"
```

**成功标准**: 平台功能稳定、权限处理完善、用户体验流畅  
**失败处理**: 
- 权限被拒绝 → 优化权限申请流程和说明
- 原生功能崩溃 → 检查平台配置和插件版本
- 集成服务异常 → 检查API配置和网络连接
**通知下游**: 完成后通知 `agent-mobile-tester` 开始测试

---

### Phase 7: 测试与优化
**负责**: agent-mobile-tester  
**协作**: agent-mobile-developer  
**起始代码**: 完整的应用功能  
**产出文档**: `docs/TEST_REPORT.md`

**详细步骤**:
1. **测试环境**: 搭建多设备测试环境(Android/iOS不同版本)
2. **单元测试**: 编写业务逻辑和工具类的单元测试
3. **Widget测试**: 编写UI组件和页面的Widget测试
4. **集成测试**: 编写端到端的集成测试
5. **性能测试**: 使用Flutter DevTools进行性能分析
6. **内存测试**: 检查内存泄漏和资源占用
7. **兼容性测试**: 测试不同设备和系统版本兼容性
8. **用户测试**: 进行真实用户场景测试

**Git提交**:
```bash
git add test/
git commit -m "test(app): add comprehensive test suite with 85%+ coverage"

git add docs/TEST_REPORT.md
git commit -m "docs(test): add mobile app testing report and performance metrics"
```

**成功标准**: 
- 测试覆盖率≥85%
- 应用启动时间<3秒
- 内存使用<100MB
- 关键页面60fps流畅运行
- 兼容主流设备和系统版本

**失败处理**: 
- 性能不达标 → 通知mobile-developer优化代码
- 兼容性问题 → 调整代码适配不同设备
- 测试覆盖率不足 → 补充测试用例
**通知下游**: 完成后通知 `agent-release-manager` 准备发布

---

### Phase 8: 应用发布
**负责**: agent-release-manager  
**协作**: agent-mobile-developer  
**起始代码**: 测试通过的应用  
**产出文档**: `docs/RELEASE_GUIDE.md`

**详细步骤**:
1. **版本管理**: 更新版本号和构建号
2. **签名配置**: 配置Android和iOS发布签名
3. **构建配置**: 配置Release构建优化选项
4. **Android构建**: 生成Release AAB/APK并签名
5. **iOS构建**: 在Xcode中Archive并生成IPA
6. **商店素材**: 准备应用描述、截图、图标等素材
7. **商店提交**: 提交到Google Play和App Store
8. **发布监控**: 监控发布后的崩溃和用户反馈
9. **热修复**: 准备代码推送或紧急更新方案

**Git提交**:
```bash
git add android/app/build.gradle ios/Runner/Info.plist
git commit -m "chore(release): bump version to v1.0.0 for app store release"

git add docs/RELEASE_GUIDE.md fastlane/
git commit -m "docs(release): add app store release guide and automation scripts"
```

**成功标准**: 
- 应用成功上架应用商店
- 审核通过率100%
- 发布后崩溃率<1%
- 用户评分>4.0星

**失败处理**: 
- 审核被拒 → 根据反馈修改并重新提交
- 发布后崩溃 → 紧急发布修复版本
- 用户反馈问题 → 记录问题并规划下个版本修复
**通知上游**: 发布成功后通知所有相关Agent应用上线完成

```dart
// 用户状态管理
class UserState {
  final User? user;
  final bool isLoading;
  final String? error;
  
  const UserState({this.user, this.isLoading = false, this.error});
}

class UserNotifier extends StateNotifier<UserState> {
  UserNotifier(this._repository) : super(const UserState());
  
  final UserRepository _repository;
  
  Future<void> login(String email, String password) async {
    state = state.copyWith(isLoading: true, error: null);
    
    try {
      final user = await _repository.login(email, password);
      state = state.copyWith(user: user, isLoading: false);
    } catch (e) {
      state = state.copyWith(error: e.toString(), isLoading: false);
    }
  }
}
```

### Phase 4: UI组件开发
**负责**: agent-mobile-developer  
**协作**: agent-ux-designer  
**产出**: UI组件库

- 基础组件库构建
- 主题系统实现
- 响应式布局适配
- 动画效果实现

```dart
// 通用按钮组件
class AppButton extends StatelessWidget {
  final String text;
  final VoidCallback? onPressed;
  final ButtonType type;
  
  const AppButton({
    Key? key,
    required this.text,
    this.onPressed,
    this.type = ButtonType.primary,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      style: _getButtonStyle(context, type),
      child: Text(text),
    );
  }
}
```

### Phase 5: 功能模块开发
**负责**: agent-mobile-developer  
**协作**: agent-state-manager  
**产出**: 功能模块代码

- 用户认证模块
- 核心业务功能
- 本地数据存储
- 网络请求处理

```dart
// 用户模块结构
lib/features/user/
├── data/
│   ├── datasources/
│   ├── models/
│   └── repositories/
├── domain/
│   ├── entities/
│   ├── repositories/
│   └── usecases/
└── presentation/
    ├── pages/
    ├── widgets/
    └── providers/
```

### Phase 6: 平台功能集成
**负责**: agent-platform-integrator  
**产出**: `docs/PLATFORM_INTEGRATION.md`

- 设备功能集成(相机、定位、传感器)
- 推送通知
- 应用内购买
- 社交登录集成

```dart
// 相机功能集成
class CameraService {
  static Future<XFile?> takePicture() async {
    final ImagePicker picker = ImagePicker();
    return await picker.pickImage(source: ImageSource.camera);
  }
  
  static Future<bool> requestPermission() async {
    final status = await Permission.camera.request();
    return status.isGranted;
  }
}
```

### Phase 7: 测试与优化
**负责**: agent-mobile-tester  
**协作**: agent-mobile-developer  
**产出**: `docs/TEST_REPORT.md`

- Widget测试
- 集成测试
- 性能测试
- 兼容性测试

```dart
// Widget测试示例
testWidgets('login form should submit with valid credentials', (tester) async {
  await tester.pumpWidget(
    ProviderScope(
      child: MaterialApp(home: LoginPage()),
    ),
  );
  
  // 输入用户名和密码
  await tester.enterText(find.byKey(const Key('email_field')), 'test@example.com');
  await tester.enterText(find.byKey(const Key('password_field')), 'password123');
  
  // 点击登录按钮
  await tester.tap(find.byKey(const Key('login_button')));
  await tester.pumpAndSettle();
  
  // 验证导航到主页
  expect(find.byType(HomePage), findsOneWidget);
});
```

### Phase 8: 应用发布
**负责**: agent-release-manager  
**产出**: `docs/RELEASE_GUIDE.md`

- Android APK/AAB构建
- iOS IPA构建
- 应用商店上架
- 版本更新策略

```bash
# Android发布构建
flutter build appbundle --release

# iOS发布构建
flutter build ios --release
# 然后在Xcode中Archive和上传
```

## 移动端特有考虑

### 性能优化
```dart
// 图片懒加载
class LazyImageWidget extends StatelessWidget {
  final String imageUrl;
  
  @override
  Widget build(BuildContext context) {
    return CachedNetworkImage(
      imageUrl: imageUrl,
      placeholder: (context, url) => const CircularProgressIndicator(),
      errorWidget: (context, url, error) => const Icon(Icons.error),
      memCacheWidth: 300, // 内存优化
    );
  }
}

// 列表优化
class OptimizedListView extends StatelessWidget {
  final List<Item> items;
  
  @override
  Widget build(BuildContext context) {
    return ListView.builder(
      itemCount: items.length,
      cacheExtent: 500, // 预加载范围
      itemBuilder: (context, index) {
        return ItemWidget(item: items[index]);
      },
    );
  }
}
```

### 离线功能
```dart
// 网络状态监听
class ConnectivityService {
  Stream<ConnectivityResult> get connectivityStream =>
      Connectivity().onConnectivityChanged;
  
  Future<bool> get isConnected async {
    final result = await Connectivity().checkConnectivity();
    return result != ConnectivityResult.none;
  }
}

// 离线数据缓存
class OfflineRepository {
  final LocalDatabase _localDb;
  final RemoteApi _remoteApi;
  
  Future<List<Post>> getPosts() async {
    if (await _connectivityService.isConnected) {
      final posts = await _remoteApi.getPosts();
      await _localDb.cachePosts(posts);
      return posts;
    } else {
      return await _localDb.getCachedPosts();
    }
  }
}
```

### 平台适配
```dart
// iOS/Android差异化处理
class PlatformAdaptiveWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    if (Platform.isIOS) {
      return CupertinoButton(
        child: Text('iOS Button'),
        onPressed: () {},
      );
    } else {
      return ElevatedButton(
        child: Text('Android Button'),
        onPressed: () {},
      );
    }
  }
}
```

## 质量标准

### 性能标准
- 应用启动时间: < 3秒
- 页面切换动画: 60fps
- 内存使用: < 100MB (空闲状态)
- 应用包大小: < 20MB

### 用户体验标准
- 界面响应时间: < 100ms
- 网络请求超时: < 10秒
- 错误恢复机制: 100%覆盖
- 无障碍功能: 符合WCAG 2.1

### 代码质量标准
- 测试覆盖率: ≥ 85%
- Widget测试: 关键界面100%
- 代码可读性: 良好
- 性能分析: 定期进行

## 发布策略

### 版本规划
```
1.0.0 - 初始版本(MVP)
1.1.0 - 第一次功能更新
1.1.1 - Bug修复版本
2.0.0 - 重大架构升级
```

### 灰度发布
```dart
// 功能开关
class FeatureFlags {
  static bool get useNewHomePage => 
      RemoteConfig.instance.getBool('use_new_home_page');
  
  static bool get enablePushNotifications =>
      RemoteConfig.instance.getBool('enable_push_notifications');
}

// 在界面中使用
@override
Widget build(BuildContext context) {
  if (FeatureFlags.useNewHomePage) {
    return NewHomePage();
  } else {
    return OldHomePage();
  }
}
```

### A/B测试
```dart
// A/B测试框架
class ABTestManager {
  static String getVariant(String testName) {
    final userId = UserService.instance.currentUser?.id;
    final hash = userId?.hashCode ?? 0;
    return hash % 2 == 0 ? 'A' : 'B';
  }
}

// 使用示例
final variant = ABTestManager.getVariant('button_color_test');
final buttonColor = variant == 'A' ? Colors.blue : Colors.green;
```