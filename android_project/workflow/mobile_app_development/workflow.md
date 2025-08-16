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

## 工作流阶段

### Phase 1: 用户体验设计
**负责**: agent-ux-designer  
**产出**: `docs/UX_DESIGN.md`, 设计原型

- 用户研究与personas
- 用户旅程映射
- 线框图与原型设计
- 交互动效设计

```
用户研究 → 信息架构 → 线框图 → 视觉设计 → 交互原型
```

### Phase 2: Flutter应用架构
**负责**: agent-flutter-architect  
**协作**: agent-ux-designer  
**产出**: `docs/APP_ARCHITECTURE.md`

- Clean Architecture适配
- 状态管理选型(Riverpod/Bloc)
- 路由导航设计
- 插件依赖规划

```dart
// 架构分层
lib/
├── core/           # 核心功能
├── data/           # 数据层
├── domain/         # 领域层
├── presentation/   # 表现层
└── shared/         # 共享组件
```

### Phase 3: 状态管理设计
**负责**: agent-state-manager  
**产出**: `docs/STATE_MANAGEMENT.md`

- 状态结构设计
- 数据流向定义
- 状态持久化策略
- 异步状态处理

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