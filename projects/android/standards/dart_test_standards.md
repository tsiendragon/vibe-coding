# Dart/Flutter 测试标准

## 测试分层策略

### 1. 单元测试 (Unit Tests)
- **覆盖目标**: 业务逻辑、工具函数、数据模型
- **测试文件**: `test/unit/`
- **命名规范**: `test_<target>_<behavior>[_<condition>].dart`
- **覆盖率要求**: ≥90%

```dart
// 测试业务逻辑
group('UserRepository', () {
  late UserRepository repository;
  late MockApiService mockApiService;
  
  setUp(() {
    mockApiService = MockApiService();
    repository = UserRepositoryImpl(mockApiService);
  });
  
  test('should return Right when API call succeeds', () async {
    // Arrange
    const user = User(id: '1', name: 'John');
    when(() => mockApiService.fetchUser('1'))
        .thenAnswer((_) async => user);
    
    // Act
    final result = await repository.getUser('1');
    
    // Assert
    expect(result, isA<Right<Failure, User>>());
    result.fold(
      (l) => fail('Should not return error'),
      (r) => expect(r, equals(user)),
    );
  });
  
  test('should return Left when API call fails', () async {
    // Arrange
    when(() => mockApiService.fetchUser('1'))
        .thenThrow(NetworkException('Network error'));
    
    // Act
    final result = await repository.getUser('1');
    
    // Assert
    expect(result, isA<Left<Failure, User>>());
  });
});
```

### 2. Widget测试 (Widget Tests)
- **覆盖目标**: UI组件、用户交互、状态变化
- **测试文件**: `test/widget/`
- **关键测试点**: 渲染、交互、状态更新

```dart
testWidgets('should display loading indicator when loading', (tester) async {
  // Arrange
  await tester.pumpWidget(
    ProviderScope(
      overrides: [
        userProvider.overrideWith((ref) {
          final notifier = MockUserNotifier();
          when(() => notifier.state).thenReturn(
            const AsyncValue.loading(),
          );
          return notifier;
        }),
      ],
      child: const MaterialApp(home: UserListPage()),
    ),
  );
  
  // Assert
  expect(find.byType(CircularProgressIndicator), findsOneWidget);
});

testWidgets('should navigate to detail page when item tapped', (tester) async {
  // Arrange
  await tester.pumpWidget(
    MaterialApp(
      home: const UserListPage(),
      routes: {
        '/detail': (context) => const UserDetailPage(),
      },
    ),
  );
  
  // Act
  await tester.tap(find.byKey(const Key('user-item-1')));
  await tester.pumpAndSettle();
  
  // Assert
  expect(find.byType(UserDetailPage), findsOneWidget);
});
```

### 3. 集成测试 (Integration Tests)
- **覆盖目标**: 端到端用户流程、API集成
- **测试文件**: `test_driver/` 或 `integration_test/`
- **测试场景**: 完整用户旅程

```dart
void main() {
  group('User Authentication Flow', () {
    testWidgets('should complete login flow successfully', (tester) async {
      // 启动应用
      await tester.pumpWidget(MyApp());
      
      // 验证登录页面
      expect(find.byType(LoginPage), findsOneWidget);
      
      // 输入凭据
      await tester.enterText(find.byKey(const Key('email-field')), 'test@example.com');
      await tester.enterText(find.byKey(const Key('password-field')), 'password123');
      
      // 点击登录按钮
      await tester.tap(find.byKey(const Key('login-button')));
      await tester.pumpAndSettle();
      
      // 验证导航到主页
      expect(find.byType(HomePage), findsOneWidget);
    });
  });
}
```

## 测试工具和Mock

### Mock策略
```dart
// 使用mocktail创建Mock
class MockUserRepository extends Mock implements UserRepository {}
class MockApiService extends Mock implements ApiService {}

// 手动Mock用于简单场景
class FakeUser extends Fake implements User {
  @override
  String get id => '1';
  @override
  String get name => 'Test User';
}
```

### 测试数据工厂
```dart
class UserFactory {
  static User createUser({
    String? id,
    String? name,
    String? email,
  }) {
    return User(
      id: id ?? '1',
      name: name ?? 'Test User',
      email: email ?? 'test@example.com',
    );
  }
  
  static List<User> createUserList(int count) {
    return List.generate(
      count,
      (index) => createUser(
        id: '$index',
        name: 'User $index',
        email: 'user$index@example.com',
      ),
    );
  }
}
```

## 测试配置

### pubspec.yaml
```yaml
dev_dependencies:
  flutter_test:
    sdk: flutter
  integration_test:
    sdk: flutter
  
  # Mock库
  mocktail: ^1.0.0
  
  # 测试工具
  golden_toolkit: ^0.15.0
  alchemist: ^0.7.0
```

### test配置
```dart
// test/helpers/pump_app.dart
extension AppTester on WidgetTester {
  Future<void> pumpApp(Widget widget) {
    return pumpWidget(
      ProviderScope(
        child: MaterialApp(
          home: widget,
          theme: AppTheme.light,
        ),
      ),
    );
  }
}

// test/helpers/mock_providers.dart
List<Override> get mockProviders => [
  userRepositoryProvider.overrideWithValue(MockUserRepository()),
  apiServiceProvider.overrideWithValue(MockApiService()),
];
```

## 测试最佳实践

### MUST遵循的规则

1. **测试隔离**: 每个测试都应该独立运行
2. **数据清理**: 使用setUp/tearDown清理测试状态
3. **命名清晰**: 测试名称应描述行为和期望结果
4. **AAA模式**: Arrange-Act-Assert结构
5. **模拟外部依赖**: 不依赖真实网络、数据库
6. **覆盖率监控**: 关键业务逻辑100%覆盖

### MUST NOT的禁止项

1. **依赖顺序**: 测试不应依赖其他测试的执行顺序
2. **硬编码时间**: 使用可控制的时间Mock
3. **网络请求**: 单元测试中不进行真实网络调用
4. **平台依赖**: 避免平台特定的测试逻辑
5. **忽略失败**: 所有测试失败都必须修复

## 测试命令

### 运行测试
```bash
# 运行所有测试
flutter test

# 运行单元测试
flutter test test/unit/

# 运行Widget测试
flutter test test/widget/

# 生成覆盖率报告
flutter test --coverage
genhtml coverage/lcov.info -o coverage/html

# 运行集成测试
flutter drive --driver=test_driver/integration_test.dart \
  --target=integration_test/app_test.dart
```

### CI/CD配置
```yaml
# .github/workflows/test.yml
- name: Run tests
  run: flutter test --coverage --reporter=github

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: coverage/lcov.info
```

## Golden测试

### 视觉回归测试
```dart
testWidgets('should match golden file', (tester) async {
  await tester.pumpWidget(
    MaterialApp(
      home: UserCard(
        user: UserFactory.createUser(),
      ),
    ),
  );
  
  await expectLater(
    find.byType(UserCard),
    matchesGoldenFile('goldens/user_card.png'),
  );
});
```

### 生成Golden文件
```bash
# 生成新的golden文件
flutter test --update-goldens
```

## 性能测试

### 基准测试
```dart
void main() {
  group('Performance Tests', () {
    testWidgets('should render large list efficiently', (tester) async {
      final users = UserFactory.createUserList(1000);
      
      await tester.pumpWidget(
        MaterialApp(
          home: UserListPage(users: users),
        ),
      );
      
      // 测量渲染时间
      final stopwatch = Stopwatch()..start();
      await tester.pumpAndSettle();
      stopwatch.stop();
      
      expect(stopwatch.elapsedMilliseconds, lessThan(1000));
    });
  });
}
```

## 测试覆盖率标准

### 覆盖率要求
- **总体覆盖率**: ≥90%
- **业务逻辑**: 100%
- **UI组件**: ≥85%
- **工具函数**: 100%

### 排除文件
```dart
// coverage: ignore-file
class GeneratedCode {
  // 生成的代码可以忽略覆盖率
}

// coverage: ignore-start
void debugOnlyFunction() {
  // 仅调试用的代码
}
// coverage: ignore-end
```

## 测试报告

### 生成测试报告
```bash
# HTML格式覆盖率报告
flutter test --coverage
genhtml coverage/lcov.info -o coverage/html

# JUnit格式测试报告
flutter test --reporter=json > test_results.json
```

### CI集成
```yaml
- name: Test Report
  uses: dorny/test-reporter@v1
  if: success() || failure()
  with:
    name: Flutter Tests
    path: test_results.json
    reporter: dart-json
```