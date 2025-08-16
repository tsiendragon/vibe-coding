# Dart 代码规范 (Dart 3.0+)

## 专用标准
- **Flutter**: `android_project/standards/flutter_standards.md`
- **测试**: `android_project/standards/dart_test_standards.md`

---

## 核心原则 (MUST/MUST NOT)

| 主题 | MUST | MUST NOT |
|------|------|----------|
| **类型系统** | 使用严格空安全；显式类型注解公开API；使用sealed类和模式匹配 | 使用dynamic类型；忽略空安全警告 |
| **异常处理** | 自定义异常继承Exception；使用Either类型处理错误 | 使用字符串作为异常；吞噬异常不处理 |
| **异步编程** | 使用async/await；正确处理Future错误；使用Stream处理数据流 | 阻塞主线程；忘记await异步操作 |
| **内存管理** | 及时释放资源；使用dispose清理；避免内存泄漏 | 忘记取消订阅；持有强引用循环 |
| **状态管理** | 使用不可变状态；集中状态管理；避免全局状态 | 直接修改状态；在build方法中修改状态 |
| **性能** | 使用const构造函数；避免不必要的重建；使用ListView.builder | 在build方法中创建对象；使用Column显示长列表 |
| **安全** | 验证所有输入；使用HTTPS；避免代码注入 | 信任用户输入；硬编码敏感信息 |

### 额外禁止
- 在build方法中进行网络请求
- 使用print进行日志输出（使用logger包）
- 忽略Flutter/Dart分析器警告
- 直接在UI层访问数据库或网络

---

## 项目结构

```
lib/
├─ core/          # 核心功能：网络、存储、常量
├─ data/          # 数据层：模型、数据源、仓库实现
├─ domain/        # 领域层：实体、仓库接口、用例
├─ presentation/  # 表现层：页面、组件、状态管理
└─ shared/        # 共享：扩展、工具、验证器
```

**规则**: UI逻辑、网络请求、数据库操作严格分层，不得跨层直接调用。

---

## 类型系统 (Dart 3.0+)

```dart
// 严格空安全
String? nullableString;
String nonNullableString = 'Hello';

// late关键字用于延迟初始化
late final String configValue;

// sealed类用于状态管理
sealed class LoadingState {}
class Loading extends LoadingState {}
class Success extends LoadingState {
  final String data;
  Success(this.data);
}
class Error extends LoadingState {
  final String message;
  Error(this.message);
}

// 模式匹配
String handleState(LoadingState state) {
  return switch (state) {
    Loading() => 'Loading...',
    Success(data: final data) => 'Success: $data',
    Error(message: final msg) => 'Error: $msg',
  };
}

// Record类型
typedef UserInfo = ({String name, int age, String email});

UserInfo getUser() => (name: 'John', age: 30, email: 'john@example.com');
```

**要求**:
- 所有公开API必须有明确类型注解
- 启用strict-casts和strict-inference
- 使用late final而不是late var

---

## 错误处理

```dart
// 自定义异常层次
abstract class AppException implements Exception {
  final String message;
  const AppException(this.message);
}

class NetworkException extends AppException {
  const NetworkException(String message) : super(message);
}

class ValidationException extends AppException {
  const ValidationException(String message) : super(message);
}

class CacheException extends AppException {
  const CacheException(String message) : super(message);
}

// Either类型用于错误处理
import 'package:dartz/dartz.dart';

typedef Result<T> = Either<AppException, T>;

class UserRepository {
  Future<Result<User>> getUser(String id) async {
    try {
      final user = await _apiService.fetchUser(id);
      return Right(user);
    } on NetworkException catch (e) {
      return Left(NetworkException('Failed to fetch user: ${e.message}'));
    } catch (e) {
      return Left(NetworkException('Unexpected error: $e'));
    }
  }
}
```

**规则**:
- 所有可能失败的操作返回Either类型
- 异常应该语义明确且可恢复
- 在数据层转换所有外部异常

---

## 异步编程

```dart
// 正确的异步模式
class DataService {
  final List<StreamSubscription> _subscriptions = [];
  
  // 使用async/await
  Future<List<User>> fetchUsers() async {
    try {
      final response = await _httpClient.get('/users');
      return response.data.map<User>((json) => User.fromJson(json)).toList();
    } on DioException catch (e) {
      throw NetworkException('Failed to fetch users: ${e.message}');
    }
  }
  
  // Stream处理
  Stream<List<User>> watchUsers() {
    return _firestore
        .collection('users')
        .snapshots()
        .map((snapshot) => snapshot.docs
            .map((doc) => User.fromJson(doc.data()))
            .toList());
  }
  
  // 资源清理
  void dispose() {
    for (final subscription in _subscriptions) {
      subscription.cancel();
    }
    _subscriptions.clear();
  }
}

// 并发处理
Future<UserProfile> fetchUserProfile(String userId) async {
  final (user, posts, followers) = await (
    fetchUser(userId),
    fetchUserPosts(userId),
    fetchUserFollowers(userId),
  ).wait;
  
  return UserProfile(
    user: user,
    posts: posts,
    followersCount: followers.length,
  );
}
```

**规则**:
- 所有异步操作必须正确处理错误
- 长时间运行的Stream必须能够取消
- 使用Future.wait进行并发操作

---

## 状态管理

```dart
// 不可变状态类
@freezed
class AppState with _$AppState {
  const factory AppState({
    @Default(false) bool isLoading,
    @Default([]) List<User> users,
    String? error,
  }) = _AppState;
}

// Riverpod StateNotifier
class AppNotifier extends StateNotifier<AppState> {
  AppNotifier(this._userRepository) : super(const AppState());
  
  final UserRepository _userRepository;
  
  Future<void> loadUsers() async {
    state = state.copyWith(isLoading: true, error: null);
    
    final result = await _userRepository.getUsers();
    result.fold(
      (error) => state = state.copyWith(
        isLoading: false,
        error: error.message,
      ),
      (users) => state = state.copyWith(
        isLoading: false,
        users: users,
      ),
    );
  }
}

// Provider定义
final appProvider = StateNotifierProvider<AppNotifier, AppState>((ref) {
  return AppNotifier(ref.read(userRepositoryProvider));
});
```

**规则**:
- 状态类必须是不可变的
- 使用copyWith方法更新状态
- 状态变更只能通过Notifier进行

---

## 性能优化

```dart
// 使用const构造函数
class CustomButton extends StatelessWidget {
  const CustomButton({
    Key? key,
    required this.text,
    required this.onPressed,
  }) : super(key: key);
  
  final String text;
  final VoidCallback onPressed;
  
  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: onPressed,
      child: Text(text),
    );
  }
}

// 避免在build方法中创建对象
class BadWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    // ❌ 错误：每次build都会创建新对象
    final style = TextStyle(color: Colors.red);
    
    return Text('Hello', style: style);
  }
}

class GoodWidget extends StatelessWidget {
  // ✅ 正确：静态常量
  static const TextStyle _style = TextStyle(color: Colors.red);
  
  @override
  Widget build(BuildContext context) {
    return const Text('Hello', style: _style);
  }
}

// 使用ListView.builder处理长列表
Widget buildList(List<Item> items) {
  return ListView.builder(
    itemCount: items.length,
    itemBuilder: (context, index) {
      return ItemWidget(item: items[index]);
    },
  );
}
```

---

## 测试策略

```dart
// 单元测试
group('UserRepository', () {
  late UserRepository repository;
  late MockApiService mockApiService;
  
  setUp(() {
    mockApiService = MockApiService();
    repository = UserRepositoryImpl(mockApiService);
  });
  
  testWidgets('should return Right when API call succeeds', () async {
    // Arrange
    const user = User(id: '1', name: 'John');
    when(() => mockApiService.fetchUser('1'))
        .thenAnswer((_) async => user);
    
    // Act
    final result = await repository.getUser('1');
    
    // Assert
    expect(result, isA<Right<AppException, User>>());
    result.fold(
      (l) => fail('Should not return error'),
      (r) => expect(r, equals(user)),
    );
  });
});

// Widget测试
testWidgets('should display loading indicator when loading', (tester) async {
  await tester.pumpWidget(
    ProviderScope(
      overrides: [
        appProvider.overrideWith((ref) {
          final notifier = MockAppNotifier();
          when(() => notifier.state).thenReturn(
            const AppState(isLoading: true),
          );
          return notifier;
        }),
      ],
      child: const MaterialApp(home: UserListPage()),
    ),
  );
  
  expect(find.byType(CircularProgressIndicator), findsOneWidget);
});
```

---

## 工具配置

**analysis_options.yaml**
```yaml
include: package:very_good_analysis/analysis_options.yaml

analyzer:
  language:
    strict-casts: true
    strict-inference: true
    strict-raw-types: true
  exclude:
    - "**/*.g.dart"
    - "**/*.freezed.dart"
    - "build/**"
  errors:
    invalid_annotation_target: ignore

linter:
  rules:
    # 性能相关
    prefer_const_constructors: true
    prefer_const_declarations: true
    prefer_const_literals_to_create_immutables: true
    
    # 代码质量
    prefer_final_locals: true
    prefer_final_in_for_each: true
    require_trailing_commas: true
    
    # 命名约定
    file_names: true
    non_constant_identifier_names: true
    
    # 类型安全
    avoid_dynamic_calls: true
    avoid_type_to_string: true
```

**pubspec.yaml依赖管理**
```yaml
dependencies:
  # 核心
  flutter:
    sdk: flutter
  
  # 状态管理
  flutter_riverpod: ^2.4.0
  
  # 功能性编程
  dartz: ^0.10.1
  freezed_annotation: ^2.4.1
  
  # 网络
  dio: ^5.3.0
  
  # 依赖注入
  get_it: ^7.6.0

dev_dependencies:
  # 测试
  flutter_test:
    sdk: flutter
  mocktail: ^1.0.0
  
  # 代码生成
  build_runner: ^2.4.6
  freezed: ^2.4.6
  json_serializable: ^6.7.1
  
  # 代码质量
  very_good_analysis: ^5.1.0
```

---

## 代码模板

```dart
// 数据模型
@freezed
class User with _$User {
  const factory User({
    required String id,
    required String name,
    required String email,
    DateTime? createdAt,
  }) = _User;
  
  factory User.fromJson(Map<String, dynamic> json) => _$UserFromJson(json);
}

// 仓库接口
abstract class UserRepository {
  Future<Result<User>> getUser(String id);
  Future<Result<List<User>>> getUsers();
  Future<Result<void>> updateUser(User user);
}

// 仓库实现
class UserRepositoryImpl implements UserRepository {
  const UserRepositoryImpl(this._apiService, this._cacheService);
  
  final ApiService _apiService;
  final CacheService _cacheService;
  
  @override
  Future<Result<User>> getUser(String id) async {
    try {
      // 先尝试缓存
      final cachedUser = await _cacheService.getUser(id);
      if (cachedUser != null) {
        return Right(cachedUser);
      }
      
      // 从API获取
      final user = await _apiService.fetchUser(id);
      await _cacheService.saveUser(user);
      
      return Right(user);
    } on NetworkException catch (e) {
      return Left(NetworkException('Failed to get user: ${e.message}'));
    } catch (e) {
      return Left(NetworkException('Unexpected error: $e'));
    }
  }
}

// 用例
class GetUserUseCase {
  const GetUserUseCase(this._repository);
  
  final UserRepository _repository;
  
  Future<Result<User>> call(String id) async {
    if (id.isEmpty) {
      return const Left(ValidationException('User ID cannot be empty'));
    }
    
    return _repository.getUser(id);
  }
}
```