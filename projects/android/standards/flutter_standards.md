# Flutter Standards

## 目录规范

```bash
your_project/
├─ pubspec.yaml
├─ analysis_options.yaml
├─ README.md
├─ android/                  # Android原生配置
├─ ios/                      # iOS原生配置
├─ web/                      # Web平台配置
├─ lib/
│  ├─ main.dart             # 应用入口
│  ├─ app/
│  │  ├─ app.dart           # 应用配置
│  │  ├─ routes/            # 路由配置
│  │  └─ theme/             # 主题配置
│  ├─ core/
│  │  ├─ constants/         # 常量定义
│  │  ├─ errors/            # 错误处理
│  │  ├─ network/           # 网络层
│  │  ├─ storage/           # 本地存储
│  │  └─ utils/             # 工具函数
│  ├─ data/
│  │  ├─ datasources/       # 数据源
│  │  │  ├─ local/          # 本地数据源
│  │  │  └─ remote/         # 远程数据源
│  │  ├─ models/            # 数据模型
│  │  └─ repositories/      # 数据仓库实现
│  ├─ domain/
│  │  ├─ entities/          # 业务实体
│  │  ├─ repositories/      # 数据仓库接口
│  │  └─ usecases/          # 用例
│  ├─ presentation/
│  │  ├─ pages/             # 页面
│  │  ├─ widgets/           # 通用组件
│  │  ├─ providers/         # 状态管理
│  │  └─ dialogs/           # 对话框
│  └─ shared/
│     ├─ extensions/        # 扩展方法
│     ├─ mixins/            # 混入
│     └─ validators/        # 验证器
├─ test/
│  ├─ unit/
│  │  ├─ core/
│  │  ├─ data/
│  │  ├─ domain/
│  │  └─ presentation/
│  ├─ widget/
│  │  └─ presentation/
│  └─ integration/
│     └─ app/
├─ test_driver/             # 集成测试
└─ assets/
   ├─ images/
   ├─ icons/
   └─ fonts/
```

## 架构模式 - Clean Architecture

### 分层架构
- **Presentation层**: UI和状态管理
- **Domain层**: 业务逻辑和实体
- **Data层**: 数据访问和外部API
- **Core层**: 共享功能和配置

### 依赖注入
```dart
// 使用get_it进行依赖注入
final GetIt locator = GetIt.instance;

void setupLocator() {
  // 核心服务
  locator.registerLazySingleton<NetworkService>(() => NetworkService());
  locator.registerLazySingleton<StorageService>(() => StorageService());
  
  // 数据源
  locator.registerLazySingleton<UserRemoteDataSource>(
    () => UserRemoteDataSourceImpl(locator<NetworkService>()),
  );
  
  // 仓库
  locator.registerLazySingleton<UserRepository>(
    () => UserRepositoryImpl(locator<UserRemoteDataSource>()),
  );
  
  // 用例
  locator.registerLazySingleton<GetUserUseCase>(
    () => GetUserUseCase(locator<UserRepository>()),
  );
}
```

## 状态管理

### 推荐方案
1. **Riverpod** - 首选（类型安全、测试友好）
2. **Bloc** - 复杂状态管理
3. **Provider** - 简单状态管理

### Riverpod示例
```dart
// Provider定义
final userProvider = StateNotifierProvider<UserNotifier, AsyncValue<User>>((ref) {
  return UserNotifier(ref.read(getUserUseCaseProvider));
});

// StateNotifier
class UserNotifier extends StateNotifier<AsyncValue<User>> {
  UserNotifier(this._getUserUseCase) : super(const AsyncValue.loading());
  
  final GetUserUseCase _getUserUseCase;
  
  Future<void> getUser(String id) async {
    state = const AsyncValue.loading();
    
    final result = await _getUserUseCase(id);
    result.fold(
      (failure) => state = AsyncValue.error(failure, StackTrace.current),
      (user) => state = AsyncValue.data(user),
    );
  }
}

// UI使用
class UserPage extends ConsumerWidget {
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final userAsync = ref.watch(userProvider);
    
    return userAsync.when(
      data: (user) => UserView(user: user),
      loading: () => const CircularProgressIndicator(),
      error: (error, _) => Text('Error: $error'),
    );
  }
}
```

## 错误处理

### 统一错误类型
```dart
abstract class Failure {
  const Failure();
}

class NetworkFailure extends Failure {
  final String message;
  const NetworkFailure(this.message);
}

class CacheFailure extends Failure {
  final String message;
  const CacheFailure(this.message);
}

class ValidationFailure extends Failure {
  final String message;
  const ValidationFailure(this.message);
}
```

### Either类型处理
```dart
// 使用dartz包的Either类型
typedef ResultFuture<T> = Future<Either<Failure, T>>;
typedef ResultVoid = Future<Either<Failure, void>>;

class UserRepositoryImpl implements UserRepository {
  @override
  ResultFuture<User> getUser(String id) async {
    try {
      final user = await _remoteDataSource.getUser(id);
      return Right(user);
    } on NetworkException catch (e) {
      return Left(NetworkFailure(e.message));
    } catch (e) {
      return Left(NetworkFailure('Unexpected error occurred'));
    }
  }
}
```

## 数据模型

### Model vs Entity
```dart
// Entity (Domain层)
class User {
  final String id;
  final String name;
  final String email;
  
  const User({
    required this.id,
    required this.name,
    required this.email,
  });
}

// Model (Data层)
class UserModel extends User {
  const UserModel({
    required String id,
    required String name,
    required String email,
  }) : super(id: id, name: name, email: email);
  
  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      id: json['id'],
      name: json['name'],
      email: json['email'],
    );
  }
  
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'email': email,
    };
  }
}
```

## 网络层

### HTTP Client配置
```dart
class NetworkService {
  late final Dio _dio;
  
  NetworkService() {
    _dio = Dio(BaseOptions(
      baseUrl: 'https://api.example.com',
      connectTimeout: const Duration(seconds: 5),
      receiveTimeout: const Duration(seconds: 3),
      headers: {
        'Content-Type': 'application/json',
      },
    ));
    
    _setupInterceptors();
  }
  
  void _setupInterceptors() {
    _dio.interceptors.addAll([
      LogInterceptor(requestBody: true, responseBody: true),
      RetryInterceptor(
        dio: _dio,
        options: const RetryOptions(retries: 3),
      ),
    ]);
  }
}
```

## 本地存储

### SharedPreferences封装
```dart
abstract class StorageService {
  Future<void> setString(String key, String value);
  Future<String?> getString(String key);
  Future<void> setBool(String key, bool value);
  Future<bool?> getBool(String key);
  Future<void> remove(String key);
  Future<void> clear();
}

class StorageServiceImpl implements StorageService {
  late final SharedPreferences _prefs;
  
  static Future<StorageServiceImpl> init() async {
    final service = StorageServiceImpl._();
    service._prefs = await SharedPreferences.getInstance();
    return service;
  }
  
  StorageServiceImpl._();
  
  @override
  Future<void> setString(String key, String value) async {
    await _prefs.setString(key, value);
  }
  
  @override
  Future<String?> getString(String key) async {
    return _prefs.getString(key);
  }
}
```

## 测试策略

### Unit Tests
```dart
group('UserRepository', () {
  late UserRepository repository;
  late MockUserRemoteDataSource mockRemoteDataSource;
  
  setUp(() {
    mockRemoteDataSource = MockUserRemoteDataSource();
    repository = UserRepositoryImpl(mockRemoteDataSource);
  });
  
  test('should return User when call to remote data source is successful', () async {
    // arrange
    const testUser = UserModel(id: '1', name: 'Test', email: 'test@example.com');
    when(() => mockRemoteDataSource.getUser(any()))
        .thenAnswer((_) async => testUser);
    
    // act
    final result = await repository.getUser('1');
    
    // assert
    expect(result, equals(const Right(testUser)));
    verify(() => mockRemoteDataSource.getUser('1')).called(1);
  });
});
```

### Widget Tests
```dart
testWidgets('should display loading indicator when state is loading', (tester) async {
  // arrange
  await tester.pumpWidget(
    ProviderScope(
      overrides: [
        userProvider.overrideWith((ref) => UserNotifier.loading()),
      ],
      child: const MaterialApp(home: UserPage()),
    ),
  );
  
  // assert
  expect(find.byType(CircularProgressIndicator), findsOneWidget);
});
```

## pubspec.yaml配置

```yaml
name: your_app
description: A Flutter application following clean architecture
version: 1.0.0+1

environment:
  sdk: '>=3.0.0 <4.0.0'
  flutter: ">=3.10.0"

dependencies:
  flutter:
    sdk: flutter
  
  # 状态管理
  flutter_riverpod: ^2.4.0
  
  # 网络
  dio: ^5.3.0
  retrofit: ^4.0.0
  
  # 本地存储
  shared_preferences: ^2.2.0
  hive: ^2.2.3
  
  # 依赖注入
  get_it: ^7.6.0
  
  # 功能性编程
  dartz: ^0.10.1
  
  # JSON序列化
  json_annotation: ^4.8.1
  
  # 路由
  go_router: ^10.0.0
  
  # UI
  flutter_svg: ^2.0.7
  cached_network_image: ^3.2.3

dev_dependencies:
  flutter_test:
    sdk: flutter
  
  # 测试
  mocktail: ^1.0.0
  
  # 代码生成
  build_runner: ^2.4.6
  json_serializable: ^6.7.1
  retrofit_generator: ^7.0.0
  
  # 代码质量
  flutter_lints: ^2.0.3
  very_good_analysis: ^5.1.0

flutter:
  uses-material-design: true
  assets:
    - assets/images/
    - assets/icons/
  fonts:
    - family: CustomFont
      fonts:
        - asset: assets/fonts/CustomFont-Regular.ttf
```

## 代码质量

### analysis_options.yaml
```yaml
include: package:very_good_analysis/analysis_options.yaml

analyzer:
  exclude:
    - "**/*.g.dart"
    - "**/*.freezed.dart"
    - "build/**"

linter:
  rules:
    # 自定义规则
    prefer_single_quotes: true
    require_trailing_commas: true
    sort_pub_dependencies: true
```

### 命名约定
- **文件名**: snake_case
- **类名**: PascalCase
- **变量/函数**: camelCase
- **常量**: SCREAMING_SNAKE_CASE
- **私有成员**: 前缀下划线

### MUST / MUST NOT

* **MUST** 使用const构造函数（性能优化）
* **MUST** 实现hashCode和==操作符
* **MUST** 使用命名参数（required关键字）
* **MUST** 处理所有异步操作的错误
* **MUST** 遵循Clean Architecture分层
* **MUST NOT** 在build方法中进行昂贵操作
* **MUST NOT** 直接在UI层访问数据源
* **MUST NOT** 使用全局变量存储状态
* **MUST NOT** 忽略Future（使用unawaited）

## 性能优化

### Widget优化
```dart
// 使用const构造函数
const Text('Hello World')

// 避免匿名函数重建
class MyWidget extends StatelessWidget {
  void _onPressed() {
    // 处理点击
  }
  
  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      onPressed: _onPressed, // 避免 () => _onPressed()
      child: const Text('Press me'),
    );
  }
}

// 使用Builder减少重建范围
class MyWidget extends StatefulWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const ExpensiveWidget(), // 不会重建
        Builder(
          builder: (context) => CheapWidget(data: _data), // 只有这部分重建
        ),
      ],
    );
  }
}
```

## 国际化

### 配置
```yaml
dependencies:
  flutter_localizations:
    sdk: flutter
  intl: any

flutter:
  generate: true
```

### 使用
```dart
// l10n.yaml
arb-dir: lib/l10n
template-arb-file: app_en.arb
output-localization-file: app_localizations.dart

// lib/l10n/app_en.arb
{
  "hello": "Hello",
  "welcome": "Welcome {name}",
  "@welcome": {
    "placeholders": {
      "name": {
        "type": "String"
      }
    }
  }
}

// 使用
Text(AppLocalizations.of(context)!.hello)
```