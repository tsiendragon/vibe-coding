# /deploy 部署命令

标准的项目部署命令，支持多环境部署和全面的验证检查。

## 语法

```bash
/deploy <target_env> <deployment_config> [options]
```

## 参数

- `target_env`: 目标环境 (dev/staging/production)
- `deployment_config`: 部署配置文件路径
- `options`: 可选参数

## 选项

- `--timeout=30m`: 部署超时时间 (默认30分钟)
- `--validation-level=comprehensive`: 验证级别 (basic/normal/comprehensive)
- `--rollback-on-failure`: 失败时自动回滚
- `--notify-slack`: 部署完成后发送Slack通知
- `--pre-deploy-tests`: 部署前运行完整测试套件
- `--post-deploy-validation`: 部署后运行验证检查

## 部署流程

1. **预检查** - 验证部署配置和环境状态
2. **备份** - 备份当前版本以便回滚
3. **构建** - 构建应用程序和资源
4. **测试** - 运行部署前测试
5. **部署** - 执行实际部署操作
6. **验证** - 部署后健康检查和功能验证
7. **通知** - 发送部署结果通知

## 安全检查

- **访问权限验证** - 确认操作者有目标环境的部署权限
- **配置安全扫描** - 检查部署配置中的敏感信息
- **依赖安全检查** - 扫描依赖包的安全漏洞
- **环境隔离验证** - 确保部署不会影响其他环境

## 示例

```bash
# 部署到开发环境
/deploy dev ./config/dev-deploy.yaml --timeout=15m

# 部署到生产环境（包含全面验证）
/deploy production ./config/prod-deploy.yaml --validation-level=comprehensive --rollback-on-failure --notify-slack

# 快速部署到测试环境
/deploy staging ./config/test-deploy.yaml --validation-level=basic --timeout=10m
```

## 配置文件格式

```yaml
# deployment_config.yaml
target:
  platform: "kubernetes"
  cluster: "prod-cluster"
  namespace: "my-app"

image:
  registry: "my-registry.com"
  repository: "my-app"
  tag: "v1.2.3"

replicas: 3
resources:
  cpu: "500m"
  memory: "1Gi"

environment:
  - name: "DATABASE_URL"
    value: "postgresql://..."
  - name: "API_KEY"
    valueFrom:
      secretKeyRef:
        name: "api-secrets"
        key: "api-key"

healthCheck:
  path: "/health"
  port: 8080
  initialDelay: 30
  timeout: 10
```

## 错误处理

- **部署失败自动回滚** - 检测到关键错误时自动回滚到上一个稳定版本
- **详细错误日志** - 提供详细的错误信息和调试建议
- **人工介入点** - 在关键步骤提供人工确认选项
- **故障恢复指南** - 提供常见问题的解决方案

## 权限要求

- 目标环境的部署权限
- 容器注册表的推送权限
- 配置和密钥的读取权限
- 监控和日志系统的写入权限
