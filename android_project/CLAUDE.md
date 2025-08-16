# Flutter Mobile Application Project

This is a Flutter-based mobile application project using an AI-assisted development workflow with specialized agents.

## 🚀 Project Start Guide

### Step 1: Understand Your Mobile App Requirements
**Before starting any development, Claude Code needs to understand your mobile app requirements.**

**Where are your requirements?**
- **Option A**: You have app specifications - Please provide the document, wireframes, or feature requirements
- **Option B**: You want to discuss your app idea - We'll clarify your features, user flows, and UI/UX needs
- **Option C**: You have an existing app to migrate/improve - Please provide current app details or screenshots
- **Option D**: You have a specific use case - Describe your mobile application needs and target users

**Important**: The Mobile Product Manager agent will work with you to ensure 100% requirement clarity before any development begins.

### Step 2: AI Agent Workflow Overview
This project uses a sophisticated multi-agent collaboration system where specialized AI agents work together:

1. **🎯 Mobile Product Manager** - Clarifies app requirements and writes specifications
2. **🏗️ Tech Lead** - Designs mobile architecture and coordinates development
3. **🎨 UI/UX Designer** - Designs user interface and user experience
4. **📱 Flutter Developer** - Implements app with production-grade Flutter code
5. **🔄 State Manager** - Designs state management architecture and data flow
6. **🧪 Mobile Tester** - Creates UI tests and validates functionality across devices
7. **🚀 Release Manager** - Handles app store deployment and versioning
8. **🔍 Code Reviewer** - Ensures code quality and Flutter best practices

## 📋 Essential Documents to Review

Before starting development, **Claude Code must thoroughly understand these key documents**:

### Workflow and Process
- **`docs/workflows/mobile_app_workflow.md`** - Complete mobile app development workflow with UI/UX patterns
  - *Purpose*: Understand mobile development lifecycle, state management, and app store deployment
  - *Why Critical*: Defines the entire development process from UI design to app store publication

### Agent Responsibilities
- **`.claude/agents/`** - Individual agent roles and responsibilities for mobile development
  - *Purpose*: Each agent specializes in different aspects of Flutter mobile development
  - *Why Critical*: Ensures proper task assignment and collaboration for complex mobile applications

### Development Standards
- **`docs/standards/dart_standards.md`** - Dart coding standards and best practices
- **`docs/standards/flutter_standards.md`** - Flutter-specific development practices and patterns
- **`docs/standards/dart_test_standards.md`** - Flutter testing standards (widget, integration tests)
- **`docs/standards/git_commit_std.md`** - Git commit format and practices
  - *Purpose*: Maintain consistent, high-quality Flutter code
  - *Why Critical*: Ensures maintainable, performant, and user-friendly mobile applications

### Document Templates
- **`docs/templates/UI_DESIGN/`** - UI design specifications and component documentation
- **`docs/templates/APP_ARCHITECTURE/`** - Mobile architecture and state management patterns
- **`docs/templates/USER_FLOW/`** - User journey and interaction flow templates
- **`docs/templates/RELEASE/`** - App store deployment and release management templates
  - *Purpose*: Standardized formats for all mobile project documentation
  - *Why Critical*: Ensures consistent UI design and app architecture documentation

### Knowledge Management
- **`docs/knowledge/best_practices/`** - Flutter patterns, state management, performance optimization
- **`docs/knowledge/error_cases/`** - Common mobile issues, platform-specific problems, and solutions
  - *Purpose*: Learn from past mobile development experience
  - *Why Critical*: Accelerates development and improves app quality and user experience

## 🎯 Development Commands

### Flutter Development
```bash
# Run development build
flutter run

# Run on specific device
flutter devices                    # List available devices
flutter run -d <device_id>        # Run on specific device

# Hot reload and hot restart during development
# Press 'r' for hot reload, 'R' for hot restart in terminal
```

### Building and Testing
```bash
# Build applications
flutter build apk                  # Build Android APK
flutter build ios                  # Build iOS (macOS only)
flutter build appbundle           # Build Android App Bundle for Play Store

# Run tests
flutter test                       # Run all tests
flutter test test/unit/           # Run unit tests only
flutter test test/widget/         # Run widget tests only
flutter test --coverage          # Run tests with coverage report
```

### Code Generation and Quality
```bash
# Code generation (for state management, JSON serialization)
flutter packages pub run build_runner build --delete-conflicting-outputs

# Code quality checks
dart format .                     # Format Dart code
dart analyze                      # Static analysis
dart fix --apply                  # Apply suggested fixes
```

### Dependencies and Packages
```bash
# Package management
flutter pub get                   # Install dependencies
flutter pub upgrade              # Upgrade packages
flutter pub deps                 # Show dependency tree
flutter pub outdated             # Check for outdated packages
```

## 🔄 AI Development Workflow

### Phase 1: App Requirements & UI Design (Mobile Product Manager + UI/UX Designer)
1. **Mobile requirement confirmation** - Iterative clarification of features and user flows
2. **UI/UX design** - Wireframes, mockups, design system, and user experience patterns
3. **Platform strategy** - iOS/Android considerations, device compatibility

### Phase 2: Architecture & State Management (Tech Lead + State Manager)
1. **Mobile architecture design** - Clean architecture, folder structure, dependency injection
2. **State management strategy** - Riverpod/Bloc/Provider selection and implementation
3. **Data flow design** - API integration, local storage, caching strategies

### Phase 3: UI Implementation & Components (Flutter Developer + UI/UX Designer)
1. **UI component library** - Reusable widgets, theming, responsive design
2. **Screen implementation** - Page layouts, navigation, animations
3. **Platform integration** - Native features, permissions, device capabilities

### Phase 4: State Management & Business Logic (Flutter Developer + State Manager)
1. **State management implementation** - Provider/Riverpod/Bloc pattern implementation
2. **Business logic integration** - API calls, data processing, validation
3. **Local storage** - SharedPreferences, Hive, SQLite integration

### Phase 5: Testing & Quality Assurance (Mobile Tester + Code Reviewer)
1. **Widget testing** - UI component tests, golden tests for visual regression
2. **Integration testing** - End-to-end user flow validation
3. **Device compatibility** - Testing across different screen sizes and platforms

### Phase 6: Release & Deployment (Release Manager + DevOps)
1. **App store preparation** - Icons, screenshots, store listings, metadata
2. **Build optimization** - Code splitting, asset optimization, performance tuning
3. **Release management** - Version control, staged rollouts, crash monitoring

## ⚠️ Critical Guidelines

### For Claude Code
1. **NEVER start coding without clear UI/UX requirements** - Always engage Mobile Product Manager first
2. **Follow Material Design or Cupertino patterns** - Use platform-appropriate design languages
3. **State management is crucial** - Choose and implement proper state management from the start
4. **Performance matters** - Consider app size, memory usage, and battery consumption
5. **Test on real devices** - Emulators don't capture all real-world scenarios

### Development Standards
- ✅ **Responsive design** - Support different screen sizes and orientations
- ✅ **Accessibility** - Include semantic labels, contrast ratios, navigation support
- ✅ **Error handling** - Graceful degradation and user-friendly error messages
- ✅ **Offline support** - Cache critical data and handle network issues
- ✅ **85% widget test coverage minimum** - Critical user flows must be tested

## 📱 Flutter Project Structure
```
lib/
├── main.dart              # Application entry
├── app/                   # App configuration and routes
├── core/                  # Constants, errors, network, storage
├── data/                  # Data sources, models, repositories
├── domain/                # Entities, repositories, use cases
├── presentation/          # Pages, widgets, providers
└── shared/                # Extensions, mixins, validators
```

## 🎨 UI/UX Considerations

### Design System
- **Material Design** for Android-first apps
- **Cupertino** for iOS-first apps  
- **Custom Design System** for brand-specific UI
- **Responsive layouts** for tablets and different screen sizes

### User Experience
- **Intuitive navigation** - Clear app structure and user flows
- **Fast performance** - Smooth animations and quick loading
- **Offline functionality** - Work without internet when possible
- **Platform conventions** - Follow iOS/Android platform guidelines

## 🤝 Getting Started

1. **Tell us about your mobile app**: What kind of mobile application do you want to build?
2. **Provide requirements**: Share your app specifications, wireframes, feature list, or let's discuss
3. **Agent activation**: The Mobile Product Manager will lead requirement clarification
4. **Follow the workflow**: Each agent will contribute their expertise in sequence
5. **Quality delivery**: Receive a production-ready Flutter app ready for app store deployment

**Ready to start? Please provide your mobile app requirements or let's discuss what application you want to build!**