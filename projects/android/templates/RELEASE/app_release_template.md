# App Release Documentation

## Release Overview
- **App Name**: [App Name]
- **Version**: 1.0.0 (Build 1)
- **Release Date**: [YYYY-MM-DD]
- **Platform**: Android / iOS / Both
- **Release Type**: Alpha / Beta / Production
- **Target Audience**: Internal / TestFlight / Public

## Version Information

### Versioning Strategy
```
Major.Minor.Patch (Build)
1.0.0 (1)

Major: Breaking changes, major features
Minor: New features, minor improvements
Patch: Bug fixes, small updates
Build: Internal build number
```

### Current Version
```yaml
# pubspec.yaml
version: 1.0.0+1  # version+build
```

### Platform Versions
| Platform | Version Code | Version Name | Min SDK | Target SDK |
|----------|-------------|--------------|---------|------------|
| Android | 1 | 1.0.0 | 21 (5.0) | 33 (13) |
| iOS | 1 | 1.0.0 | 12.0 | 16.0 |

## Pre-Release Checklist

### Code Quality
- [ ] All features implemented and tested
- [ ] Code review completed
- [ ] No critical bugs in issue tracker
- [ ] Performance benchmarks met
- [ ] Memory leaks checked
- [ ] Security vulnerabilities scanned

### Testing
- [ ] Unit tests passing (>90% coverage)
- [ ] Widget tests passing
- [ ] Integration tests passing
- [ ] Manual testing on all target devices
- [ ] Accessibility testing completed
- [ ] Localization testing completed

### Documentation
- [ ] README updated
- [ ] API documentation complete
- [ ] User guide updated
- [ ] Release notes prepared
- [ ] Privacy policy updated
- [ ] Terms of service updated

## Build Configuration

### Android Build

#### Release Signing
```bash
# Generate keystore (first time only)
keytool -genkey -v -keystore android/app/release-key.keystore \
  -alias release-key-alias -keyalg RSA -keysize 2048 -validity 10000

# Configure key.properties
cat > android/key.properties << EOF
storePassword=your-store-password
keyPassword=your-key-password
keyAlias=release-key-alias
storeFile=release-key.keystore
EOF
```

#### Build Configuration
```gradle
// android/app/build.gradle
android {
    defaultConfig {
        applicationId "com.company.appname"
        minSdkVersion 21
        targetSdkVersion 33
        versionCode flutterVersionCode.toInteger()
        versionName flutterVersionName
        multiDexEnabled true
    }
    
    signingConfigs {
        release {
            keyAlias keystoreProperties['keyAlias']
            keyPassword keystoreProperties['keyPassword']
            storeFile keystoreProperties['storeFile'] ? file(keystoreProperties['storeFile']) : null
            storePassword keystoreProperties['storePassword']
        }
    }
    
    buildTypes {
        release {
            signingConfig signingConfigs.release
            minifyEnabled true
            shrinkResources true
            proguardFiles getDefaultProguardFile('proguard-android.txt'), 'proguard-rules.pro'
        }
    }
}
```

#### Build Commands
```bash
# Clean build
flutter clean

# Get dependencies
flutter pub get

# Build APK
flutter build apk --release

# Build App Bundle (recommended for Play Store)
flutter build appbundle --release

# Build with specific flavor
flutter build apk --release --flavor production

# Split APKs by ABI
flutter build apk --release --split-per-abi
```

### iOS Build

#### Certificates & Provisioning
```bash
# Open Xcode
open ios/Runner.xcworkspace

# Configure in Xcode:
# 1. Select Runner target
# 2. Signing & Capabilities tab
# 3. Select team
# 4. Choose provisioning profile
```

#### Build Configuration
```ruby
# ios/Runner/Info.plist
<key>CFBundleDisplayName</key>
<string>App Name</string>
<key>CFBundleIdentifier</key>
<string>com.company.appname</string>
<key>CFBundleShortVersionString</key>
<string>$(FLUTTER_BUILD_NAME)</string>
<key>CFBundleVersion</key>
<string>$(FLUTTER_BUILD_NUMBER)</string>
```

#### Build Commands
```bash
# Build iOS app
flutter build ios --release

# Build IPA for App Store
flutter build ipa --release

# Archive in Xcode
# 1. Product > Archive
# 2. Distribute App
# 3. App Store Connect
```

## Store Listings

### Google Play Store

#### Store Listing Information
```
Title: App Name (30 chars max)
Short Description: Brief app description (80 chars max)
Full Description: Detailed description (4000 chars max)

Categories:
- Primary: [Category]
- Secondary: [Category]

Content Rating: Everyone / Teen / Mature

Contact Information:
- Email: support@example.com
- Website: https://example.com
- Privacy Policy: https://example.com/privacy
```

#### Screenshots Requirements
- Phone: 2-8 screenshots (required)
- 7-inch tablet: 1-8 screenshots
- 10-inch tablet: 1-8 screenshots
- Dimensions: See Play Console guidelines
- Format: JPEG or PNG (no alpha)

#### Feature Graphic
- Dimensions: 1024x500px
- Format: JPEG or PNG (no alpha)

### App Store

#### App Information
```
App Name: App Name (30 chars max)
Subtitle: Brief tagline (30 chars max)
Primary Category: [Category]
Secondary Category: [Category]

Age Rating: 4+ / 9+ / 12+ / 17+

Keywords: keyword1, keyword2, ... (100 chars max)
Support URL: https://example.com/support
Marketing URL: https://example.com
```

#### Screenshots Requirements
- iPhone 6.7": 2-10 screenshots (required)
- iPhone 6.5": 2-10 screenshots
- iPhone 5.5": 2-10 screenshots
- iPad 12.9": 2-10 screenshots
- Format: PNG or JPEG

#### App Preview Video
- Duration: 15-30 seconds
- Format: See App Store Connect guidelines

## Release Process

### 1. Version Bump
```bash
# Update version in pubspec.yaml
# version: 1.0.1+2

# Commit version bump
git add pubspec.yaml
git commit -m "chore: bump version to 1.0.1+2"
git tag v1.0.1
git push origin main --tags
```

### 2. Build Release

#### Android
```bash
# Build App Bundle
flutter build appbundle --release

# Output: build/app/outputs/bundle/release/app-release.aab
```

#### iOS
```bash
# Build for iOS
flutter build ios --release

# Archive and upload via Xcode
open ios/Runner.xcworkspace
# Product > Archive > Distribute App
```

### 3. Internal Testing

#### Android (Internal Testing Track)
1. Upload AAB to Play Console
2. Create internal release
3. Add testers
4. Share testing link

#### iOS (TestFlight)
1. Upload build via Xcode/Transporter
2. Add build to TestFlight
3. Add internal testers
4. Submit for external testing

### 4. Beta Testing

#### Android (Open/Closed Beta)
```
1. Promote from internal to beta
2. Set rollout percentage (optional)
3. Monitor crash reports and feedback
4. Fix critical issues
```

#### iOS (TestFlight External)
```
1. Submit for beta app review
2. Add external testers
3. Monitor TestFlight feedback
4. Address reported issues
```

### 5. Production Release

#### Android
```bash
# Final checks
- Review crash reports
- Check performance metrics
- Verify all translations
- Test on various devices

# Release steps
1. Promote from beta to production
2. Set staged rollout (e.g., 10% → 50% → 100%)
3. Monitor metrics closely
4. Be ready to halt rollout if needed
```

#### iOS
```bash
# App Store submission
1. Submit for App Store review
2. Provide review notes if needed
3. Respond to review feedback
4. Schedule release (immediate/scheduled)
```

## Post-Release

### Monitoring

#### Crash Reporting
```dart
// Firebase Crashlytics setup
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp();
  
  FlutterError.onError = FirebaseCrashlytics.instance.recordFlutterError;
  
  runZonedGuarded(() {
    runApp(MyApp());
  }, FirebaseCrashlytics.instance.recordError);
}
```

#### Analytics
```dart
// Track key events
FirebaseAnalytics.instance.logEvent(
  name: 'app_open',
  parameters: {
    'version': '1.0.0',
    'platform': Platform.operatingSystem,
  },
);
```

#### Performance Monitoring
```dart
// Firebase Performance
final trace = FirebasePerformance.instance.newTrace('app_startup');
await trace.start();
// ... app initialization
await trace.stop();
```

### User Feedback

#### In-App Feedback
```dart
// Feedback dialog
void showFeedbackDialog(BuildContext context) {
  showDialog(
    context: context,
    builder: (context) => FeedbackDialog(
      onSubmit: (feedback) async {
        await sendFeedback(feedback);
      },
    ),
  );
}
```

#### Store Reviews
```dart
// Request store review
import 'package:in_app_review/in_app_review.dart';

final InAppReview inAppReview = InAppReview.instance;

if (await inAppReview.isAvailable()) {
  inAppReview.requestReview();
}
```

### Hotfixes

#### Emergency Fix Process
1. Identify critical issue
2. Create hotfix branch
3. Fix and test thoroughly
4. Build new version (increment patch)
5. Expedited review (if available)
6. Release immediately

#### Over-The-Air Updates (CodePush alternative)
```dart
// Using shorebird.dev for Flutter
shorebird patch android --release-version 1.0.0+1
shorebird patch ios --release-version 1.0.0+1
```

## Release Notes Template

### Version 1.0.0
**Release Date**: YYYY-MM-DD

**What's New**:
- 🎉 Initial release
- ✨ Feature 1: Description
- ✨ Feature 2: Description
- ✨ Feature 3: Description

**Improvements**:
- 🚀 Performance improvements
- 💄 UI enhancements
- 🌍 Added language support

**Bug Fixes**:
- 🐛 Fixed issue with...
- 🐛 Resolved crash when...

**Known Issues**:
- Issue 1: Workaround available
- Issue 2: Fix planned for next release

## Rollback Plan

### Android Rollback
```bash
# Halt staged rollout
# In Play Console: Release > Halt rollout

# Release previous version
# Upload previous AAB as new release
```

### iOS Rollback
```bash
# Cannot rollback on App Store
# Must submit new build with fixes

# Emergency: Remove from sale temporarily
# App Store Connect: Pricing and Availability
```

## Distribution Channels

### Alternative Stores
- Amazon Appstore
- Samsung Galaxy Store
- Huawei AppGallery
- F-Droid (open source)

### Enterprise Distribution
```bash
# iOS Enterprise
# Requires Apple Developer Enterprise Program

# Android Enterprise
# Use managed Google Play or private app hosting
```

### Direct Distribution
```bash
# Android APK hosting
# Host APK on website with proper MIME type

# iOS Ad Hoc
# Limited to 100 devices per year
```

## Legal Requirements

### Privacy Compliance
- [ ] GDPR compliance (EU)
- [ ] CCPA compliance (California)
- [ ] COPPA compliance (children's apps)
- [ ] App Tracking Transparency (iOS)
- [ ] Data safety section (Android)

### Export Compliance
- [ ] Encryption declaration
- [ ] Export control classification
- [ ] Country restrictions

### Third-Party Licenses
```bash
# Generate license file
flutter pub run flutter_oss_licenses:generate.dart
```

## Success Metrics

### Key Performance Indicators
- Downloads: Target vs Actual
- Active Users: DAU/MAU
- Retention: D1/D7/D30
- Crash Rate: <1%
- App Store Rating: >4.0
- User Reviews: Sentiment analysis

### A/B Testing
```dart
// Feature flags for gradual rollout
final bool useNewFeature = RemoteConfig.instance.getBool('use_new_feature');

if (useNewFeature) {
  return NewFeatureWidget();
} else {
  return OldFeatureWidget();
}
```