# UI Design Specification

## Design Overview
- **App Name**: [App Name]
- **Platform**: Android / iOS / Both
- **Design System**: Material Design 3 / Cupertino
- **Target Devices**: Phone / Tablet / Both
- **Minimum Screen Size**: 360x640dp
- **Theme**: Light / Dark / System

## Design Principles

### Visual Hierarchy
1. **Primary Actions**: Prominent buttons with brand colors
2. **Secondary Actions**: Text buttons or outlined buttons
3. **Information Architecture**: Clear navigation and content grouping
4. **White Space**: Adequate padding and margins for readability

### Interaction Patterns
- **Navigation**: Bottom navigation / Drawer / Tab bar
- **Gestures**: Swipe, pinch, long press
- **Feedback**: Loading states, error states, success confirmations
- **Animations**: Smooth transitions, micro-interactions

## Color Palette

### Light Theme
```dart
class AppColors {
  static const Color primary = Color(0xFF1976D2);
  static const Color primaryVariant = Color(0xFF004BA0);
  static const Color secondary = Color(0xFFFF5722);
  static const Color background = Color(0xFFF5F5F5);
  static const Color surface = Color(0xFFFFFFFF);
  static const Color error = Color(0xFFB00020);
  static const Color onPrimary = Color(0xFFFFFFFF);
  static const Color onSecondary = Color(0xFFFFFFFF);
  static const Color onBackground = Color(0xFF212121);
  static const Color onSurface = Color(0xFF212121);
}
```

### Dark Theme
```dart
class AppColorsDark {
  static const Color primary = Color(0xFF90CAF9);
  static const Color primaryVariant = Color(0xFF42A5F5);
  static const Color secondary = Color(0xFFFF7043);
  static const Color background = Color(0xFF121212);
  static const Color surface = Color(0xFF1E1E1E);
  static const Color error = Color(0xFFCF6679);
  static const Color onPrimary = Color(0xFF000000);
  static const Color onSecondary = Color(0xFF000000);
  static const Color onBackground = Color(0xFFE0E0E0);
  static const Color onSurface = Color(0xFFE0E0E0);
}
```

## Typography

### Text Styles
```dart
class AppTextStyles {
  static const TextStyle headline1 = TextStyle(
    fontSize: 96,
    fontWeight: FontWeight.w300,
    letterSpacing: -1.5,
  );
  
  static const TextStyle headline2 = TextStyle(
    fontSize: 60,
    fontWeight: FontWeight.w300,
    letterSpacing: -0.5,
  );
  
  static const TextStyle headline3 = TextStyle(
    fontSize: 48,
    fontWeight: FontWeight.w400,
  );
  
  static const TextStyle headline4 = TextStyle(
    fontSize: 34,
    fontWeight: FontWeight.w400,
    letterSpacing: 0.25,
  );
  
  static const TextStyle headline5 = TextStyle(
    fontSize: 24,
    fontWeight: FontWeight.w400,
  );
  
  static const TextStyle headline6 = TextStyle(
    fontSize: 20,
    fontWeight: FontWeight.w500,
    letterSpacing: 0.15,
  );
  
  static const TextStyle bodyText1 = TextStyle(
    fontSize: 16,
    fontWeight: FontWeight.w400,
    letterSpacing: 0.5,
  );
  
  static const TextStyle bodyText2 = TextStyle(
    fontSize: 14,
    fontWeight: FontWeight.w400,
    letterSpacing: 0.25,
  );
  
  static const TextStyle button = TextStyle(
    fontSize: 14,
    fontWeight: FontWeight.w500,
    letterSpacing: 1.25,
  );
  
  static const TextStyle caption = TextStyle(
    fontSize: 12,
    fontWeight: FontWeight.w400,
    letterSpacing: 0.4,
  );
}
```

## Screen Specifications

### Splash Screen
**Purpose**: Brand presentation and app initialization

**Layout**:
```
┌─────────────────────────┐
│                         │
│                         │
│        [Logo]           │
│                         │
│     App Name            │
│                         │
│    [Progress Bar]       │
│                         │
└─────────────────────────┘
```

**Specifications**:
- Logo: 120x120dp, centered
- App name: headline5, below logo with 24dp spacing
- Progress bar: 200dp width, 4dp height
- Background: primary color or gradient
- Duration: 2-3 seconds maximum

### Login Screen
**Purpose**: User authentication

**Layout**:
```
┌─────────────────────────┐
│ ←                       │
├─────────────────────────┤
│                         │
│        [Logo]           │
│      Welcome Back       │
│                         │
│  ┌───────────────────┐  │
│  │ Email             │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │
│  │ Password          │  │
│  └───────────────────┘  │
│                         │
│  [ ] Remember me        │
│              Forgot?    │
│                         │
│  ┌───────────────────┐  │
│  │     LOGIN          │  │
│  └───────────────────┘  │
│                         │
│  ──── OR ────          │
│                         │
│  [G] [f] [🍎]          │
│                         │
│  Don't have account?    │
│  Sign Up                │
└─────────────────────────┘
```

**Specifications**:
- Logo: 80x80dp
- Input fields: 56dp height, 16dp horizontal padding
- Primary button: 56dp height, full width minus 32dp
- Social login buttons: 48x48dp
- Spacing between elements: 16dp

### Home Screen
**Purpose**: Main navigation hub

**Layout**:
```
┌─────────────────────────┐
│ ☰  Home          🔍 👤  │
├─────────────────────────┤
│                         │
│  Welcome, User!         │
│                         │
│  ┌─────────┬─────────┐  │
│  │ Card 1  │ Card 2  │  │
│  └─────────┴─────────┘  │
│                         │
│  Recent Activities      │
│  ┌───────────────────┐  │
│  │ Item 1            │  │
│  ├───────────────────┤  │
│  │ Item 2            │  │
│  ├───────────────────┤  │
│  │ Item 3            │  │
│  └───────────────────┘  │
│                         │
├─────────────────────────┤
│  🏠   📊   ➕   💬   ⚙️  │
└─────────────────────────┘
```

**Specifications**:
- App bar: 56dp height
- Cards: 50% width minus spacing, 120dp height
- List items: 72dp height with dividers
- Bottom navigation: 56dp height, 5 items max

### Detail Screen
**Purpose**: Display detailed information

**Layout**:
```
┌─────────────────────────┐
│ ←  Details         ⋮    │
├─────────────────────────┤
│                         │
│  ┌───────────────────┐  │
│  │                   │  │
│  │     [Image]       │  │
│  │                   │  │
│  └───────────────────┘  │
│                         │
│  Title                  │
│  Subtitle               │
│                         │
│  ┌───────────────────┐  │
│  │ Tab 1 │ Tab 2│Tab3│  │
│  ├───────────────────┤  │
│  │                   │  │
│  │  Tab Content      │  │
│  │                   │  │
│  │                   │  │
│  └───────────────────┘  │
│                         │
│  ┌───────────────────┐  │
│  │   Primary Action  │  │
│  └───────────────────┘  │
└─────────────────────────┘
```

## Component Library

### Buttons
```dart
// Primary Button
ElevatedButton(
  onPressed: () {},
  style: ElevatedButton.styleFrom(
    minimumSize: Size(double.infinity, 56),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(8),
    ),
  ),
  child: Text('PRIMARY ACTION'),
)

// Secondary Button
OutlinedButton(
  onPressed: () {},
  style: OutlinedButton.styleFrom(
    minimumSize: Size(double.infinity, 48),
  ),
  child: Text('SECONDARY ACTION'),
)

// Text Button
TextButton(
  onPressed: () {},
  child: Text('TEXT ACTION'),
)
```

### Input Fields
```dart
// Standard Input
TextField(
  decoration: InputDecoration(
    labelText: 'Label',
    hintText: 'Hint text',
    prefixIcon: Icon(Icons.email),
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(8),
    ),
    filled: true,
    fillColor: Colors.grey[100],
  ),
)

// Password Input
TextField(
  obscureText: true,
  decoration: InputDecoration(
    labelText: 'Password',
    prefixIcon: Icon(Icons.lock),
    suffixIcon: IconButton(
      icon: Icon(Icons.visibility),
      onPressed: () {},
    ),
  ),
)
```

### Cards
```dart
Card(
  elevation: 2,
  shape: RoundedRectangleBorder(
    borderRadius: BorderRadius.circular(12),
  ),
  child: Padding(
    padding: EdgeInsets.all(16),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Card Title', style: headline6),
        SizedBox(height: 8),
        Text('Card content', style: bodyText2),
      ],
    ),
  ),
)
```

### Lists
```dart
ListTile(
  leading: CircleAvatar(
    child: Icon(Icons.person),
  ),
  title: Text('List Item Title'),
  subtitle: Text('Supporting text'),
  trailing: Icon(Icons.chevron_right),
  onTap: () {},
)
```

## Responsive Design

### Breakpoints
```dart
class Breakpoints {
  static const double mobile = 0;      // 0-599dp
  static const double tablet = 600;    // 600-839dp
  static const double desktop = 840;   // 840dp+
}
```

### Adaptive Layouts
```dart
class AdaptiveLayout extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        if (constraints.maxWidth < Breakpoints.tablet) {
          return MobileLayout();
        } else if (constraints.maxWidth < Breakpoints.desktop) {
          return TabletLayout();
        } else {
          return DesktopLayout();
        }
      },
    );
  }
}
```

## Animation Specifications

### Page Transitions
```dart
// Slide transition
PageRouteBuilder(
  transitionDuration: Duration(milliseconds: 300),
  pageBuilder: (_, __, ___) => NewPage(),
  transitionsBuilder: (_, animation, __, child) {
    return SlideTransition(
      position: Tween<Offset>(
        begin: Offset(1.0, 0.0),
        end: Offset.zero,
      ).animate(CurvedAnimation(
        parent: animation,
        curve: Curves.easeInOut,
      )),
      child: child,
    );
  },
)
```

### Micro-interactions
```dart
// Button press animation
AnimatedContainer(
  duration: Duration(milliseconds: 150),
  transform: Matrix4.identity()..scale(isPressed ? 0.95 : 1.0),
  child: ElevatedButton(...),
)

// Loading animation
CircularProgressIndicator(
  strokeWidth: 2,
  valueColor: AlwaysStoppedAnimation<Color>(primary),
)
```

## Accessibility

### Content Labels
```dart
Semantics(
  label: 'Submit button',
  hint: 'Double tap to submit the form',
  child: ElevatedButton(...),
)
```

### Touch Targets
- Minimum touch target: 48x48dp
- Spacing between targets: 8dp minimum
- Clear visual feedback on interaction

### Text Contrast
- Normal text: 4.5:1 minimum
- Large text: 3:1 minimum
- Interactive elements: 3:1 minimum

## Platform-Specific Design

### Android (Material)
```dart
ThemeData(
  useMaterial3: true,
  colorScheme: ColorScheme.fromSeed(
    seedColor: primaryColor,
    brightness: Brightness.light,
  ),
)
```

### iOS (Cupertino)
```dart
CupertinoThemeData(
  primaryColor: CupertinoColors.systemBlue,
  brightness: Brightness.light,
  textTheme: CupertinoTextThemeData(...),
)
```

## Error States

### Empty State
```
┌─────────────────────────┐
│                         │
│         [Icon]          │
│                         │
│     No Data Found       │
│                         │
│  Try adding some items  │
│                         │
│    [Add Button]         │
│                         │
└─────────────────────────┘
```

### Error State
```
┌─────────────────────────┐
│                         │
│      [Error Icon]       │
│                         │
│   Something went wrong  │
│                         │
│  Please try again later │
│                         │
│    [Retry Button]       │
│                         │
└─────────────────────────┘
```

### Loading State
```
┌─────────────────────────┐
│                         │
│                         │
│    [Progress Spinner]   │
│                         │
│       Loading...        │
│                         │
│                         │
└─────────────────────────┘
```

## Design Tokens

### Spacing
```dart
class Spacing {
  static const double xs = 4;
  static const double sm = 8;
  static const double md = 16;
  static const double lg = 24;
  static const double xl = 32;
  static const double xxl = 48;
}
```

### Border Radius
```dart
class BorderRadius {
  static const double sm = 4;
  static const double md = 8;
  static const double lg = 12;
  static const double xl = 16;
  static const double round = 999;
}
```

### Elevation
```dart
class Elevation {
  static const double card = 2;
  static const double modal = 8;
  static const double dropdown = 4;
  static const double appBar = 4;
}
```