# GOLD AI - Build & Packaging Guide

## Prerequisites

1. Python 3.10 or higher
2. pip package manager
3. Git (optional, for version control)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd gold_ai
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Development Mode
```bash
python main.py
```

### With Custom Configuration
```bash
python main.py --config config/settings.yaml
```

## Building Executable (.exe)

### Using PyInstaller

#### 1. Install PyInstaller
```bash
pip install pyinstaller
```

#### 2. Build Single Executable
```bash
pyinstaller --onefile --windowed --name "GOLD_AI" main.py
```

Options explained:
- `--onefile`: Creates a single .exe file
- `--windowed`: No console window (GUI application)
- `--name`: Name of the executable

#### 3. Build with Icon and Additional Options
```bash
pyinstaller --onefile --windowed --icon=assets/icon.ico --add-data "config;config" --add-data "assets;assets" main.py
```

#### 4. Advanced Build Specification
Create a `.spec` file for more control:

```bash
pyinstaller --name=GOLD_AI --onefile --windowed main.py
```

This creates `build/GOLD_AI.spec` which can be edited for advanced configurations.

### Build Output
After successful build, the executable will be located at:
```
dist/GOLD_AI.exe
```

## Creating Installer (Optional)

### Using Inno Setup (Windows)

1. Download and install Inno Setup from https://jrsoftware.org/isinfo.php

2. Create `installer.iss` script:

```ini
[Setup]
AppName=GOLD AI
AppVersion=1.0.0
DefaultDirName={pf}\GOLD AI
DefaultGroupName=GOLD AI
OutputBaseFilename=GOLD_AI_Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "dist\GOLD_AI.exe"; DestDir: "{app}"
Source: "config\*"; DestDir: "{app}\config"; Flags: recursesubdirs
Source: "assets\*"; DestDir: "{app}\assets"; Flags: recursesubdirs

[Icons]
Name: "{group}\GOLD AI"; Filename: "{app}\GOLD_AI.exe"
Name: "{commondesktop}\GOLD AI"; Filename: "{app}\GOLD_AI.exe"

[Run]
Filename: "{app}\GOLD_AI.exe"; Description: "Launch GOLD AI"; Flags: nowait postinstall skipifsilent
```

3. Compile installer:
```bash
iscc installer.iss
```

The installer will be created in `Output/GOLD_AI_Setup.exe`

## Cross-Platform Builds

### Windows
Use PyInstaller on Windows to create .exe files.

### macOS
```bash
pyinstaller --onefile --windowed --name "GOLD_AI" main.py
```
Creates `dist/GOLD_AI.app` bundle.

### Linux
```bash
pyinstaller --onefile --name "GOLD_AI" main.py
```
Creates `dist/GOLD_AI` executable.

## Troubleshooting

### Missing Dependencies
If you encounter missing module errors after building:

1. Check all imports are included in hidden imports:
```bash
pyinstaller --onefile --hidden-import=numpy --hidden-import=pandas main.py
```

2. Or add to spec file:
```python
hiddenimports = ['numpy', 'pandas', 'sklearn']
```

### Large Executable Size
Reduce size by excluding unnecessary modules:
```bash
pyinstaller --onefile --exclude-module matplotlib --exclude-module scipy main.py
```

### Application Crashes on Startup
1. Run without `--windowed` to see error messages
2. Check logs in `%APPDATA%/GOLD_AI/logs/` (Windows) or `~/.local/share/GOLD_AI/logs/` (Linux)

## Performance Optimization

For better performance:

1. Use optimized builds:
```bash
pyinstaller --onefile --optimize=2 main.py
```

2. Enable LTO (Link Time Optimization) if available

3. Consider using Nuitka as alternative compiler:
```bash
pip install nuitka
nuitka --standalone --onefile --windows-disable-console main.py
```

## Security Notes

1. Never include API keys in source code
2. Use environment variables for sensitive configuration
3. Consider code signing for production executables
4. Validate all external data inputs

## Support

For issues or questions:
- Check README.md for documentation
- Review logs directory for error details
- Contact development team
