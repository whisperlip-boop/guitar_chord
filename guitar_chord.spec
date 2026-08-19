# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec file for Guitar Chord Viewer
#
# Build command (run from the project root):
#   pyinstaller guitar_chord.spec
#
# Output: dist/guitar_chord.exe  (single-file, no console window)

block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # External chord JSON datasets — bundled into the exe and extracted
        # at runtime to sys._MEIPASS/data/.
        ('data/chords_db_guitar.json', 'data'),
        ('data/tvk_chords_complete.json', 'data'),
        # App icon — must be accessible via get_resource_path() at runtime.
        ('guitar.ico', '.'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Unused PySide6 modules — trims onefile exe size significantly.
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.QtNetwork",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtSql",
        "PySide6.QtDesigner",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtBluetooth",
        "PySide6.QtPositioning",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtTest",
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DRender",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='guitar_chord',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,               # compress with UPX if available (smaller exe)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='guitar.ico',      # title-bar / exe file icon
)
