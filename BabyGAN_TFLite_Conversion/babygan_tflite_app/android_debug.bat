@echo off
echo ============================================
echo  BabyGAN Android Gray Background Debugger
echo ============================================
echo.
echo This tool tests different Android configurations
echo to identify why faces appear as gray backgrounds.
echo.

REM Set working directory
cd /d E:\GitHub\BabyGAN\BabyGAN_TFLite_Conversion\babygan_tflite_app

REM Get dependencies
echo Getting dependencies...
flutter pub get

echo.
echo Running Android-specific diagnostic tool...
echo.
echo This will test:
echo - CPU vs GPU delegate execution
echo - NNAPI compatibility
echo - Memory alignment issues
echo - Float precision handling
echo - Threading configurations
echo.

REM Run Android debug analyzer
flutter run -d emulator-5554 -t lib/android_debug_analyzer.dart

pause