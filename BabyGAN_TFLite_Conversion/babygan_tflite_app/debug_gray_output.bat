@echo off
echo ============================================
echo  BabyGAN Gray Output Debug Tool
echo ============================================
echo.
echo This tool will diagnose why generated faces
echo appear as gray backgrounds and test fixes.
echo.

REM Set working directory
cd /d E:\GitHub\BabyGAN\BabyGAN_TFLite_Conversion\babygan_tflite_app

REM Get dependencies
echo Getting dependencies...
flutter pub get

echo.
echo Running debug version...
echo.
echo This will:
echo 1. Analyze the model's output range
echo 2. Test different normalization methods
echo 3. Show 4 different conversion approaches
echo.

REM Run debug version
flutter run -d emulator-5554 -t lib/main_debug.dart

pause