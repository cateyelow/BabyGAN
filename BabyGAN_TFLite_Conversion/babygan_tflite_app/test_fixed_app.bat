@echo off
echo ======================================
echo  Testing BabyGAN TFLite Fixed Version
echo ======================================
echo.

REM Set working directory
cd /d E:\GitHub\BabyGAN\BabyGAN_TFLite_Conversion\babygan_tflite_app

REM Get dependencies
echo Getting dependencies...
flutter pub get

echo.
echo Running fixed version with proper tensor handling...
echo This version fixes the tensor shape mismatch error.
echo.

REM Run the fixed version
flutter run -d emulator-5554 -t lib/main_fixed.dart

pause