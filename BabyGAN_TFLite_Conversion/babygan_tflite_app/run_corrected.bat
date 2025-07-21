@echo off
echo ======================================
echo  BabyGAN TFLite Corrected Version
echo ======================================
echo.
echo This version fixes the gray output issue
echo by using proper tanh normalization.
echo.

REM Set working directory
cd /d E:\GitHub\BabyGAN\BabyGAN_TFLite_Conversion\babygan_tflite_app

REM Get dependencies
echo Getting dependencies...
flutter pub get

echo.
echo Running corrected version...
echo - Uses [-1, 1] to [0, 255] conversion
echo - Truncated normal distribution
echo - Toggle between tanh/sigmoid modes
echo.

REM Run corrected version
flutter run -d emulator-5554 -t lib/main_corrected.dart

pause