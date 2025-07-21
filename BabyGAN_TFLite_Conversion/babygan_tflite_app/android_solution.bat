@echo off
echo ============================================
echo  BabyGAN Android Solution
echo ============================================
echo.
echo Running the Android-fixed version that solves
echo the gray background issue.
echo.
echo Key fixes:
echo - CPU-only execution (no GPU/NNAPI)
echo - Explicit buffer management
echo - Auto-scaling output normalization
echo - Android memory alignment
echo.

REM Set working directory
cd /d E:\GitHub\BabyGAN\BabyGAN_TFLite_Conversion\babygan_tflite_app

REM Get dependencies
echo Getting dependencies...
flutter pub get

echo.
echo Starting Android solution app...
echo.
echo IMPORTANT: Keep these settings enabled:
echo [x] CPU Only Mode
echo [x] Explicit Buffer Management
echo [x] Auto-Scale Output
echo.

REM Run Android solution
flutter run -d emulator-5554 -t lib/android_solution.dart

pause