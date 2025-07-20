@echo off
echo ======================================
echo  BabyGAN TFLite App Setup Verification
echo ======================================
echo.

REM Set working directory
cd /d E:\GitHub\BabyGAN\BabyGAN_TFLite_Conversion\babygan_tflite_app

echo [1/5] Checking Flutter installation...
flutter --version
if %errorlevel% neq 0 (
    echo ERROR: Flutter not found! Please install Flutter.
    pause
    exit /b 1
)

echo.
echo [2/5] Checking Android emulator...
flutter devices | findstr "emulator-5554"
if %errorlevel% neq 0 (
    echo WARNING: Android emulator not found. Please start an emulator.
)

echo.
echo [3/5] Checking TFLite model file...
if exist "assets\models\stylegan_mobile_working.tflite" (
    echo SUCCESS: TFLite model found!
) else (
    echo ERROR: TFLite model not found in assets/models/
    pause
    exit /b 1
)

echo.
echo [4/5] Checking fixed main.dart...
if exist "lib\main_fixed.dart" (
    echo SUCCESS: Fixed Flutter code found!
) else (
    echo ERROR: main_fixed.dart not found in lib/
    pause
    exit /b 1
)

echo.
echo [5/5] Getting Flutter dependencies...
flutter pub get
if %errorlevel% neq 0 (
    echo ERROR: Failed to get Flutter dependencies
    pause
    exit /b 1
)

echo.
echo ======================================
echo  ✅ All checks passed!
echo ======================================
echo.
echo Ready to test the app. Run one of these:
echo   - test_fixed_app.bat  (to test the fixed version)
echo   - run_app.bat         (to run the app)
echo.
pause