@echo off
echo ======================================
echo  BabyGAN TFLite Flutter App
echo ======================================
echo.

REM Set working directory
cd /d E:\GitHub\BabyGAN\BabyGAN_TFLite_Conversion\babygan_tflite_app

REM Get dependencies
echo Getting dependencies...
flutter pub get

REM Check devices
echo.
echo Available devices:
flutter devices

REM Run on Android emulator
echo.
echo Running app on Android emulator...
echo Using fixed version with proper tensor handling...
flutter run -d emulator-5554 -t lib/main_fixed.dart

pause