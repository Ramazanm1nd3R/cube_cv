"""
Main Entry Point для Fluid Gesture Simulator - GPU ВЕРСИЯ

Запуск:
    python main.py
"""
import sys
import os

# Добавляем src в Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.app import run_app


def main():
    """
    Главная функция - точка входа
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║            🌊 Fluid Gesture Simulator 🌊                     ║
    ║                  GPU ACCELERATED VERSION                     ║
    ║                                                              ║
    ║  Интерактивная симуляция жидкости управляемая жестами рук   ║
    ║         Физика SPH выполняется на GPU (Compute Shaders)     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Технологии:
    • Computer Vision: MediaPipe + OpenCV
    • Physics: SPH на GPU (OpenGL Compute Shaders)  🔥
    • Rendering: ModernGL (OpenGL 4.3+) + Pygame
    • Math: NumPy + PyGLM
    
    Требования:
    ✓ OpenGL 4.3+ (для compute shaders)
    ✓ Веб-камера подключена
    ✓ GPU с поддержкой compute shaders
    ✓ Хорошее освещение для hand tracking
    
    Нажмите Ctrl+C для выхода
    """)
    
    try:
        # Запускаем приложение
        run_app()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Получен сигнал прерывания (Ctrl+C)")
        print("Завершение работы...")
        
    except Exception as e:
        print(f"\n\n❌ КРИТИЧЕСКАЯ ОШИБКА:\n{e}")
        print("\nПроверьте:")
        print("  1. OpenGL 4.3+ поддерживается (для compute shaders)")
        print("  2. Установлены ли все зависимости (requirements.txt)")
        print("  3. Подключена ли камера")
        print("  4. GPU поддерживает compute shaders")
        print("\nДля старой версии без GPU: используйте app.py")
        
        import traceback
        traceback.print_exc()
        
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())