"""
Простой тест камеры - проверяем что камера работает
"""
import cv2
import sys

def test_camera():
    """Тестируем камеру без MediaPipe"""
    print("🎥 Тестирование камеры...")
    print("=" * 50)
    
    # Пробуем разные camera_id
    for camera_id in range(3):
        print(f"\nПробую камеру ID={camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"  ❌ Камера {camera_id} не доступна")
            continue
        
        # Устанавливаем разрешение
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Получаем реальное разрешение
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"  ✅ Камера {camera_id} работает: {width}x{height}")
        
        # Пробуем прочитать кадр
        success, frame = cap.read()
        if not success:
            print(f"  ⚠️ Не удалось прочитать кадр с камеры {camera_id}")
            cap.release()
            continue
        
        print(f"  ✅ Кадр прочитан успешно!")
        print(f"  📸 Тестирую камеру {camera_id}...")
        print(f"     Нажмите 'q' для перехода к следующей камере")
        print(f"     Нажмите 's' чтобы использовать эту камеру")
        
        # Создаем окно
        cv2.namedWindow(f'Camera Test - ID {camera_id}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Camera Test - ID {camera_id}', 800, 600)
        
        selected = False
        while True:
            success, frame = cap.read()
            if not success:
                print(f"  ❌ Потеряно соединение с камерой {camera_id}")
                break
            
            # Отзеркаливаем для удобства
            frame = cv2.flip(frame, 1)
            
            # Добавляем информацию на экран
            cv2.putText(frame, f"Camera ID: {camera_id}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {width}x{height}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' - next camera | 's' - select this camera", (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(f'Camera Test - ID {camera_id}', frame)
            
            key = cv2.waitKey(5) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                selected = True
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if selected:
            print(f"\n✅ Камера {camera_id} выбрана!")
            print(f"\n📝 Обновите src/config.py:")
            print(f"   camera_id: int = {camera_id}")
            return camera_id
    
    print("\n❌ Не найдено ни одной рабочей камеры!")
    return None


if __name__ == "__main__":
    print("=" * 50)
    print("🎥 Camera Diagnostic Tool")
    print("=" * 50)
    
    result = test_camera()
    
    if result is not None:
        print(f"\n🎉 Готово! Используйте camera_id = {result}")
    else:
        print("\n💡 Убедитесь что:")
        print("   1. Камера подключена")
        print("   2. Камера не занята другим приложением")
        print("   3. У приложения есть разрешение на доступ к камере")
    
    print("\nНажмите Enter для выхода...")
    input()