"""
Hand Tracking через MediaPipe
Отслеживает руку через веб-камеру и преобразует её движения в 3D трансформации
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from config import Config


@dataclass
class HandTransform:
    """
    3D трансформация руки (позиция + ротация)
    Это то, что мы получаем из hand tracking и передаем в контейнер
    """
    position: np.ndarray  # [x, y, z]
    rotation: np.ndarray  # [rx, ry, rz] в радианах
    scale: float = 1.0
    is_pinching: bool = False  # Жест "схватить"


class HandTracker:
    """
    Отслеживание руки и преобразование в 3D координаты
    
    Как это работает:
    1. MediaPipe находит 21 landmark на руке
    2. Мы берем центр ладони (landmark 9)
    3. Считаем ротацию по углу между пальцами
    4. Маппим 2D координаты камеры -> 3D координаты сцены
    """
    
    def __init__(self):
        # Инициализация MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,  # Для видео потока
            max_num_hands=Config.hand_tracking.max_num_hands,
            min_detection_confidence=Config.hand_tracking.min_detection_confidence,
            min_tracking_confidence=Config.hand_tracking.min_tracking_confidence
        )
        
        # Для рисования landmarks на изображении
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Инициализация камеры (КАК В РАБОЧЕМ ПРИМЕРЕ)
        self.cap = cv2.VideoCapture(Config.camera.camera_id)
        
        # Проверяем что камера открылась
        if not self.cap.isOpened():
            print(f"❌ Не удалось открыть камеру с ID={Config.camera.camera_id}")
            print("💡 Попробуйте изменить camera_id в config.py")
            raise RuntimeError("Camera not available")
        
        # Устанавливаем разрешение (КАК В РАБОЧЕМ ПРИМЕРЕ)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.camera.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.camera.height)
        self.cap.set(cv2.CAP_PROP_FPS, Config.camera.fps)
        
        # Получаем реальное разрешение
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📷 Камера настроена: {actual_width}x{actual_height}")
        
        # Прогреваем камеру (первые кадры могут быть черными)
        for _ in range(5):
            self.cap.read()
        
        # Предыдущее состояние для smoothing (сглаживания)
        self.prev_position = np.array([0.0, 0.0, -5.0], dtype=np.float32)
        self.prev_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Текущий frame
        self.current_frame = None
        
        print(f"[HandTracker] Камера инициализирована: {Config.camera.width}x{Config.camera.height}")
    
    def read_frame(self) -> bool:
        """
        Читаем frame с камеры (КАК В РАБОЧЕМ ПРИМЕРЕ)
        Returns: True если frame прочитан успешно
        """
        success, frame = self.cap.read()
        if success:
            # Flip для зеркального эффекта (КАК В РАБОЧЕМ ПРИМЕРЕ)
            # Это делает управление более интуитивным
            self.current_frame = cv2.flip(frame, 1)
        else:
            self.current_frame = None
        return success
    
    def process_hand(self) -> Optional[list]:
        """
        Обрабатываем текущий frame и извлекаем информацию о руках
        
        Returns:
            List[HandTransform] если руки найдены (1 или 2 руки), иначе None
        """
        if self.current_frame is None:
            return None
        
        # Конвертируем BGR (OpenCV) -> RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        
        # Запускаем MediaPipe detection
        results = self.hands.process(rgb_frame)
        
        # Если рук не найдено
        if not results.multi_hand_landmarks:
            return None
        
        # Обрабатываем все найденные руки (1 или 2)
        hand_transforms = []
        
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Рисуем landmarks на frame (если включено) - КРУПНЕЕ КАК В ПРИМЕРЕ
            if Config.render.show_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    self.current_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    # Точки (landmarks) - зеленые и крупные
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                    # Линии (connections) - фиолетовые
                    self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=3)
                )
            
            # Извлекаем трансформацию из landmarks
            transform = self._landmarks_to_transform(hand_landmarks, idx)
            hand_transforms.append(transform)
        
        return hand_transforms
        
        # Рисуем landmarks на frame (если включено) - КРУПНЕЕ КАК В ПРИМЕРЕ
        if Config.render.show_hand_landmarks:
            self.mp_draw.draw_landmarks(
                self.current_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                # Точки (landmarks) - зеленые и крупные
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
                # Линии (connections) - фиолетовые
                self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=3)
            )
        
        # Извлекаем трансформацию из landmarks
        transform = self._landmarks_to_transform(hand_landmarks)
        
        return transform
    
    def _landmarks_to_transform(self, landmarks, hand_index: int = 0) -> HandTransform:
        """
        Преобразуем MediaPipe landmarks в 3D трансформацию
        
        Landmarks которые мы используем:
        - Landmark 0: запястье (wrist)
        - Landmark 9: центр ладони
        - Landmark 5: основание указательного пальца
        - Landmark 17: основание мизинца
        - Landmark 4: кончик большого пальца
        - Landmark 8: кончик указательного пальца
        """
        # Извлекаем ключевые точки
        wrist = np.array([landmarks.landmark[0].x, landmarks.landmark[0].y, landmarks.landmark[0].z])
        palm_center = np.array([landmarks.landmark[9].x, landmarks.landmark[9].y, landmarks.landmark[9].z])
        index_base = np.array([landmarks.landmark[5].x, landmarks.landmark[5].y, landmarks.landmark[5].z])
        pinky_base = np.array([landmarks.landmark[17].x, landmarks.landmark[17].y, landmarks.landmark[17].z])
        
        # === ПОЗИЦИЯ ===
        # Берем центр ладони как позицию руки
        # Координаты MediaPipe: x [0,1], y [0,1], z [отрицательные]
        # Преобразуем в мировые координаты: x [-5, 5], y [-5, 5], z [-10, 0]
        position = np.array([
            (palm_center[0] - 0.5) * Config.hand_tracking.hand_to_world_scale,  # x: центрируем и масштабируем
            (0.5 - palm_center[1]) * Config.hand_tracking.hand_to_world_scale,  # y: инвертируем (OpenCV y вниз)
            palm_center[2] * Config.hand_tracking.hand_to_world_scale - 5.0,    # z: смещаем от камеры
        ], dtype=np.float32)
        
        # === РОТАЦИЯ ===
        # Считаем ротацию по векторам между landmarks
        
        # Вектор вдоль ладони (от запястья к пальцам)
        palm_direction = palm_center - wrist
        palm_direction = palm_direction / (np.linalg.norm(palm_direction) + 1e-6)  # Нормализуем
        
        # Вектор поперек ладони (от мизинца к указательному)
        palm_width = index_base - pinky_base
        palm_width = palm_width / (np.linalg.norm(palm_width) + 1e-6)
        
        # Вычисляем углы Эйлера (упрощенная версия)
        # rotation_x: наклон вперед-назад
        rotation_x = np.arctan2(palm_direction[2], palm_direction[1])
        
        # rotation_y: поворот влево-вправо  
        rotation_y = np.arctan2(palm_direction[0], palm_direction[2])
        
        # rotation_z: крен (roll)
        rotation_z = np.arctan2(palm_width[1], palm_width[0])
        
        rotation = np.array([rotation_x, rotation_y, rotation_z], dtype=np.float32)
        
        # === SMOOTHING (сглаживание) ===
        # Без сглаживания движения будут дергаными
        alpha_pos = Config.hand_tracking.position_smoothing
        alpha_rot = Config.hand_tracking.rotation_smoothing
        
        smooth_position = alpha_pos * self.prev_position + (1 - alpha_pos) * position
        smooth_rotation = alpha_rot * self.prev_rotation + (1 - alpha_rot) * rotation
        
        self.prev_position = smooth_position
        self.prev_rotation = smooth_rotation
        
        # === PINCH DETECTION (жест "схватить") ===
        # Расстояние между большим и указательным пальцем
        thumb_tip = np.array([landmarks.landmark[4].x, landmarks.landmark[4].y, landmarks.landmark[4].z])
        index_tip = np.array([landmarks.landmark[8].x, landmarks.landmark[8].y, landmarks.landmark[8].z])
        pinch_distance = np.linalg.norm(thumb_tip - index_tip)
        
        is_pinching = pinch_distance < Config.hand_tracking.pinch_threshold
        
        return HandTransform(
            position=smooth_position,
            rotation=smooth_rotation,
            scale=1.0,
            is_pinching=is_pinching
        )
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Возвращает текущий frame для отображения"""
        return self.current_frame
    
    def release(self):
        """Освобождаем ресурсы"""
        self.cap.release()
        self.hands.close()
        print("[HandTracker] Ресурсы освобождены")


# Тестирование модуля
if __name__ == "__main__":
    """
    Тест: просто показываем камеру с отслеживанием руки
    Запустить: python src/hand_tracking.py
    """
    print("🚀 Hand Tracker Test - запуск...")
    print("Показываю камеру с hand tracking.")
    print("Нажмите 'q' для выхода\n")
    
    tracker = HandTracker()
    
    # Создаем окно с правильным размером (КАК В ПРИМЕРЕ)
    cv2.namedWindow("Hand Tracking Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Tracking Test", 1240, 700)
    
    while True:
        # Читаем frame
        if not tracker.read_frame():
            print("❌ Ошибка чтения с камеры")
            break
        
        # Обрабатываем hand tracking
        transform = tracker.process_hand()
        
        # Получаем frame для отображения
        frame = tracker.get_frame()
        if frame is None:
            continue
        
        # Добавляем информацию на экран (КАК В ПРИМЕРЕ)
        h, w, _ = frame.shape
        
        if transform is not None:
            # Показываем координаты руки
            pos_text = f"Position: ({transform.position[0]:.2f}, {transform.position[1]:.2f}, {transform.position[2]:.2f})"
            rot_text = f"Rotation: ({transform.rotation[0]:.2f}, {transform.rotation[1]:.2f}, {transform.rotation[2]:.2f})"
            pinch_text = f"Pinch: {'YES' if transform.is_pinching else 'NO'}"
            
            cv2.putText(frame, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, rot_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, pinch_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255) if transform.is_pinching else (0, 255, 0), 2)
        else:
            # Рука не найдена
            cv2.putText(frame, "No hand detected - show your hand!", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        
        # Инструкции
        cv2.putText(frame, "Press 'q' to quit", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Показываем frame
        cv2.imshow("Hand Tracking Test", frame)
        
        # Выход по 'q' (КАК В ПРИМЕРЕ)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    
    # Очистка
    tracker.release()
    cv2.destroyAllWindows()
    print("✅ Test завершен")