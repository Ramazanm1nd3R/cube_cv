from dataclasses import dataclass
import numpy as np


@dataclass
class WindowConfig:
    """Настройки окна приложения"""
    width: int = 1280
    height: int = 720
    title: str = "Fluid Gesture Simulator"
    fps_target: int = 60
    

@dataclass
class CameraConfig:
    """Настройки камеры для hand tracking"""
    width: int = 640
    height: int = 480
    fps: int = 30
    camera_id: int = 0  # 0 = дефолтная веб-камера


@dataclass
class SPHConfig:
    """
    Параметры SPH (Smoothed Particle Hydrodynamics)
    Это ядро физической симуляции жидкости
    """
    # Количество частиц
    num_particles: int = 500
    
    # Физические свойства частиц
    particle_radius: float = 0.052  # Радиус влияния частицы
    particle_mass: float = 1.0      # Масса каждой частицы
    rest_density: float = 1000.0    # Плотность покоя (как у воды)
    
    # Силы взаимодействия
    gas_constant: float = 2000.0    # Константа давления (stiffness)
    viscosity: float = 0.001        # Вязкость жидкости (0.001 = вода, больше = мед)
    cohesion: float = 0.036         # Сила сцепления частиц (surface tension)
    adhesion: float = 0.0           # Прилипание к стенкам
    
    # Физика движения
    damping: float = 0.059          # Затухание скорости (энергопотери)
    friction: float = 0.0           # Трение о стенки
    restitution: float = 0.08       # Упругость отскока (0-1)
    
    # Гравитация
    gravity: np.ndarray = None      # Будет установлено в __post_init__
    
    # Оптимизация
    smoothing_radius: float = 0.1   # Радиус сглаживания для SPH kernel
    max_speed: float = 100.0        # Максимальная скорость частицы
    
    # Spatial hashing (для быстрого поиска соседей)
    cell_size: float = 0.1          # Размер ячейки spatial grid
    
    def __post_init__(self):
        """Инициализация значений по умолчанию"""
        if self.gravity is None:
            self.gravity = np.array([0.0, -9.81, 0.0], dtype=np.float32)


@dataclass
class ContainerConfig:
    """Параметры 3D контейнера"""
    # Размеры контейнера (куб)
    width: float = 2.0
    height: float = 2.0
    depth: float = 2.0
    
    # Начальная позиция
    initial_position: np.ndarray = None
    initial_rotation: np.ndarray = None
    
    # Визуализация
    wireframe_color: tuple = (1.0, 0.6, 0.2, 1.0)  # Оранжевый
    wireframe_thickness: float = 2.0
    
    def __post_init__(self):
        if self.initial_position is None:
            self.initial_position = np.array([0.0, 0.0, -5.0], dtype=np.float32)
        if self.initial_rotation is None:
            self.initial_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class HandTrackingConfig:
    """Параметры hand tracking через MediaPipe"""
    # MediaPipe параметры
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5
    max_num_hands: int = 2  # Отслеживаем только одну руку
    
    # Маппинг координат руки -> 3D пространство
    hand_to_world_scale: float = 10.0  # Множитель для координат руки
    
    # Smoothing (сглаживание движений)
    position_smoothing: float = 0.7  # 0-1, чем больше тем плавнее
    rotation_smoothing: float = 0.8
    
    # Жесты
    pinch_threshold: float = 0.05  # Расстояние для "схватить"


@dataclass
class RenderConfig:
    """Настройки рендеринга"""
    # Цвет частиц (оранжевый как на картинке)
    particle_color: tuple = (1.0, 0.5, 0.2, 1.0)
    particle_size: float = 8.0  # Размер точки на экране
    
    # Фон
    background_color: tuple = (0.0, 0.0, 0.0, 1.0)  # Черный
    
    # Камера (3D вид)
    fov: float = 60.0  # Field of view
    near_plane: float = 0.1
    far_plane: float = 100.0
    
    # UI
    show_debug_info: bool = True
    show_hand_landmarks: bool = True


# Глобальный конфиг - единая точка доступа ко всем настройкам
class Config:
    """Центральный конфиг приложения"""
    window = WindowConfig()
    camera = CameraConfig()
    sph = SPHConfig()
    container = ContainerConfig()
    hand_tracking = HandTrackingConfig()
    render = RenderConfig()
    
    @classmethod
    def print_summary(cls):
        """Вывод всех настроек для отладки"""
        print("=== Fluid Gesture Simulator Config ===")
        print(f"Window: {cls.window.width}x{cls.window.height}")
        print(f"Particles: {cls.sph.num_particles}")
        print(f"Viscosity: {cls.sph.viscosity}")
        print(f"Camera: {cls.camera.width}x{cls.camera.height}")
        print("=" * 40)