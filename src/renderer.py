import moderngl
import numpy as np
from pyrr import Matrix44, Vector3
import glm
from typing import Tuple

from config import Config


class Camera:
    """
    3D камера для просмотра сцены
    Поддерживает orbit camera (вращение вокруг центра)
    """
    
    def __init__(self, aspect_ratio: float):
        self.aspect_ratio = aspect_ratio
        
        # Параметры камеры
        self.position = Vector3([0.0, 0.0, 10.0])  # Позиция камеры
        self.target = Vector3([0.0, 0.0, 0.0])     # Точка на которую смотрит
        self.up = Vector3([0.0, 1.0, 0.0])         # Вектор "верх"
        
        # Параметры проекции
        self.fov = Config.render.fov
        self.near = Config.render.near_plane
        self.far = Config.render.far_plane
        
        print(f"[Camera] Инициализирована с aspect={aspect_ratio:.2f}")
    
    def get_view_matrix(self) -> np.ndarray:
        """
        View matrix - трансформирует мировые координаты в координаты камеры
        """
        view = Matrix44.look_at(
            self.position,
            self.target,
            self.up
        )
        return np.array(view, dtype=np.float32)
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Projection matrix - создает перспективу (далекие объекты меньше)
        """
        proj = Matrix44.perspective_projection(
            self.fov,
            self.aspect_ratio,
            self.near,
            self.far
        )
        return np.array(proj, dtype=np.float32)
    
    def get_view_projection_matrix(self) -> np.ndarray:
        """
        VP matrix = Projection * View
        Это главная матрица для трансформации координат
        """
        view = self.get_view_matrix()
        proj = self.get_projection_matrix()
        return np.matmul(proj, view)


class ParticleRenderer:
    """
    Рендерер для частиц жидкости
    Использует point sprites (каждая частица = точка)
    """
    
    def __init__(self, ctx: moderngl.Context, max_particles: int):
        self.ctx = ctx
        self.max_particles = max_particles
        
        # Vertex shader - обрабатывает каждую вершину (частицу)
        self.vertex_shader = """
        #version 330
        
        // Input: позиция частицы
        in vec3 in_position;
        
        // Uniform: матрицы трансформации
        uniform mat4 mvp;  // Model-View-Projection matrix
        
        void main() {
            // Трансформируем позицию частицы в screen space
            gl_Position = mvp * vec4(in_position, 1.0);
            
            // Размер точки на экране
            gl_PointSize = 8.0;
        }
        """
        
        # Fragment shader - определяет цвет каждого пикселя
        self.fragment_shader = """
        #version 330
        
        // Output: цвет пикселя
        out vec4 fragColor;
        
        // Uniform: цвет частиц
        uniform vec4 particle_color;
        
        void main() {
            // Делаем круглые частицы (вместо квадратных точек)
            vec2 coord = gl_PointCoord - vec2(0.5);  // Центрируем координаты
            float dist = length(coord);
            
            if (dist > 0.5) {
                discard;  // Отбрасываем пиксели вне круга
            }
            
            // Мягкие края (anti-aliasing)
            float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
            
            fragColor = vec4(particle_color.rgb, alpha * particle_color.a);
        }
        """
        
        # Компилируем шейдеры
        self.program = ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader
        )
        
        # Создаем буфер для позиций частиц (будет обновляться каждый кадр)
        self.vbo = ctx.buffer(reserve=max_particles * 12)  # 3 floats * 4 bytes = 12 bytes per particle
        
        # Vertex Array Object - описывает структуру данных
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, '3f', 'in_position')]  # 3 floats для xyz
        )
        
        print(f"[ParticleRenderer] Инициализирован для {max_particles} частиц")
    
    def render(self, positions: np.ndarray, vp_matrix: np.ndarray):
        """
        Рендерим частицы
        
        Args:
            positions: numpy array [N, 3] с позициями частиц
            vp_matrix: view-projection матрица камеры
        """
        num_particles = positions.shape[0]
        
        # Обновляем буфер с позициями
        self.vbo.write(positions.astype(np.float32).tobytes())
        
        # Передаем uniforms в шейдер
        self.program['mvp'].write(vp_matrix.tobytes())
        self.program['particle_color'].value = Config.render.particle_color
        
        # Рендерим точки
        self.vao.render(moderngl.POINTS, vertices=num_particles)


class ContainerRenderer:
    """
    Рендерер для wireframe контейнера
    Рисует ребра куба
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Vertex shader для линий контейнера
        self.vertex_shader = """
        #version 330
        
        in vec3 in_position;
        
        uniform mat4 mvp;
        uniform mat4 model;  // Трансформация контейнера (позиция + ротация)
        
        void main() {
            // Сначала применяем model transform, потом view-projection
            vec4 world_pos = model * vec4(in_position, 1.0);
            gl_Position = mvp * world_pos;
        }
        """
        
        self.fragment_shader = """
        #version 330
        
        out vec4 fragColor;
        uniform vec4 line_color;
        
        void main() {
            fragColor = line_color;
        }
        """
        
        # Компилируем шейдеры
        self.program = ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader
        )
        
        # Создаем геометрию куба (vertices)
        self._create_cube_geometry()
        
        print("[ContainerRenderer] Инициализирован")
    
    def _create_cube_geometry(self):
        """
        Создаем vertices и indices для wireframe куба
        
        Куб имеет 8 вершин и 12 ребер
        """
        w = Config.container.width / 2
        h = Config.container.height / 2
        d = Config.container.depth / 2
        
        # 8 вершин куба
        vertices = np.array([
            # Нижние 4 вершины
            [-w, -h, -d],  # 0
            [ w, -h, -d],  # 1
            [ w, -h,  d],  # 2
            [-w, -h,  d],  # 3
            # Верхние 4 вершины
            [-w,  h, -d],  # 4
            [ w,  h, -d],  # 5
            [ w,  h,  d],  # 6
            [-w,  h,  d],  # 7
        ], dtype=np.float32)
        
        # 12 ребер куба (каждое ребро = 2 индекса)
        indices = np.array([
            # Нижняя грань
            0, 1,  1, 2,  2, 3,  3, 0,
            # Верхняя грань
            4, 5,  5, 6,  6, 7,  7, 4,
            # Вертикальные ребра
            0, 4,  1, 5,  2, 6,  3, 7,
        ], dtype=np.uint32)
        
        # Создаем буферы
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.ibo = self.ctx.buffer(indices.tobytes())
        
        # VAO
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f', 'in_position')],
            self.ibo
        )
        
        self.num_indices = len(indices)
    
    def render(self, vp_matrix: np.ndarray, model_matrix: np.ndarray):
        """
        Рендерим контейнер
        
        Args:
            vp_matrix: view-projection матрица камеры
            model_matrix: трансформация контейнера (позиция + ротация от руки)
        """
        # Передаем uniforms
        self.program['mvp'].write(vp_matrix.tobytes())
        self.program['model'].write(model_matrix.tobytes())
        self.program['line_color'].value = Config.container.wireframe_color
        
        # Рендерим линии
        self.vao.render(moderngl.LINES, vertices=self.num_indices)


class Renderer:
    """
    Главный рендерер - управляет всеми рендерерами и камерой
    """
    
    def __init__(self, ctx: moderngl.Context, window_size: Tuple[int, int]):
        self.ctx = ctx
        self.window_size = window_size
        
        # Настройка OpenGL
        ctx.enable(moderngl.DEPTH_TEST)  # Z-buffer для корректной глубины
        ctx.enable(moderngl.BLEND)        # Прозрачность
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)  # Размер точек в шейдере
        
        # Создаем камеру
        aspect_ratio = window_size[0] / window_size[1]
        self.camera = Camera(aspect_ratio)
        
        # Создаем рендереры
        self.particle_renderer = ParticleRenderer(ctx, Config.sph.num_particles)
        self.container_renderer = ContainerRenderer(ctx)
        
        print(f"[Renderer] Инициализирован {window_size[0]}x{window_size[1]}")
    
    def render_frame(self, particle_positions: np.ndarray, container_transform: np.ndarray):
        """
        Рендерим один кадр
        
        Args:
            particle_positions: позиции всех частиц [N, 3]
            container_transform: 4x4 матрица трансформации контейнера
        """
        # Очищаем экран
        self.ctx.clear(
            Config.render.background_color[0],
            Config.render.background_color[1],
            Config.render.background_color[2],
            Config.render.background_color[3]
        )
        
        # Получаем VP матрицу камеры
        vp_matrix = self.camera.get_view_projection_matrix()
        
        # Рендерим контейнер
        self.container_renderer.render(vp_matrix, container_transform)
        
        # Рендерим частицы
        self.particle_renderer.render(particle_positions, vp_matrix)
    
    def resize(self, width: int, height: int):
        """Обновляем размер окна"""
        self.window_size = (width, height)
        self.camera.aspect_ratio = width / height
        self.ctx.viewport = (0, 0, width, height)


def create_transform_matrix(position: np.ndarray, rotation: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Создает 4x4 матрицу трансформации из позиции, ротации и масштаба
    
    Args:
        position: [x, y, z]
        rotation: [rx, ry, rz] в радианах (углы Эйлера)
        scale: uniform scale
    
    Returns:
        4x4 numpy array
    """
    # Используем glm для создания матрицы
    mat = glm.mat4(1.0)  # Identity matrix
    
    # Применяем трансформации в порядке: Scale -> Rotate -> Translate
    mat = glm.translate(mat, glm.vec3(position[0], position[1], position[2]))
    
    # Ротация (Euler angles: XYZ order)
    mat = glm.rotate(mat, rotation[0], glm.vec3(1, 0, 0))  # Rotate X
    mat = glm.rotate(mat, rotation[1], glm.vec3(0, 1, 0))  # Rotate Y
    mat = glm.rotate(mat, rotation[2], glm.vec3(0, 0, 1))  # Rotate Z
    
    mat = glm.scale(mat, glm.vec3(scale, scale, scale))
    
    # Конвертируем в numpy
    return np.array(mat, dtype=np.float32).T  # Transpose для column-major


# Тест модуля
if __name__ == "__main__":
    """
    Тест рендерера (создаем окно и рисуем статичную сцену)
    """
    import moderngl_window as mglw
    
    class TestWindow(mglw.WindowConfig):
        gl_version = (3, 3)
        title = "Renderer Test"
        window_size = (1280, 720)
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # Создаем рендерер
            self.renderer = Renderer(self.ctx, self.window_size)
            
            # Тестовые данные
            # Создаем несколько частиц в форме куба
            n = 10
            positions = []
            for x in range(n):
                for y in range(n):
                    for z in range(n):
                        positions.append([
                            (x - n/2) * 0.2,
                            (y - n/2) * 0.2,
                            (z - n/2) * 0.2
                        ])
            
            self.test_positions = np.array(positions[:Config.sph.num_particles], dtype=np.float32)
            
            # Контейнер в начальной позиции
            self.container_transform = create_transform_matrix(
                position=np.array([0.0, 0.0, 0.0]),
                rotation=np.array([0.0, 0.0, 0.0])
            )
            
            self.rotation_angle = 0.0
            
            print("Тест рендерера запущен. Закройте окно для выхода.")
        
        def render(self, time, frame_time):
            # Анимируем ротацию контейнера
            self.rotation_angle += frame_time
            self.container_transform = create_transform_matrix(
                position=np.array([0.0, 0.0, 0.0]),
                rotation=np.array([self.rotation_angle * 0.5, self.rotation_angle, 0.0])
            )
            
            # Рендерим
            self.renderer.render_frame(self.test_positions, self.container_transform)
    
    # Запускаем тест
    mglw.run_window_config(TestWindow)