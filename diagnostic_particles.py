"""
Диагностика частиц - проверяем почему они не видны
"""
import moderngl
import pygame
import numpy as np
from pyrr import Matrix44

def test_particles():
    """Минимальный тест - просто частицы без физики"""
    
    print("="*70)
    print("🔍 ДИАГНОСТИКА ЧАСТИЦ")
    print("="*70)
    
    # Инициализация
    pygame.init()
    pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    
    print(f"\n✅ OpenGL Context: {ctx.version_code}")
    print(f"✅ OpenGL Version: {ctx.info['GL_VERSION']}")
    
    # === СОЗДАЕМ ЧАСТИЦЫ ===
    num_particles = 100  # Начнем с малого
    
    # Позиции - просто в центре экрана
    positions = np.zeros((num_particles, 3), dtype='f4')
    
    # Распределяем в небольшой области
    for i in range(num_particles):
        positions[i] = [
            np.random.uniform(-0.5, 0.5),  # x
            np.random.uniform(-0.5, 0.5),  # y
            0.0  # z = 0 (прямо перед камерой)
        ]
    
    print(f"\n📊 Частицы созданы:")
    print(f"  Количество: {num_particles}")
    print(f"  Позиции min: {np.min(positions, axis=0)}")
    print(f"  Позиции max: {np.max(positions, axis=0)}")
    print(f"  Первая частица: {positions[0]}")
    
    # Создаем GPU buffer
    position_buffer = ctx.buffer(positions.tobytes())
    
    # === SHADER (МАКСИМАЛЬНО ПРОСТОЙ) ===
    vertex_shader = """
    #version 330
    in vec3 in_position;
    
    void main() {
        // Прямо в clip space, без трансформаций
        gl_Position = vec4(in_position, 1.0);
        gl_PointSize = 50.0;  // ОГРОМНЫЙ размер для теста!
    }
    """
    
    fragment_shader = """
    #version 330
    out vec4 fragColor;
    
    void main() {
        // ЯРКИЙ КРАСНЫЙ для теста!
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
    """
    
    # Создаем программу
    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader
    )
    
    # VAO
    vao = ctx.vertex_array(
        program,
        [(position_buffer, '3f', 'in_position')]
    )
    
    print(f"\n✅ Shader программа создана")
    print(f"✅ VAO создан")
    
    # === РЕНДЕРИНГ ===
    print(f"\n🎨 Начинаю рендеринг...")
    print(f"  Нажми ESC для выхода")
    print(f"  Если видишь КРАСНЫЕ точки → рендеринг работает!")
    
    clock = pygame.time.Clock()
    running = True
    frame_count = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        # Clear
        ctx.clear(1.0, 1.0, 1.0)  # БЕЛЫЙ фон для контраста
        
        # Отключаем depth test для простоты
        ctx.disable(moderngl.DEPTH_TEST)
        
        # Рендерим частицы
        vao.render(moderngl.POINTS, vertices=num_particles)
        
        # Показываем
        pygame.display.flip()
        clock.tick(60)
        
        frame_count += 1
        
        # Каждую секунду выводим info
        if frame_count % 60 == 0:
            print(f"  Frame {frame_count}: рендерим {num_particles} частиц...")
    
    print(f"\n✅ Тест завершен")
    pygame.quit()


def test_particles_with_camera():
    """Тест с camera/projection матрицами"""
    
    print("="*70)
    print("🔍 ДИАГНОСТИКА ЧАСТИЦ (с камерой)")
    print("="*70)
    
    # Инициализация
    pygame.init()
    pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    
    print(f"\n✅ OpenGL Context создан")
    
    # === СОЗДАЕМ ЧАСТИЦЫ ===
    num_particles = 100
    
    # Позиции в 3D пространстве
    positions = np.zeros((num_particles, 3), dtype='f4')
    
    for i in range(num_particles):
        positions[i] = [
            np.random.uniform(-2, 2),  # x
            np.random.uniform(-2, 2),  # y
            np.random.uniform(-2, 2)   # z
        ]
    
    print(f"\n📊 Частицы:")
    print(f"  Количество: {num_particles}")
    print(f"  Диапазон: -2 до +2 по всем осям")
    
    position_buffer = ctx.buffer(positions.tobytes())
    
    # === SHADER С МАТРИЦАМИ ===
    vertex_shader = """
    #version 330
    in vec3 in_position;
    uniform mat4 mvp;
    
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        gl_PointSize = 50.0;
    }
    """
    
    fragment_shader = """
    #version 330
    out vec4 fragColor;
    
    void main() {
        // Красный круг
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (length(coord) > 0.5) discard;
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
    """
    
    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader
    )
    
    vao = ctx.vertex_array(
        program,
        [(position_buffer, '3f', 'in_position')]
    )
    
    # === КАМЕРА ===
    aspect = 1280 / 720
    proj = Matrix44.perspective_projection(60.0, aspect, 0.1, 100.0)
    view = Matrix44.look_at(
        [0, 0, 10],  # Камера на расстоянии 10
        [0, 0, 0],   # Смотрим на центр
        [0, 1, 0]
    )
    mvp = (proj * view).astype('f4')
    
    print(f"\n📷 Камера:")
    print(f"  Позиция: [0, 0, 10]")
    print(f"  Смотрит на: [0, 0, 0]")
    print(f"  FOV: 60°")
    
    program['mvp'].write(mvp.tobytes())
    
    print(f"\n🎨 Рендеринг (с камерой)...")
    print(f"  Нажми ESC для выхода")
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        # Clear - белый фон
        ctx.clear(1.0, 1.0, 1.0)
        ctx.disable(moderngl.DEPTH_TEST)
        
        # Рендерим
        vao.render(moderngl.POINTS, vertices=num_particles)
        
        pygame.display.flip()
        clock.tick(60)
    
    print(f"\n✅ Тест завершен")
    pygame.quit()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              🔍 PARTICLE DIAGNOSTICS TOOL                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Этот скрипт проверит почему частицы не видны.

Выберите тест:
  1 - Простой тест (без камеры)
  2 - Тест с камерой
  
Или нажмите Enter для запуска обоих.
    """)
    
    choice = input("Выбор (1/2/Enter): ").strip()
    
    if choice == "1":
        test_particles()
    elif choice == "2":
        test_particles_with_camera()
    else:
        print("\n=== Тест 1: Без камеры ===")
        test_particles()
        
        print("\n\n=== Тест 2: С камерой ===")
        test_particles_with_camera()
    
    print("\n✅ Диагностика завершена!")