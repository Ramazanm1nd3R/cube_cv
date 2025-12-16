"""
Простейший тест частиц - БЕЗ физики, БЕЗ камеры, БЕЗ всего лишнего
Просто рисуем точки на экране
"""
import pygame
import moderngl
import numpy as np
from pyrr import Matrix44

def simple_test():
    """Самый простой возможный тест"""
    
    print("=" * 70)
    print("🔴 SIMPLE PARTICLE TEST")
    print("=" * 70)
    
    # Pygame + OpenGL
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    
    print(f"\n✅ OpenGL: {ctx.info['GL_VERSION']}")
    
    # === ЧАСТИЦЫ ===
    num_particles = 10  # Всего 10 для теста
    
    # Позиции - ПРЯМО В ЦЕНТРЕ ЭКРАНА
    positions = np.array([
        # x     y     z
        [0.0,  0.0,  0.0],   # Центр
        [0.2,  0.0,  0.0],   # Справа
        [-0.2, 0.0,  0.0],   # Слева
        [0.0,  0.2,  0.0],   # Сверху
        [0.0, -0.2,  0.0],   # Снизу
        [0.2,  0.2,  0.0],   # Правый верх
        [-0.2, 0.2,  0.0],   # Левый верх
        [0.2, -0.2,  0.0],   # Правый низ
        [-0.2,-0.2,  0.0],   # Левый низ
        [0.0,  0.0,  0.0],   # Дубль центра
    ], dtype='f4')
    
    print(f"\n📊 Создано {num_particles} частиц")
    print(f"   Позиции: от -0.2 до +0.2 (центр экрана)")
    
    # GPU buffer
    vbo = ctx.buffer(positions.tobytes())
    
    # === SHADER (МАКСИМАЛЬНО ПРОСТОЙ) ===
    vertex_shader = """
    #version 330
    in vec3 in_position;
    
    void main() {
        gl_Position = vec4(in_position, 1.0);
        gl_PointSize = 100.0;  // ОГРОМНЫЙ!
    }
    """
    
    fragment_shader = """
    #version 330
    out vec4 fragColor;
    
    void main() {
        // КРАСНЫЙ круг
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (length(coord) > 0.5) discard;
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
    }
    """
    
    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader
    )
    
    vao = ctx.vertex_array(program, [(vbo, '3f', 'in_position')])
    
    print(f"\n✅ Shader создан")
    print(f"   Point size: 100px")
    print(f"   Цвет: КРАСНЫЙ")
    
    # === РЕНДЕРИНГ ===
    print(f"\n🎨 Рендеринг...")
    print(f"   Ты ДОЛЖЕН видеть 10 КРАСНЫХ КРУГОВ в центре экрана!")
    print(f"   Если не видишь - проблема в OpenGL/драйверах")
    print(f"\n   ESC - выход")
    
    clock = pygame.time.Clock()
    running = True
    frame = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        # Clear - БЕЛЫЙ фон
        ctx.clear(1.0, 1.0, 1.0, 1.0)
        
        # Рендер частиц
        vao.render(moderngl.POINTS, vertices=num_particles)
        
        # Flip
        pygame.display.flip()
        clock.tick(60)
        
        frame += 1
        if frame == 60:
            print(f"\n   Frame 60: Рендерим {num_particles} частиц...")
            print(f"   Видишь красные круги? (должно быть ОЧЕВИДНО)")
    
    pygame.quit()
    print(f"\n✅ Тест завершен")


def test_with_camera():
    """Тест с камерой и трансформациями"""
    
    print("\n" + "=" * 70)
    print("🔴 TEST WITH CAMERA")
    print("=" * 70)
    
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    
    print(f"\n✅ OpenGL: {ctx.info['GL_VERSION']}")
    
    # === ЧАСТИЦЫ В 3D ===
    num_particles = 50
    
    # Позиции в 3D пространстве (куб)
    positions = np.zeros((num_particles, 3), dtype='f4')
    for i in range(num_particles):
        positions[i] = [
            np.random.uniform(-1, 1),  # x
            np.random.uniform(-1, 1),  # y
            np.random.uniform(-1, 1),  # z
        ]
    
    print(f"\n📊 Создано {num_particles} частиц в кубе 2x2x2")
    
    vbo = ctx.buffer(positions.tobytes())
    
    # === SHADER С МАТРИЦАМИ ===
    vertex_shader = """
    #version 330
    in vec3 in_position;
    uniform mat4 mvp;
    
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        gl_PointSize = 80.0;
    }
    """
    
    fragment_shader = """
    #version 330
    out vec4 fragColor;
    
    void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (length(coord) > 0.5) discard;
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);  // КРАСНЫЙ
    }
    """
    
    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader
    )
    
    vao = ctx.vertex_array(program, [(vbo, '3f', 'in_position')])
    
    # === КАМЕРА ===
    aspect = 1280 / 720
    proj = Matrix44.perspective_projection(60.0, aspect, 0.1, 100.0)
    view = Matrix44.look_at(
        [0, 0, 5],   # Камера на расстоянии 5
        [0, 0, 0],   # Смотрит в центр
        [0, 1, 0]
    )
    mvp = (proj * view).astype('f4')
    
    program['mvp'].write(mvp.tobytes())
    
    print(f"\n📷 Камера:")
    print(f"   Position: [0, 0, 5]")
    print(f"   Looking at: [0, 0, 0]")
    print(f"   FOV: 60°")
    
    print(f"\n🎨 Рендеринг с камерой...")
    print(f"   Должно быть видно облако красных точек")
    print(f"\n   ESC - выход")
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        # Clear - БЕЛЫЙ
        ctx.clear(1.0, 1.0, 1.0, 1.0)
        
        # Рендер
        vao.render(moderngl.POINTS, vertices=num_particles)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print(f"\n✅ Тест завершен")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              🔴 SIMPLE PARTICLE TEST                         ║
║                                                              ║
║  Минимальный тест - проверим работает ли рендеринг вообще   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Выбери тест:
  1 - Простейший (2D, без камеры)
  2 - С камерой (3D)
  
Enter - оба
    """)
    
    choice = input("Выбор: ").strip()
    
    if choice == "1":
        simple_test()
    elif choice == "2":
        test_with_camera()
    else:
        print("\n=== Тест 1: Простейший ===")
        simple_test()
        
        input("\nНажми Enter для теста 2...")
        
        print("\n=== Тест 2: С камерой ===")
        test_with_camera()
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 70)
    print("""
Если в ТЕСТЕ 1 видел красные круги:
  ✅ OpenGL работает
  ✅ Рендеринг точек работает
  ✅ Проблема в main app

Если в ТЕСТЕ 1 НЕ видел:
  ❌ Проблема в OpenGL/драйверах
  ❌ Возможно GPU не поддерживает точки большого размера
  
Если в ТЕСТЕ 2 видел красные круги:
  ✅ Камера работает
  ✅ Трансформации работают
  ✅ Проблема где-то еще в main app

Отправь результаты!
    """)