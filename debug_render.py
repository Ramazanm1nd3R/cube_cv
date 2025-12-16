"""
DEBUG: Рендерим ТОЛЬКО частицы
Убираем камеру, куб, debug info - всё!
Просто частицы на черном фоне
"""

import sys
sys.path.append('src')

import pygame
import moderngl
import numpy as np
from pyrr import Matrix44
from config import Config

def main():
    print("=" * 70)
    print("DEBUG: ТОЛЬКО ЧАСТИЦЫ")
    print("=" * 70)
    
    # Init
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    
    print(f"\n✅ OpenGL: {ctx.info['GL_VERSION']}")
    
    # === ЧАСТИЦЫ ===
    num_particles = 100  # Немного для теста
    
    # Позиции (vec4 как в main app)
    positions = np.zeros((num_particles, 4), dtype='f4')
    
    # Размеры контейнера
    half_w = Config.container.width / 2
    half_h = Config.container.height / 2
    half_d = Config.container.depth / 2
    
    # Заполняем (как в main app)
    positions[:, 0] = np.random.uniform(-half_w * 0.8, half_w * 0.8, num_particles)
    positions[:, 1] = np.random.uniform(-half_h * 0.9, 0, num_particles)
    positions[:, 2] = np.random.uniform(-half_d * 0.8, half_d * 0.8, num_particles)
    positions[:, 3] = 1.0
    
    print(f"\n📊 Частицы:")
    print(f"  Количество: {num_particles}")
    print(f"  Позиции X: {positions[:, 0].min():.2f} .. {positions[:, 0].max():.2f}")
    print(f"  Позиции Y: {positions[:, 1].min():.2f} .. {positions[:, 1].max():.2f}")
    print(f"  Позиции Z: {positions[:, 2].min():.2f} .. {positions[:, 2].max():.2f}")
    
    position_buffer = ctx.buffer(positions.tobytes())
    
    # === SHADER (ТОЧНО КАК В MAIN APP) ===
    vertex_shader = """
    #version 330
    in vec4 in_position;
    uniform mat4 vp;
    uniform mat4 model;
    
    void main() {
        gl_Position = vp * model * vec4(in_position.xyz, 1.0);
        gl_PointSize = 200.0;
    }
    """
    
    fragment_shader = """
    #version 330
    out vec4 fragColor;
    uniform vec4 color;
    
    void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord);
        if (dist > 0.5) discard;
        
        float brightness = 1.0 - (dist * 1.5);
        brightness = clamp(brightness, 0.3, 1.0);
        
        fragColor = vec4(color.rgb * brightness, color.a);
    }
    """
    
    program = ctx.program(
        vertex_shader=vertex_shader,
        fragment_shader=fragment_shader
    )
    
    vao = ctx.vertex_array(program, [(position_buffer, '4f', 'in_position')])
    
    print(f"\n✅ Shader создан")
    
    # === КАМЕРА (КАК В MAIN APP) ===
    aspect = 1280 / 720
    proj = Matrix44.perspective_projection(60.0, aspect, 0.1, 100.0)
    view = Matrix44.look_at([0, 0, 15], [0, 0, 0], [0, 1, 0])
    vp = (proj * view).astype('f4')
    
    # Model transform (identity)
    model = np.eye(4, dtype='f4')
    
    program['vp'].write(vp.tobytes())
    program['model'].write(model.tobytes())
    program['color'].value = (1.0, 0.0, 0.0, 1.0)  # КРАСНЫЙ
    
    print(f"\n📷 Камера:")
    print(f"  Position: [0, 0, 15]")
    print(f"  Looking at: [0, 0, 0]")
    
    # === НАСТРОЙКИ РЕНДЕРИНГА ===
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    
    print(f"\n🎨 РЕНДЕРИНГ...")
    print(f"   Должны быть видны КРАСНЫЕ ШАРЫ!")
    print(f"   Если видны → проблема в main app (камера/куб что-то затирает)")
    print(f"   Если НЕ видны → проблема в GL_POINTS")
    print(f"\n   ESC - выход")
    
    clock = pygame.time.Clock()
    running = True
    frame = 0
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
        
        # Clear - ЧЕРНЫЙ фон
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # NO depth test
        ctx.disable(moderngl.DEPTH_TEST)
        
        # Рендер частиц
        vao.render(moderngl.POINTS, vertices=num_particles)
        
        pygame.display.flip()
        clock.tick(60)
        
        frame += 1
        if frame % 60 == 0:
            print(f"   Frame {frame}: Рендерим {num_particles} частиц...")
    
    pygame.quit()
    print(f"\n✅ Тест завершен")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              🔴 DEBUG: ТОЛЬКО ЧАСТИЦЫ                        ║
║                                                              ║
║  Убираем ВСЁ кроме частиц                                   ║
║  Если видно → проблема в main app                           ║
║  Если НЕ видно → проблема в GL_POINTS                       ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТ:")
    print("=" * 70)
    
    saw = input("\nВидел КРАСНЫЕ ШАРЫ? (y/n): ").lower()
    
    if saw == 'y':
        print("\n✅ GL_POINTS РАБОТАЕТ!")
        print("\n   Значит проблема в main app:")
        print("   - Камера что-то затирает?")
        print("   - Куб перекрывает?")
        print("   - Depth test не там?")
        print("   - Blend mode неправильный?")
        print("\n   Нужно debugить main app дальше")
    else:
        print("\n❌ GL_POINTS НЕ РАБОТАЕТ!")
        print("\n   Твоя GPU/драйвер не поддерживает большие gl_PointSize")
        print("\n   РЕШЕНИЕ: Нужен QUAD рендеринг (instanced)")
        print("   Скажи мне, я создам версию с quads!")