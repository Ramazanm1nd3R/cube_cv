"""
Проверка данных частиц
Читаем буфер и смотрим что там на самом деле
"""

import sys
sys.path.append('src')

import pygame
import moderngl
import numpy as np
from config import Config
from app import GPUParticleSystem

def main():
    print("=" * 70)
    print("ПРОВЕРКА ДАННЫХ ЧАСТИЦ")
    print("=" * 70)
    
    # Init
    pygame.init()
    screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    
    print(f"\n✅ OpenGL: {ctx.info['GL_VERSION']}")
    
    # === СОЗДАЕМ PARTICLE SYSTEM ===
    print(f"\nСоздаем {Config.sph.num_particles} частиц...")
    particles = GPUParticleSystem(ctx, Config.sph.num_particles)
    
    print(f"✅ Система создана")
    
    # === ЧИТАЕМ БУФЕР ===
    print(f"\nЧитаем position buffer...")
    
    # Читаем данные из GPU
    data = particles.position_buffer.read()
    positions = np.frombuffer(data, dtype='f4').reshape(-1, 4)
    
    print(f"\n📊 АНАЛИЗ ДАННЫХ:")
    print(f"  Buffer size: {len(data)} bytes")
    print(f"  Частиц: {len(positions)}")
    print(f"  Shape: {positions.shape}")
    
    print(f"\n  X: min={positions[:, 0].min():.3f}, max={positions[:, 0].max():.3f}")
    print(f"  Y: min={positions[:, 1].min():.3f}, max={positions[:, 1].max():.3f}")
    print(f"  Z: min={positions[:, 2].min():.3f}, max={positions[:, 2].max():.3f}")
    print(f"  W: min={positions[:, 3].min():.3f}, max={positions[:, 3].max():.3f}")
    
    print(f"\n  Первые 5 частиц:")
    for i in range(min(5, len(positions))):
        print(f"    [{i}] = [{positions[i, 0]:7.3f}, {positions[i, 1]:7.3f}, {positions[i, 2]:7.3f}, {positions[i, 3]:7.3f}]")
    
    # === ПРОВЕРКА НА ОШИБКИ ===
    print(f"\n🔍 ПРОВЕРКА НА ПРОБЛЕМЫ:")
    
    errors = []
    
    # Все нули?
    if np.all(positions[:, :3] == 0):
        errors.append("❌ ВСЕ ПОЗИЦИИ = 0! Частицы не созданы!")
    else:
        print(f"  ✅ Позиции не нулевые")
    
    # NaN или Inf?
    if np.any(np.isnan(positions)):
        errors.append("❌ Есть NaN значения!")
    else:
        print(f"  ✅ Нет NaN")
    
    if np.any(np.isinf(positions)):
        errors.append("❌ Есть Inf значения!")
    else:
        print(f"  ✅ Нет Inf")
    
    # W component = 1?
    if not np.all(positions[:, 3] == 1.0):
        errors.append("⚠️ W компонент не всегда = 1.0")
    else:
        print(f"  ✅ W компонент = 1.0")
    
    # Разумные границы?
    half_w = Config.container.width / 2
    half_h = Config.container.height / 2
    half_d = Config.container.depth / 2
    
    if np.any(positions[:, 0] < -half_w * 2) or np.any(positions[:, 0] > half_w * 2):
        errors.append("⚠️ X выходит за границы!")
    else:
        print(f"  ✅ X в пределах контейнера")
    
    if np.any(positions[:, 1] < -half_h * 2) or np.any(positions[:, 1] > half_h * 2):
        errors.append("⚠️ Y выходит за границы!")
    else:
        print(f"  ✅ Y в пределах контейнера")
    
    if np.any(positions[:, 2] < -half_d * 2) or np.any(positions[:, 2] > half_d * 2):
        errors.append("⚠️ Z выходит за границы!")
    else:
        print(f"  ✅ Z в пределах контейнера")
    
    # === СИМУЛИРУЕМ 1 ШАГ ФИЗИКИ ===
    print(f"\n⏱️ Симулируем 1 шаг физики...")
    
    container_transform = np.eye(4, dtype='f4')
    particles.update(1.0 / 60.0, container_transform)
    
    # Читаем опять
    data_after = particles.position_buffer.read()
    positions_after = np.frombuffer(data_after, dtype='f4').reshape(-1, 4)
    
    print(f"\n📊 ПОСЛЕ ФИЗИКИ:")
    print(f"  X: min={positions_after[:, 0].min():.3f}, max={positions_after[:, 0].max():.3f}")
    print(f"  Y: min={positions_after[:, 1].min():.3f}, max={positions_after[:, 1].max():.3f}")
    print(f"  Z: min={positions_after[:, 2].min():.3f}, max={positions_after[:, 2].max():.3f}")
    
    # Изменились ли позиции?
    diff = np.abs(positions_after - positions).max()
    print(f"\n  Максимальное изменение: {diff:.6f}")
    
    if diff < 0.0001:
        errors.append("⚠️ Позиции НЕ изменились! Физика не работает?")
    else:
        print(f"  ✅ Позиции изменились, физика работает")
    
    # === ИТОГ ===
    print(f"\n" + "=" * 70)
    print("ИТОГ:")
    print("=" * 70)
    
    if errors:
        print("\n❌ НАЙДЕНЫ ПРОБЛЕМЫ:")
        for err in errors:
            print(f"  {err}")
    else:
        print("\n✅ ВСЕ В ПОРЯДКЕ С ДАННЫМИ!")
        print("\n  Частицы созданы правильно")
        print("  Позиции в разумных пределах")
        print("  Физика работает")
        print("\n  → Проблема скорее всего в РЕНДЕРИНГЕ!")
        print("  → Запусти debug_render_only_particles.py")
    
    pygame.quit()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              🔍 ПРОВЕРКА ДАННЫХ ЧАСТИЦ                       ║
║                                                              ║
║  Читаем буфер и смотрим что там                             ║
║  Проверяем что частицы создались правильно                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Проверка завершена")