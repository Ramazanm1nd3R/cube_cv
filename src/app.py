"""
App - полностью переработанное приложение
GPU ускорение + правильная архитектура БЕЗ КОСТЫЛЕЙ
"""
import pygame
import moderngl
import numpy as np
import cv2
import time
from pathlib import Path

from config import Config
from hand_tracking import HandTracker, HandTransform
from renderer import create_transform_matrix


class GPUParticleSystem:
    """
    Particle System на GPU с compute shaders
    Вся физика SPH выполняется на GPU для максимальной производительности
    """
    
    def __init__(self, ctx: moderngl.Context, num_particles: int):
        self.ctx = ctx
        self.num_particles = num_particles
        
        print(f"[GPUParticleSystem] Инициализация {num_particles} частиц на GPU...")
        
        # === Буферы на GPU ===
        # Позиции частиц [x, y, z, _padding]
        initial_positions = self._generate_initial_positions()
        self.position_buffer = ctx.buffer(initial_positions.tobytes())
        
        # Скорости частиц [vx, vy, vz, _padding]
        initial_velocities = np.zeros((num_particles, 4), dtype='f4')
        self.velocity_buffer = ctx.buffer(initial_velocities.tobytes())
        
        # Плотности и давления [density, pressure, _p1, _p2]
        self.density_buffer = ctx.buffer(reserve=num_particles * 16)
        
        # === Compute Shader для физики SPH ===
        self.compute_shader = self._create_compute_shader()
        
        print("[GPUParticleSystem] ✅ Инициализация завершена")
    
    def _generate_initial_positions(self) -> np.ndarray:
        """Генерируем начальные позиции частиц"""
        positions = np.zeros((self.num_particles, 4), dtype='f4')
        
        # Размещаем в кубе
        width = Config.container.width / 4
        height = Config.container.height / 4
        depth = Config.container.depth / 4
        
        positions[:, 0] = np.random.uniform(-width, width, self.num_particles)   # x
        positions[:, 1] = np.random.uniform(height/2, height, self.num_particles) # y (сверху)
        positions[:, 2] = np.random.uniform(-depth, depth, self.num_particles)   # z
        positions[:, 3] = 1.0  # padding
        
        return positions
    
    def _create_compute_shader(self) -> moderngl.ComputeShader:
        """
        Создаем compute shader для SPH физики на GPU
        Выполняется параллельно для всех частиц
        """
        source = """
        #version 430
        
        layout(local_size_x = 256) in;
        
        // Буферы данных
        layout(std430, binding = 0) buffer Positions {
            vec4 positions[];
        };
        
        layout(std430, binding = 1) buffer Velocities {
            vec4 velocities[];
        };
        
        layout(std430, binding = 2) buffer Densities {
            vec4 densities[];  // [density, pressure, _, _]
        };
        
        // Uniform параметры
        uniform float dt;
        uniform int num_particles;
        uniform vec3 gravity;
        uniform float particle_mass;
        uniform float rest_density;
        uniform float gas_constant;
        uniform float viscosity;
        uniform float smoothing_radius;
        uniform float damping;
        uniform vec3 container_min;
        uniform vec3 container_max;
        
        // SPH Kernel functions
        float poly6_kernel(float r, float h) {
            if (r >= 0.0 && r <= h) {
                float factor = 315.0 / (64.0 * 3.14159 * pow(h, 9.0));
                return factor * pow(h * h - r * r, 3.0);
            }
            return 0.0;
        }
        
        float spiky_kernel_gradient(float r, float h) {
            if (r >= 0.0 && r <= h && r > 0.0001) {
                float factor = -45.0 / (3.14159 * pow(h, 6.0));
                return factor * pow(h - r, 2.0) / r;
            }
            return 0.0;
        }
        
        float viscosity_kernel_laplacian(float r, float h) {
            if (r >= 0.0 && r <= h) {
                float factor = 45.0 / (3.14159 * pow(h, 6.0));
                return factor * (h - r);
            }
            return 0.0;
        }
        
        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= num_particles) return;
            
            vec3 pos_i = positions[i].xyz;
            vec3 vel_i = velocities[i].xyz;
            
            // === 1. COMPUTE DENSITY ===
            float density = 0.0;
            for (uint j = 0; j < num_particles; j++) {
                vec3 pos_j = positions[j].xyz;
                float r = length(pos_i - pos_j);
                density += particle_mass * poly6_kernel(r, smoothing_radius);
            }
            density = max(density, rest_density);
            
            // === 2. COMPUTE PRESSURE ===
            float pressure = gas_constant * (density - rest_density);
            
            // Сохраняем плотность и давление
            densities[i] = vec4(density, pressure, 0.0, 0.0);
            
            // === 3. COMPUTE FORCES ===
            vec3 force_pressure = vec3(0.0);
            vec3 force_viscosity = vec3(0.0);
            
            for (uint j = 0; j < num_particles; j++) {
                if (i == j) continue;
                
                vec3 pos_j = positions[j].xyz;
                vec3 r_vec = pos_i - pos_j;
                float r = length(r_vec);
                
                if (r < 0.0001) continue;
                
                vec3 r_normalized = r_vec / r;
                float density_j = densities[j].x;
                float pressure_j = densities[j].y;
                vec3 vel_j = velocities[j].xyz;
                
                // Pressure force
                float pressure_term = (pressure + pressure_j) / (2.0 * density_j);
                float spiky_grad = spiky_kernel_gradient(r, smoothing_radius);
                force_pressure -= particle_mass * pressure_term * spiky_grad * r_normalized;
                
                // Viscosity force
                vec3 vel_diff = vel_j - vel_i;
                float visc_lap = viscosity_kernel_laplacian(r, smoothing_radius);
                force_viscosity += viscosity * particle_mass * (vel_diff / density_j) * visc_lap;
            }
            
            // Gravity force
            vec3 force_gravity = gravity * density;
            
            // Total force
            vec3 force_total = force_pressure + force_viscosity + force_gravity;
            
            // === 4. INTEGRATE (Semi-Implicit Euler) ===
            vec3 acceleration = force_total / density;
            vel_i += acceleration * dt;
            vel_i *= (1.0 - damping);  // Damping
            
            // Limit velocity
            float speed = length(vel_i);
            if (speed > 100.0) {
                vel_i *= 100.0 / speed;
            }
            
            pos_i += vel_i * dt;
            
            // === 5. COLLISION WITH CONTAINER ===
            float restitution = 0.3;
            
            // X bounds
            if (pos_i.x < container_min.x) {
                pos_i.x = container_min.x;
                vel_i.x *= -restitution;
            }
            if (pos_i.x > container_max.x) {
                pos_i.x = container_max.x;
                vel_i.x *= -restitution;
            }
            
            // Y bounds
            if (pos_i.y < container_min.y) {
                pos_i.y = container_min.y;
                vel_i.y *= -restitution;
            }
            if (pos_i.y > container_max.y) {
                pos_i.y = container_max.y;
                vel_i.y *= -restitution;
            }
            
            // Z bounds
            if (pos_i.z < container_min.z) {
                pos_i.z = container_min.z;
                vel_i.z *= -restitution;
            }
            if (pos_i.z > container_max.z) {
                pos_i.z = container_max.z;
                vel_i.z *= -restitution;
            }
            
            // Обновляем буферы
            positions[i] = vec4(pos_i, 1.0);
            velocities[i] = vec4(vel_i, 0.0);
        }
        """
        
        return self.ctx.compute_shader(source)
    
    def update(self, dt: float, container_transform: np.ndarray):
        """
        Обновляем физику на GPU
        Compute shader выполняется параллельно для всех частиц
        """
        # Bind буферы
        self.position_buffer.bind_to_storage_buffer(0)
        self.velocity_buffer.bind_to_storage_buffer(1)
        self.density_buffer.bind_to_storage_buffer(2)
        
        # Устанавливаем uniforms
        self.compute_shader['dt'].value = dt
        self.compute_shader['num_particles'].value = self.num_particles
        self.compute_shader['gravity'].value = tuple(Config.sph.gravity)
        self.compute_shader['particle_mass'].value = Config.sph.particle_mass
        self.compute_shader['rest_density'].value = Config.sph.rest_density
        self.compute_shader['gas_constant'].value = Config.sph.gas_constant
        self.compute_shader['viscosity'].value = Config.sph.viscosity
        self.compute_shader['smoothing_radius'].value = Config.sph.smoothing_radius
        self.compute_shader['damping'].value = Config.sph.damping
        
        # Container bounds
        half_w = Config.container.width / 2
        half_h = Config.container.height / 2
        half_d = Config.container.depth / 2
        self.compute_shader['container_min'].value = (-half_w, -half_h, -half_d)
        self.compute_shader['container_max'].value = (half_w, half_h, half_d)
        
        # Запускаем compute shader
        # 256 частиц на work group, округляем вверх
        num_groups = (self.num_particles + 255) // 256
        self.compute_shader.run(num_groups)
        
        # Синхронизация GPU
        self.ctx.memory_barrier()
    
    def get_positions(self) -> np.ndarray:
        """Читаем позиции с GPU (только для рендеринга)"""
        data = self.position_buffer.read()
        positions = np.frombuffer(data, dtype='f4').reshape(-1, 4)
        return positions[:, :3]  # Только xyz


class FluidGestureApp:
    """
    Главное приложение - полностью переработанное
    GPU ускорение + pygame window + правильная архитектура
    """
    
    def __init__(self):
        print("=" * 70)
        print("🌊 Fluid Gesture Simulator - GPU Accelerated Version")
        print("=" * 70)
        Config.print_summary()
        
        # === 1. Pygame + OpenGL ===
        print("\n[1/4] Инициализация Pygame + OpenGL...")
        pygame.init()
        
        # OpenGL 4.3+ для compute shaders
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, 
                                       pygame.GL_CONTEXT_PROFILE_CORE)
        
        self.window_size = (Config.window.width, Config.window.height)
        self.screen = pygame.display.set_mode(
            self.window_size,
            pygame.OPENGL | pygame.DOUBLEBUF
        )
        pygame.display.set_caption(Config.window.title + " - GPU Accelerated")
        
        # ModernGL context
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        print(f"✅ OpenGL {self.ctx.version_code // 100}.{self.ctx.version_code % 100}")
        
        # === 2. Hand Tracking ===
        print("\n[2/4] Инициализация Hand Tracking...")
        self.hand_tracker = HandTracker()
        
        # === 3. GPU Particle System ===
        print("\n[3/4] Инициализация GPU Particle System...")
        self.particles = GPUParticleSystem(self.ctx, Config.sph.num_particles)
        
        # === 4. Рендеринг ===
        print("\n[4/4] Инициализация Renderer...")
        self._setup_renderer()
        self._setup_camera_overlay()
        
        # === Состояние ===
        self.running = True
        self.paused = False
        self.show_camera = True
        self.show_debug = True
        
        # Container
        self.container_transform = create_transform_matrix(
            Config.container.initial_position,
            Config.container.initial_rotation
        )
        self.smooth_position = Config.container.initial_position.copy()
        self.smooth_rotation = Config.container.initial_rotation.copy()
        self.current_hand_transform = None  # Для debug info
        self.hand2_transform = None  # Вторая рука (если есть)
        
        # Two-hand gesture (pinch-to-zoom)
        self.two_hand_mode = False
        self.initial_two_hand_distance = None
        self.initial_scale = 1.0
        self.current_scale = 1.0  # Текущий scale контейнера
        
        # Timing
        self.clock = pygame.time.Clock()
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Physics
        self.physics_dt = 1.0 / 60.0  # Fixed 60 Hz physics
        self.physics_accumulator = 0.0
        
        print("\n" + "=" * 70)
        print("✅ Инициализация завершена! GPU ускорение включено!")
        print("=" * 70)
        print("\n🎮 Управление:")
        print("  SPACE - Пауза")
        print("  C     - Камера вкл/выкл")
        print("  D     - Debug info")
        print("  R     - Сброс")
        print("  ESC   - Выход")
        print("\n👋 Покажите руку перед камерой!\n")
    
    def _setup_renderer(self):
        """Настройка 3D рендеринга"""
        from pyrr import Matrix44
        
        # View-Projection matrix
        aspect = self.window_size[0] / self.window_size[1]
        proj = Matrix44.perspective_projection(60.0, aspect, 0.1, 100.0)
        view = Matrix44.look_at([0, 0, 10], [0, 0, 0], [0, 1, 0])
        self.vp_matrix = (proj * view).astype('f4')
        
        # === Particle Renderer ===
        particle_vertex = """
        #version 330
        in vec3 in_position;
        uniform mat4 vp;
        uniform mat4 model;
        
        void main() {
            gl_Position = vp * model * vec4(in_position, 1.0);
            gl_PointSize = 8.0;
        }
        """
        
        particle_fragment = """
        #version 330
        out vec4 fragColor;
        uniform vec4 color;
        
        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            if (length(coord) > 0.5) discard;
            fragColor = color;
        }
        """
        
        self.particle_program = self.ctx.program(
            vertex_shader=particle_vertex,
            fragment_shader=particle_fragment
        )
        
        # VAO для частиц (используем position_buffer из GPU)
        self.particle_vao = self.ctx.vertex_array(
            self.particle_program,
            [(self.particles.position_buffer, '4f', 'in_position')]
        )
        
        # === Container Renderer ===
        container_vertex = """
        #version 330
        in vec3 in_position;
        uniform mat4 vp;
        uniform mat4 model;
        
        void main() {
            gl_Position = vp * model * vec4(in_position, 1.0);
        }
        """
        
        container_fragment = """
        #version 330
        out vec4 fragColor;
        uniform vec4 color;
        
        void main() {
            fragColor = color;
        }
        """
        
        self.container_program = self.ctx.program(
            vertex_shader=container_vertex,
            fragment_shader=container_fragment
        )
        
        # Container geometry (cube wireframe)
        w, h, d = Config.container.width/2, Config.container.height/2, Config.container.depth/2
        vertices = np.array([
            -w,-h,-d,  w,-h,-d,  w,-h, d, -w,-h, d,  # bottom
            -w, h,-d,  w, h,-d,  w, h, d, -w, h, d,  # top
        ], dtype='f4')
        
        indices = np.array([
            0,1, 1,2, 2,3, 3,0,  # bottom
            4,5, 5,6, 6,7, 7,4,  # top
            0,4, 1,5, 2,6, 3,7,  # vertical
        ], dtype='i4')
        
        vbo = self.ctx.buffer(vertices)
        ibo = self.ctx.buffer(indices)
        self.container_vao = self.ctx.vertex_array(
            self.container_program,
            [(vbo, '3f', 'in_position')],
            ibo
        )
        self.container_indices_count = len(indices)
    
    def _setup_camera_overlay(self):
        """
        Настройка камеры на ВЕСЬ ЭКРАН как фон
        3D элементы рендерятся поверх камеры
        """
        vertex = """
        #version 330
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 uv;
        
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            uv = in_uv;
        }
        """
        
        fragment = """
        #version 330
        uniform sampler2D tex;
        in vec2 uv;
        out vec4 fragColor;
        
        void main() {
            fragColor = texture(tex, uv);
        }
        """
        
        self.camera_program = self.ctx.program(vertex_shader=vertex, fragment_shader=fragment)
        
        # Quad на ВЕСЬ ЭКРАН (от -1 до 1 в NDC)
        vertices = np.array([
            # pos          uv
            -1.0, -1.0,    0, 0,  # bottom-left
            1.0, -1.0,     1, 0,  # bottom-right
            -1.0, 1.0,     0, 1,  # top-left
            1.0, 1.0,      1, 1,  # top-right
        ], dtype='f4')
        
        vbo = self.ctx.buffer(vertices)
        self.camera_vao = self.ctx.vertex_array(
            self.camera_program,
            [(vbo, '2f 2f', 'in_pos', 'in_uv')]
        )
        
        self.camera_texture = None
        
        # === Setup для текста (debug info) ===
        # Будем рисовать текст через pygame surface → texture
        self.font = pygame.font.Font(None, 36)  # Размер шрифта
        self.font_small = pygame.font.Font(None, 24)
        
        # Текстура для debug overlay
        self.text_texture = None
        self._setup_text_overlay()
    
    def _setup_text_overlay(self):
        """Setup для рендеринга текста поверх всего"""
        # Используем тот же shader что и для камеры
        self.text_surface = pygame.Surface((self.window_size[0], self.window_size[1]), pygame.SRCALPHA)
    
    def _update_camera_texture(self):
        """Обновляем текстуру камеры"""
        frame = self.hand_tracker.get_frame()
        if frame is None:
            return
        
        # Resize до размера окна для fullscreen
        frame = cv2.resize(frame, self.window_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 0)
        
        if self.camera_texture is None:
            self.camera_texture = self.ctx.texture(self.window_size, 3, frame.tobytes())
            self.camera_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            self.camera_texture.write(frame.tobytes())
    
    def update(self, dt: float):
        """Update логика с поддержкой одной и двух рук"""
        # Читаем камеру
        if not self.hand_tracker.read_frame():
            return
        
        # Hand tracking (возвращает список рук или None)
        hand_transforms = self.hand_tracker.process_hand()
        
        if hand_transforms is not None and len(hand_transforms) > 0:
            # === ОДНА РУКА: просто двигаем куб ===
            if len(hand_transforms) == 1:
                hand = hand_transforms[0]
                
                # Smoothing позиции и ротации
                alpha = 0.3
                self.smooth_position = alpha * hand.position + (1-alpha) * self.smooth_position
                self.smooth_rotation = alpha * hand.rotation + (1-alpha) * self.smooth_rotation
                
                # Обновляем transform с текущим scale
                self.container_transform = create_transform_matrix(
                    self.smooth_position, 
                    self.smooth_rotation,
                    scale=self.current_scale
                )
                
                # Сохраняем для debug
                self.current_hand_transform = hand
                self.two_hand_mode = False
            
            # === ДВЕ РУКИ: pinch-to-zoom ===
            elif len(hand_transforms) == 2:
                hand1 = hand_transforms[0]
                hand2 = hand_transforms[1]
                
                # Вычисляем расстояние между руками
                distance = np.linalg.norm(hand1.position - hand2.position)
                
                # Если это первый кадр с двумя руками - запоминаем начальное расстояние
                if not self.two_hand_mode or self.initial_two_hand_distance is None:
                    self.initial_two_hand_distance = distance
                    self.initial_scale = self.current_scale
                    self.two_hand_mode = True
                
                # Вычисляем новый scale на основе изменения расстояния
                scale_factor = distance / self.initial_two_hand_distance
                new_scale = self.initial_scale * scale_factor
                
                # Ограничиваем scale (0.2x - 5x)
                new_scale = np.clip(new_scale, 0.2, 5.0)
                
                # Smoothing для scale
                self.current_scale = 0.7 * self.current_scale + 0.3 * new_scale
                
                # Позиция - среднее между двумя руками
                center_position = (hand1.position + hand2.position) / 2.0
                
                # Smoothing позиции
                alpha = 0.3
                self.smooth_position = alpha * center_position + (1-alpha) * self.smooth_position
                
                # Ротация - от первой руки
                self.smooth_rotation = alpha * hand1.rotation + (1-alpha) * self.smooth_rotation
                
                # Обновляем transform с новым scale
                self.container_transform = create_transform_matrix(
                    self.smooth_position,
                    self.smooth_rotation,
                    scale=self.current_scale
                )
                
                # Сохраняем для debug
                self.current_hand_transform = hand1  # Показываем первую руку
                self.hand2_transform = hand2
        
        # Physics (fixed timestep)
        if not self.paused:
            self.physics_accumulator += dt
            while self.physics_accumulator >= self.physics_dt:
                self.particles.update(self.physics_dt, self.container_transform)
                self.physics_accumulator -= self.physics_dt
        
        # Camera texture (fullscreen)
        if self.show_camera:
            self._update_camera_texture()
        
        # FPS
        self.frame_count += 1
        if time.time() - self.last_fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_fps_time = time.time()
            if self.show_debug:
                print(f"FPS: {self.fps} | Particles: {self.particles.num_particles} (GPU) | Scale: {self.current_scale:.2f}x")
    
    def _render_debug_info(self):
        """Рендерим debug информацию на экране"""
        if not self.show_debug:
            return
        
        # Очищаем surface
        self.text_surface.fill((0, 0, 0, 0))
        
        y_offset = 20
        line_height = 40
        
        # Заголовок
        text = self.font.render("Fluid Particle Simulation", True, (0, 255, 0))
        self.text_surface.blit(text, (10, y_offset))
        y_offset += line_height
        
        # FPS
        text = self.font_small.render(f"FPS: {self.fps}", True, (0, 255, 0))
        self.text_surface.blit(text, (10, y_offset))
        y_offset += 30
        
        # Режим (1 рука / 2 руки)
        if self.two_hand_mode:
            mode_text = f"MODE: TWO HANDS (Pinch-to-Zoom)"
            mode_color = (255, 255, 0)  # Желтый
        else:
            mode_text = f"MODE: ONE HAND (Move)"
            mode_color = (0, 255, 0)  # Зеленый
        
        text = self.font_small.render(mode_text, True, mode_color)
        self.text_surface.blit(text, (10, y_offset))
        y_offset += 30
        
        # Scale
        text = self.font_small.render(f"SCALE: {self.current_scale:.2f}x", True, (0, 255, 255))
        self.text_surface.blit(text, (10, y_offset))
        y_offset += 40
        
        # Параметры
        params = [
            f"radius: {Config.sph.smoothing_radius:.3f}",
            f"dfriction: {Config.sph.damping:.3f}",
            f"sfriction: {Config.sph.damping:.3f}",
            f"pfriction: {Config.sph.damping:.3f}",
            f"rest: {Config.sph.restitution:.3f}",
            f"adhesion: 0.0",
            f"sleepthresh: 0.0",
            f"clampspeed: 0",
            f"maxspeed: {Config.sph.max_speed:.1f}",
            f"clampaccel: 1",
            f"maxaccel: 100.0",
            f"diss: 0.0",
            f"damping: {Config.sph.damping:.3f}",
            f"cohesion: {Config.sph.cohesion:.3f}",
            f"surftension: 0.0",
            f"viscosity: {Config.sph.viscosity:.3f}",
            f"buoyancy: 0.0",
            f"colldist: 0.1",
            f"scollmargin: 0.1",
            f"smoothing: 0.0",
            f"vortconf: 90.0",
        ]
        
        for param in params:
            text = self.font_small.render(param, True, (200, 200, 200))
            self.text_surface.blit(text, (10, y_offset))
            y_offset += 25
        
        # Hand tracking info - ПРАВАЯ СТОРОНА
        if self.current_hand_transform is not None:
            y_offset = 20
            x_offset = self.window_size[0] - 450
            
            pos = self.current_hand_transform.position
            rot = self.current_hand_transform.rotation
            
            # Заголовок руки 1
            text = self.font_small.render("=== HAND 1 ===", True, (0, 255, 255))
            self.text_surface.blit(text, (x_offset, y_offset))
            y_offset += 30
            
            hand_info = [
                f"ROTATION: {np.degrees(rot[1]):.1f}°",
                f"POSITION_X: {pos[0]:.2f}",
                f"POSITION_Y: {pos[1]:.2f}",
                f"POSITION_Z: {pos[2]:.2f}",
            ]
            
            for info in hand_info:
                text = self.font_small.render(info, True, (0, 255, 0))
                self.text_surface.blit(text, (x_offset, y_offset))
                y_offset += 30
            
            # Если две руки - показываем вторую
            if self.hand2_transform is not None:
                y_offset += 20
                
                # Заголовок руки 2
                text = self.font_small.render("=== HAND 2 ===", True, (255, 255, 0))
                self.text_surface.blit(text, (x_offset, y_offset))
                y_offset += 30
                
                pos2 = self.hand2_transform.position
                rot2 = self.hand2_transform.rotation
                
                hand2_info = [
                    f"ROTATION: {np.degrees(rot2[1]):.1f}°",
                    f"POSITION_X: {pos2[0]:.2f}",
                    f"POSITION_Y: {pos2[1]:.2f}",
                    f"POSITION_Z: {pos2[2]:.2f}",
                ]
                
                for info in hand2_info:
                    text = self.font_small.render(info, True, (255, 255, 0))
                    self.text_surface.blit(text, (x_offset, y_offset))
                    y_offset += 30
                
                # Расстояние между руками
                y_offset += 10
                distance = np.linalg.norm(pos - pos2)
                text = self.font_small.render(f"DISTANCE: {distance:.2f}", True, (255, 0, 255))
                self.text_surface.blit(text, (x_offset, y_offset))
        
        # Создаем текстуру из surface
        text_data = pygame.image.tostring(self.text_surface, "RGBA", True)
        
        if self.text_texture is None:
            self.text_texture = self.ctx.texture(self.window_size, 4, text_data)
            self.text_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            self.text_texture.write(text_data)
    
    
    def render(self):
        """
        Рендеринг в правильном порядке:
        1. Камера на фоне (fullscreen)
        2. 3D элементы поверх камеры
        3. Debug текст сверху всего
        """
        # Clear
        self.ctx.clear(0, 0, 0, 1)
        
        # === 1. КАМЕРА НА ФОНЕ (fullscreen) ===
        if self.show_camera and self.camera_texture:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.camera_texture.use(0)
            self.camera_program['tex'].value = 0
            self.camera_vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.enable(moderngl.DEPTH_TEST)
        
        # === 2. 3D ЭЛЕМЕНТЫ ПОВЕРХ КАМЕРЫ ===
        # Включаем blending для прозрачности
        self.ctx.enable(moderngl.BLEND)
        
        # Container (с прозрачностью)
        self.container_program['vp'].write(self.vp_matrix.tobytes())
        self.container_program['model'].write(self.container_transform.tobytes())
        self.container_program['color'].value = Config.container.wireframe_color
        self.container_vao.render(moderngl.LINES, vertices=self.container_indices_count)
        
        # Particles (с прозрачностью)
        self.particle_program['vp'].write(self.vp_matrix.tobytes())
        self.particle_program['model'].write(self.container_transform.tobytes())
        self.particle_program['color'].value = Config.render.particle_color
        self.particle_vao.render(moderngl.POINTS, vertices=self.particles.num_particles)
        
        # === 3. DEBUG INFO СВЕРХУ ВСЕГО ===
        if self.show_debug:
            self._render_debug_info()
            
            if self.text_texture:
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.text_texture.use(0)
                self.camera_program['tex'].value = 0
                self.camera_vao.render(moderngl.TRIANGLE_STRIP)
                self.ctx.enable(moderngl.DEPTH_TEST)
        
        pygame.display.flip()
    
    def handle_events(self):
        """События"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"\n{'⏸️  ПАУЗА' if self.paused else '▶️  ЗАПУСК'}")
                elif event.key == pygame.K_c:
                    self.show_camera = not self.show_camera
                    print(f"\n📷 Камера: {'вкл' if self.show_camera else 'выкл'}")
                elif event.key == pygame.K_d:
                    self.show_debug = not self.show_debug
                    print(f"\n🐛 Debug: {'вкл' if self.show_debug else 'выкл'}")
                elif event.key == pygame.K_r:
                    print("\n🔄 Сброс...")
                    self.particles = GPUParticleSystem(self.ctx, Config.sph.num_particles)
                    self._setup_renderer()
    
    def run(self):
        """Main loop"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            self.update(dt)
            self.render()
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup"""
        print("\n🧹 Очистка...")
        self.hand_tracker.release()
        pygame.quit()
        print("✅ Готово!")


def run_app():
    """Entry point"""
    try:
        app = FluidGestureApp()
        app.run()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_app()