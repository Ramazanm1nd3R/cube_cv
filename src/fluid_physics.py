import numpy as np
from numba import jit
from typing import Tuple
from config import Config


class ParticleSystem:
    """
    Система частиц жидкости
    Хранит все данные о частицах: позиции, скорости, плотности и т.д.
    """
    
    def __init__(self, num_particles: int):
        self.num_particles = num_particles
        
        # === Основные свойства частиц ===
        self.positions = np.zeros((num_particles, 3), dtype=np.float32)  # [x, y, z]
        self.velocities = np.zeros((num_particles, 3), dtype=np.float32) # [vx, vy, vz]
        self.forces = np.zeros((num_particles, 3), dtype=np.float32)     # [fx, fy, fz]
        
        # === SPH свойства (вычисляются каждый кадр) ===
        self.densities = np.zeros(num_particles, dtype=np.float32)       # Плотность
        self.pressures = np.zeros(num_particles, dtype=np.float32)       # Давление
        
        # Инициализируем частицы в случайных позициях внутри контейнера
        self._initialize_particles()
        
        print(f"[ParticleSystem] Создано {num_particles} частиц")
    
    def _initialize_particles(self):
        """
        Инициализация частиц в кубе
        Размещаем их случайно в верхней половине контейнера
        """
        container_width = Config.container.width
        container_height = Config.container.height
        container_depth = Config.container.depth
        
        # Случайные позиции в верхней половине контейнера
        self.positions[:, 0] = np.random.uniform(-container_width/4, container_width/4, self.num_particles)   # x
        self.positions[:, 1] = np.random.uniform(container_height/4, container_height/2, self.num_particles)  # y (верх)
        self.positions[:, 2] = np.random.uniform(-container_depth/4, container_depth/4, self.num_particles)   # z
        
        # Начальные скорости = 0
        self.velocities.fill(0.0)


class SPHSolver:
    """
    SPH Solver - решатель физики жидкости
    
    Алгоритм SPH состоит из шагов:
    1. Compute Density - вычисляем плотность каждой частицы
    2. Compute Pressure - вычисляем давление из плотности
    3. Compute Forces - вычисляем силы (давление, вязкость, гравитация)
    4. Integrate - обновляем позиции и скорости
    5. Handle Collisions - обрабатываем столкновения со стенками
    """
    
    def __init__(self, particle_system: ParticleSystem):
        self.particles = particle_system
        self.config = Config.sph
        
        # Spatial grid для оптимизации поиска соседей
        self.spatial_grid = {}
        
        print(f"[SPHSolver] Инициализирован с {self.config.num_particles} частицами")
    
    def update(self, dt: float, container_transform: np.ndarray):
        """
        Основной update loop физики
        
        Args:
            dt: время шага симуляции (обычно 0.016 для 60 FPS)
            container_transform: матрица трансформации контейнера (позиция + ротация)
        """
        # Шаг 1: Строим spatial grid для быстрого поиска соседей
        self._build_spatial_grid()
        
        # Шаг 2: Вычисляем плотность каждой частицы
        self._compute_densities()
        
        # Шаг 3: Вычисляем давление из плотности
        self._compute_pressures()
        
        # Шаг 4: Вычисляем все силы (давление, вязкость, гравитация и т.д.)
        self._compute_forces(container_transform)
        
        # Шаг 5: Интегрируем (обновляем позиции и скорости)
        self._integrate(dt)
        
        # Шаг 6: Обрабатываем коллизии со стенками контейнера
        self._handle_collisions(container_transform)
    
    def _build_spatial_grid(self):
        """
        Spatial Hashing - оптимизация для поиска соседей
        
        Проблема: чтобы найти соседей частицы, нужно проверить все N частиц -> O(N²)
        Решение: разбиваем пространство на сетку ячеек, проверяем только соседние ячейки
        
        Это ускоряет поиск с O(N²) до O(N)
        """
        self.spatial_grid.clear()
        cell_size = self.config.cell_size
        
        for i in range(self.particles.num_particles):
            pos = self.particles.positions[i]
            
            # Вычисляем индекс ячейки
            cell_x = int(pos[0] / cell_size)
            cell_y = int(pos[1] / cell_size)
            cell_z = int(pos[2] / cell_size)
            cell_key = (cell_x, cell_y, cell_z)
            
            # Добавляем частицу в ячейку
            if cell_key not in self.spatial_grid:
                self.spatial_grid[cell_key] = []
            self.spatial_grid[cell_key].append(i)
    
    def _get_neighbors(self, particle_idx: int) -> list:
        """
        Находит всех соседей частицы в радиусе smoothing_radius
        Использует spatial grid для оптимизации
        """
        pos = self.particles.positions[particle_idx]
        cell_size = self.config.cell_size
        radius = self.config.smoothing_radius
        
        # Определяем какие ячейки нужно проверить
        cell_x = int(pos[0] / cell_size)
        cell_y = int(pos[1] / cell_size)
        cell_z = int(pos[2] / cell_size)
        
        neighbors = []
        
        # Проверяем 27 соседних ячеек (3x3x3 куб)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    cell_key = (cell_x + dx, cell_y + dy, cell_z + dz)
                    
                    if cell_key in self.spatial_grid:
                        for neighbor_idx in self.spatial_grid[cell_key]:
                            if neighbor_idx != particle_idx:
                                # Проверяем расстояние
                                dist = np.linalg.norm(pos - self.particles.positions[neighbor_idx])
                                if dist < radius:
                                    neighbors.append(neighbor_idx)
        
        return neighbors
    
    def _compute_densities(self):
        """
        Вычисляем плотность каждой частицы
        
        Плотность = сумма масс всех соседей * kernel_function(расстояние)
        Kernel function - это математическая функция которая делает влияние соседей
        плавным (чем дальше сосед, тем меньше его влияние)
        """
        for i in range(self.particles.num_particles):
            density = 0.0
            neighbors = self._get_neighbors(i)
            
            for j in neighbors:
                r = np.linalg.norm(self.particles.positions[i] - self.particles.positions[j])
                
                # Poly6 kernel - стандартная kernel функция для SPH
                if r < self.config.smoothing_radius:
                    kernel_value = self._poly6_kernel(r)
                    density += self.config.particle_mass * kernel_value
            
            # Добавляем собственную массу
            density += self.config.particle_mass * self._poly6_kernel(0.0)
            
            self.particles.densities[i] = max(density, self.config.rest_density)  # Clamp к минимуму
    
    def _poly6_kernel(self, r: float) -> float:
        """
        Poly6 kernel function для SPH
        
        Эта функция определяет как сильно влияют соседи на частицу
        Возвращает большое значение если r близко к 0, и 0 если r > h
        """
        h = self.config.smoothing_radius
        
        if r >= 0 and r <= h:
            factor = 315.0 / (64.0 * np.pi * h**9)
            return factor * (h**2 - r**2)**3
        return 0.0
    
    def _compute_pressures(self):
        """
        Вычисляем давление из плотности
        
        Pressure = k * (density - rest_density)
        k = gas_constant (насколько "жесткая" жидкость)
        
        Чем больше density, тем больше давление -> частицы расталкиваются
        """
        for i in range(self.particles.num_particles):
            self.particles.pressures[i] = self.config.gas_constant * (
                self.particles.densities[i] - self.config.rest_density
            )
    
    def _compute_forces(self, container_transform: np.ndarray):
        """
        Вычисляем все силы действующие на частицы
        
        Силы в SPH:
        1. Pressure force - сила давления (расталкивает частицы)
        2. Viscosity force - сила вязкости (замедляет частицы)
        3. Gravity - гравитация
        4. Surface tension - поверхностное натяжение (cohesion)
        """
        self.particles.forces.fill(0.0)
        
        for i in range(self.particles.num_particles):
            pressure_force = np.zeros(3, dtype=np.float32)
            viscosity_force = np.zeros(3, dtype=np.float32)
            
            neighbors = self._get_neighbors(i)
            
            for j in neighbors:
                r_vec = self.particles.positions[i] - self.particles.positions[j]
                r = np.linalg.norm(r_vec)
                
                if r < 0.0001:  # Избегаем деления на ноль
                    continue
                
                r_normalized = r_vec / r
                
                # === PRESSURE FORCE ===
                # Использует Spiky kernel gradient
                pressure_term = (self.particles.pressures[i] + self.particles.pressures[j]) / (2.0 * self.particles.densities[j])
                spiky_grad = self._spiky_kernel_gradient(r)
                pressure_force -= self.config.particle_mass * pressure_term * spiky_grad * r_normalized
                
                # === VISCOSITY FORCE ===
                # Использует Laplacian of viscosity kernel
                velocity_diff = self.particles.velocities[j] - self.particles.velocities[i]
                viscosity_laplacian = self._viscosity_kernel_laplacian(r)
                viscosity_force += self.config.viscosity * self.config.particle_mass * (velocity_diff / self.particles.densities[j]) * viscosity_laplacian
            
            # Суммируем все силы
            self.particles.forces[i] = pressure_force + viscosity_force
            
            # Добавляем гравитацию (трансформируем по ротации контейнера)
            # TODO: применить трансформацию контейнера к гравитации
            gravity = self.config.gravity * self.particles.densities[i]
            self.particles.forces[i] += gravity
    
    def _spiky_kernel_gradient(self, r: float) -> float:
        """
        Градиент Spiky kernel для вычисления pressure force
        """
        h = self.config.smoothing_radius
        
        if r >= 0 and r <= h:
            factor = -45.0 / (np.pi * h**6)
            return factor * (h - r)**2
        return 0.0
    
    def _viscosity_kernel_laplacian(self, r: float) -> float:
        """
        Лапласиан viscosity kernel для вычисления viscosity force
        """
        h = self.config.smoothing_radius
        
        if r >= 0 and r <= h:
            factor = 45.0 / (np.pi * h**6)
            return factor * (h - r)
        return 0.0
    
    def _integrate(self, dt: float):
        """
        Интегрируем физику (обновляем позиции и скорости)
        
        Используем Semi-Implicit Euler:
        1. velocity = velocity + (force / mass) * dt
        2. position = position + velocity * dt
        """
        for i in range(self.particles.num_particles):
            # F = ma => a = F/m
            acceleration = self.particles.forces[i] / self.particles.densities[i]
            
            # Обновляем скорость
            self.particles.velocities[i] += acceleration * dt
            
            # Damping (затухание) для стабильности
            self.particles.velocities[i] *= (1.0 - self.config.damping)
            
            # Ограничиваем максимальную скорость
            speed = np.linalg.norm(self.particles.velocities[i])
            if speed > self.config.max_speed:
                self.particles.velocities[i] *= self.config.max_speed / speed
            
            # Обновляем позицию
            self.particles.positions[i] += self.particles.velocities[i] * dt
    
    def _handle_collisions(self, container_transform: np.ndarray):
        """
        Обрабатываем коллизии частиц со стенками контейнера
        
        Простая версия: если частица вышла за границы контейнера,
        возвращаем её обратно и инвертируем скорость (с потерей энергии)
        
        TODO: учитывать трансформацию контейнера (ротацию)
        """
        half_width = Config.container.width / 2
        half_height = Config.container.height / 2
        half_depth = Config.container.depth / 2
        
        for i in range(self.particles.num_particles):
            pos = self.particles.positions[i]
            vel = self.particles.velocities[i]
            
            # Проверяем каждую ось
            # X axis
            if pos[0] < -half_width:
                self.particles.positions[i, 0] = -half_width
                self.particles.velocities[i, 0] *= -self.config.restitution
            elif pos[0] > half_width:
                self.particles.positions[i, 0] = half_width
                self.particles.velocities[i, 0] *= -self.config.restitution
            
            # Y axis
            if pos[1] < -half_height:
                self.particles.positions[i, 1] = -half_height
                self.particles.velocities[i, 1] *= -self.config.restitution
            elif pos[1] > half_height:
                self.particles.positions[i, 1] = half_height
                self.particles.velocities[i, 1] *= -self.config.restitution
            
            # Z axis
            if pos[2] < -half_depth:
                self.particles.positions[i, 2] = -half_depth
                self.particles.velocities[i, 2] *= -self.config.restitution
            elif pos[2] > half_depth:
                self.particles.positions[i, 2] = half_depth
                self.particles.velocities[i, 2] *= -self.config.restitution


# Тест модуля
if __name__ == "__main__":
    """
    Простой тест физики без рендеринга
    Запустить: python fluid_physics.py
    """
    print("=== Testing Fluid Physics ===")
    
    # Создаем систему частиц
    particles = ParticleSystem(num_particles=100)
    solver = SPHSolver(particles)
    
    # Симулируем несколько шагов
    dt = 0.016  # ~60 FPS
    container_transform = np.eye(4, dtype=np.float32)  # Identity matrix
    
    print(f"Начальная позиция первой частицы: {particles.positions[0]}")
    
    for step in range(100):
        solver.update(dt, container_transform)
        
        if step % 20 == 0:
            print(f"Step {step}: particle[0] pos={particles.positions[0]}, vel={particles.velocities[0]}")
    
    print(f"Финальная позиция первой частицы: {particles.positions[0]}")
    print("=== Test Complete ===")