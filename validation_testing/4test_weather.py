#!/usr/bin/env python3

# Copyright (c) 2025 Endang Rusiana.
# This work is licensed under the terms of the MIT License.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Single RGB Camera Evaluation with RouteManager - Manual Weather Toggle
Tekan tombol W untuk mengganti cuaca.
Author: Endang Rusiana (2025)
"""

import carla
import random
import time
import pygame
import numpy as np
import os
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.global_route_planner import GlobalRoutePlanner

# ==========================================================
# === SETUP KONEKSI DAN WORLD ===
# ==========================================================
def init_carla(host="localhost", port=2000, town="Town10HD"):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.load_world(town)
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    print(f"[INFO] Connected to {town}")
    return client, world, tm

# ==========================================================
# === ROUTE MANAGER ===
# ==========================================================
class RouteManager:
    def __init__(self, world, agent, sampling_resolution=2.0):
        self.world = world
        self.map = world.get_map()
        self.agent = agent
        self.sampling_resolution = sampling_resolution
        self.route_planner = GlobalRoutePlanner(self.map, self.sampling_resolution)
        self._plan = None
        print("[INFO] GlobalRoutePlanner initialized (CARLA 0.9.15).")

    def set_route(self, start, end):
        self._plan = self.route_planner.trace_route(start.location, end.location)
        self.agent.set_global_plan(self._plan)
        print(f"[INFO] Route with {len(self._plan)} waypoints set.")

# ==========================================================
# === SPAWN HERO VEHICLE ===
# ==========================================================
def spawn_hero(world):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = random.choice(bp_lib.filter("vehicle.nissan.micra"))
    vehicle_bp.set_attribute("role_name", "hero")
    spawn_point = random.choice(world.get_map().get_spawn_points())
    hero = world.try_spawn_actor(vehicle_bp, spawn_point)
    if hero is None:
        raise RuntimeError("❌ Failed to spawn hero vehicle.")
    print(f"[INFO] Hero vehicle spawned at {spawn_point.location}")
    return hero, spawn_point

# ==========================================================
# === SETUP RGB CAMERA ===
# ==========================================================
def spawn_rgb_camera(world, hero, width=1280, height=720, fov="100"):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", fov)
    bp.set_attribute("sensor_tick", "0.05")  # 20 FPS

    transform = carla.Transform(carla.Location(x=0, y=0, z=2.4), carla.Rotation(pitch=2.0))
    camera = world.spawn_actor(bp, transform, attach_to=hero)
    queue = []
    camera.listen(lambda data: queue.append(data))
    print(f"[INFO] RGB camera attached ({width}x{height}, FOV={fov})")
    return camera, queue

# ==========================================================
# === DISPLAY / HUD ===
# ==========================================================
def init_display(width, height):
    pygame.init()
    pygame.font.init()
    display = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CARLA Route Navigation View")
    return display

def draw_image(display, image, width, height):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((height, width, 4))[:, :, :3][:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    pygame.display.flip()

def spawn_traffic(world, tm, num_vehicles=30):
    blueprint_library = world.get_blueprint_library()
    vehicle_bps = blueprint_library.filter("vehicle.*")
    spawn_points = world.get_map().get_spawn_points()

    vehicles = []
    for i in range(num_vehicles):
        bp = random.choice(vehicle_bps)
        bp.set_attribute("role_name", "autopilot")
        spawn_point = random.choice(spawn_points)
        npc = world.try_spawn_actor(bp, spawn_point)
        if npc is not None:
            npc.set_autopilot(True, tm.get_port())  # aktifkan autopilot di Traffic Manager
            vehicles.append(npc)
    print(f"[INFO] Spawned {len(vehicles)} traffic vehicles.")
    return vehicles

# ==========================================================
# === MAIN LOOP ===
# ==========================================================
def main():
    WIDTH, HEIGHT = 1280, 720
    client, world, tm = init_carla()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)

    hero, start_point = spawn_hero(world)
    rgb_cam, rgb_queue = spawn_rgb_camera(world, hero, WIDTH, HEIGHT)
    display = init_display(WIDTH, HEIGHT)

    agent = BehaviorAgent(hero, behavior="normal")
    route_manager = RouteManager(world, agent)

    # ==== Tentukan rute random ====
    all_spawns = world.get_map().get_spawn_points()
    end_point = random.choice(all_spawns)
    route_manager.set_route(start_point, end_point)

    print(f"[INFO] Start: {start_point.location}")
    print(f"[INFO] End:   {end_point.location}")

    vehicles_list = spawn_traffic(world, tm, num_vehicles=40)

    # ==== Dictionary cuaca ====
    weathers_dict = {
        "ClearNoon": carla.WeatherParameters.ClearNoon,
        "ClearSunset": carla.WeatherParameters.ClearSunset,
        "WetNoon": carla.WeatherParameters.WetNoon,
        "HardRainNoon": carla.WeatherParameters.HardRainNoon,
        "WetSunset": carla.WeatherParameters.WetSunset,
        "SoftRainSunset": carla.WeatherParameters.SoftRainSunset
    }
    weather_names = list(weathers_dict.keys())
    weather_idx = 0

    world.set_weather(weathers_dict[weather_names[weather_idx]])
    print(f"[INFO] Initial weather set to: {weather_names[weather_idx]}")

    clock = pygame.time.Clock()
    try:
        stat_nav = 0
        while True:
            dt = clock.tick()
            world.tick()

            if rgb_queue:
                image = rgb_queue.pop(0)
                draw_image(display, image, WIDTH, HEIGHT)


            control = agent.run_step()
            if stat_nav == 1:
                control.brake = 1.0
                control.throttle = 0.0
                control.steer = 0.0

            hero.apply_control(control)

            # === Event keyboard ===
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        raise KeyboardInterrupt
                    elif event.key == pygame.K_w:
                        weather_idx = (weather_idx + 1) % len(weather_names)
                        current_name = weather_names[weather_idx]
                        world.set_weather(weathers_dict[current_name])
                        print(f"[INFO] Weather changed to: {current_name}")
                    elif event.key == pygame.K_g:
                        if stat_nav == 0:
                            stat_nav = 1
                        else:
                            stat_nav = 0

    except KeyboardInterrupt:
        print("[INFO] Simulation terminated by user.")

    finally:
        rgb_cam.stop()
        rgb_cam.destroy()
        hero.destroy()
        for v in vehicles_list:
            v.destroy()
        world.apply_settings(carla.WorldSettings(no_rendering_mode=False, synchronous_mode=False))
        pygame.quit()
        print("[INFO] Cleaned up and exited.")

if __name__ == "__main__":
    main()
