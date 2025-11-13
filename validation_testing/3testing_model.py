#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
# Modifications (c) 2025 Endang Rusiana

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import csv
import os
import numpy.random as random
import re
import sys
import weakref
import time
import cv2  # ✅ Tambahkan ini agar OpenCV bisa digunakan
from itertools import islice
import gc  # tambahkan di atas
import queue
import psutil
import copy  # tambahkan di atas file
from tensorflow.keras.models import load_model
import tensorflow as tf
import random as pyrandom  # jangan bentrok dengan numpy.random

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"[INFO] {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs set with memory growth.")
    except RuntimeError as e:
        print(f"[ERROR] Cannot set memory growth: {e}")

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.custom_agent import CustomAgent

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
import h5py
from agents.tools.misc import get_trafficlight_trigger_location, is_within_distance

from multiprocessing import Process, Queue
# Buffer untuk menyimpan nilai kontrol kendaraan
control_buffer = []

flush_queue = Queue()

def flush_worker(q):
    import time
    import gc
    import h5py
    import numpy as np
    from agents.tools.misc import get_trafficlight_trigger_location, is_within_distance

    while True:
        task = q.get()
        if task is None:
            break
        buffer, batch_id = task
        print(f"[WORKER] Mulai flush batch-{batch_id}, metadata: {len(buffer['metadata'])}, rgb: {len(buffer['rgb'])}")
        try:
            from __main__ import flush_to_h5  # agar bisa panggil versi lokal Anda
            flush_to_h5(buffer, batch_id)
        except Exception as e:
            print(f"[FLUSH WORKER ERROR] Failed to flush batch-{batch_id:03d}: {e}")

def affected_by_traffic_light(vehicle, lights_list, map_inst, max_distance=15.0):
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_waypoint = map_inst.get_waypoint(vehicle_location)

    for light in lights_list:
        if light.state != carla.TrafficLightState.Red:
            continue

        trigger_location = get_trafficlight_trigger_location(light)
        if trigger_location is None:
            continue

        trigger_waypoint = map_inst.get_waypoint(trigger_location)

        if trigger_waypoint.road_id != vehicle_waypoint.road_id:
            continue

        ve_dir = vehicle_waypoint.transform.get_forward_vector()
        wp_dir = trigger_waypoint.transform.get_forward_vector()
        dot = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
        if dot < 0:
            continue  # TL di belakang

        if is_within_distance(trigger_waypoint.transform, vehicle_transform, max_distance, [0, 90]):
            return True, light

    return False, None

# Simulasi penyimpanan frame
frame_count = 0  # Counter untuk jumlah frame yang sudah diproses
frame_count_h5 = 0  # Counter untuk jumlah frame yang sudah diproses
target_frame_count = 5000  # Batch size per flush
# target_frame_count = 100  # Batch size per flush
batch_counter = 1  # Nomor batch pertama

# Fungsi untuk mendapatkan nama file berdasarkan nomor batch
def get_batch_filename(batch_number):
    return args.output  # langsung pakai path dari command-line

class SteerGraphDisplay:
    def __init__(self, display_man, display_pos, history_len=200, fps=20):
        self.display_man = display_man
        self.display_pos = display_pos
        self.surface = pygame.Surface(display_man.get_display_size())

        # Riwayat steer
        self.noise_history = [0.0] * history_len
        self.control_history = [0.0] * history_len
        self.resultant_history = [0.0] * history_len

        # Riwayat throttle, brake & speed
        self.throttle_history = [0.0] * history_len
        self.brake_history = [0.0] * history_len
        self.speed_history = [0.0] * history_len  # km/h, normalisasi saat render

        self.history_len = history_len
        self.fps = fps
        self.font = pygame.font.SysFont("monospace", 14)

        self.display_man.add_sensor(self)

    def update(self, noise_val, control_val, resultant_val, throttle_val, brake_val, speed_val):
        # Update steer
        self.noise_history.pop(0)
        self.noise_history.append(noise_val)

        self.control_history.pop(0)
        self.control_history.append(control_val)

        self.resultant_history.pop(0)
        self.resultant_history.append(resultant_val)

        # Update throttle
        self.throttle_history.pop(0)
        self.throttle_history.append(throttle_val)

        # Update brake
        self.brake_history.pop(0)
        self.brake_history.append(brake_val)

        # Update speed
        self.speed_history.pop(0)
        self.speed_history.append(speed_val)

    def render_graph(self, surface, histories, colors, labels, fixed_max=None):
        surface.fill((0, 0, 0))
        max_val = fixed_max if fixed_max is not None else max(
            1.0, max(abs(x) for h in histories for x in h)
        )
        w, h = surface.get_size()

        # Grid horizontal (interval 0.1)
        grid_step = 0.1
        y_val = -max_val
        while y_val <= max_val:
            y_pixel = int((0.5 - (y_val / (2 * max_val))) * h)
            pygame.draw.line(surface, (50, 50, 50), (0, y_pixel), (w, y_pixel), 1)
            label = f"{y_val:+.2f}"
            text_surf = self.font.render(label, True, (200, 200, 200))
            surface.blit(text_surf, (5, y_pixel - text_surf.get_height() // 2))
            y_val += grid_step

        # Grid vertikal waktu
        total_time = self.history_len / self.fps
        for t in range(0, int(total_time) + 1):
            x_pixel = int((t / total_time) * w)
            pygame.draw.line(surface, (50, 50, 50), (x_pixel, 0), (x_pixel, h), 1)
            label = f"{t}s"
            text_surf = self.font.render(label, True, (200, 200, 200))
            surface.blit(text_surf, (x_pixel + 2, h - text_surf.get_height() - 2))

        # Normalisasi
        def normalize(history):
            return [0.5 + 0.5 * (x / max_val) for x in history]
        norm_histories = [normalize(h) for h in histories]

        # Titik
        def make_points(vals):
            return [(i * (w / len(vals)), (1 - y) * h) for i, y in enumerate(vals)]

        # Gambar garis
        for hist, color, lbl in zip(norm_histories, colors, labels):
            pygame.draw.lines(surface, color, False, make_points(hist), 2)

    def render(self):
        self.surface.fill((0, 0, 0))
        full_w, full_h = self.surface.get_size()
        half_w = full_w // 2

        # Panel kiri → throttle (biru), brake (kuning), speed (/3, merah)
        throttle_hist = self.throttle_history
        brake_hist = self.brake_history
        speed_norm_hist = [s / 30.0 for s in self.speed_history]

        left_surface = pygame.Surface((half_w, full_h))
        self.render_graph(
            left_surface,
            [throttle_hist, brake_hist, speed_norm_hist],
            [(0, 0, 255), (255, 255, 0), (255, 0, 0)],  # biru=throttle, kuning=brake, merah=speed
            ["Throttle", "Brake", "Speed (/3)"],
            fixed_max=1.0
        )

        # Panel kanan → steer (noise, control, resultant)
        right_surface = pygame.Surface((half_w, full_h))
        self.render_graph(
            right_surface,
            [self.noise_history, self.control_history, self.resultant_history],
            [(255, 136, 0), (255, 255, 255), (0, 255, 0)],
            ["Noise", "Control", "Resultant"]
        )

        # Gabungkan
        self.surface.blit(left_surface, (0, 0))
        self.surface.blit(right_surface, (half_w, 0))

        offset = self.display_man.get_display_offset(self.display_pos)
        self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        pass

class WaypointDisplay:
    def __init__(self, agent, world, display_man, display_pos, route_manager=None):
        self.local_waypoints = []  # 🔥 untuk menyimpan waypoint lokal (x, y)
        self.route_manager = route_manager
        self.agent = agent
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.surface = pygame.Surface(display_man.get_display_size())
        display_man.add_sensor(self)

    def render(self):
        self.surface.fill((0, 0, 0))  # background hitam
        player_tf = self.world.player.get_transform()
        desired_waypoint_count = 5

        if hasattr(self.agent, '_local_planner') and hasattr(self.agent._local_planner, '_waypoints_queue'):
            plan = self.route_manager.get_plan()
        else:
            plan = []

        plan_list = list(plan)
        waypoints = [w[0] for w in plan_list[:desired_waypoint_count]]
        # print(f"len wp: {len(waypoints)}")

        # Jika waypoint kurang dari desired, duplikasi waypoint terakhir (jika ada)
        while len(waypoints) < desired_waypoint_count:
            if waypoints:
                waypoints.append(waypoints[-1])
            else:
                break  # Tidak bisa menambahkan apapun jika kosong

        self.local_waypoints = []
        origin_x, origin_y = None, None

        for i, wp in enumerate(waypoints):
            local_x, local_y = world_to_local(player_tf, wp.transform)

            if i == 0:
                origin_x, origin_y = local_x, local_y

            norm_x = round(local_x - origin_x, 2)
            norm_y = round(local_y - origin_y, 2)

            self.local_waypoints.append((norm_x, norm_y))

            draw_x = int(self.surface.get_width() / 2 + norm_y * 10)
            draw_y = int(self.surface.get_height() - norm_x * 10)
            color = (255, 0, 0) if i == 0 else (0, 255, 0)
            pygame.draw.circle(self.surface, color, (draw_x, draw_y), 5)

        offset = self.display_man.get_display_offset(self.display_pos)
        self.display_man.display.blit(self.surface, offset)
        
    def get_last_local_waypoints(self):
        return self.local_waypoints
    
    def destroy(self):
        pass  

class RouteManager:
    def __init__(self, agent):
        self.agent = agent
        self._plan = None
        self._last_frame_updated = -1

    def update_plan(self, frame_id=None):
        # Hindari update berulang di frame yang sama
        if frame_id is not None and frame_id == self._last_frame_updated:
            return
        self._plan = self.agent.get_local_planner().get_plan()
        self._last_frame_updated = frame_id

    def get_plan(self):
        return self._plan

def lidar_to_histogram_bev(points, height=88, width=128, ground_thresh=0.3):
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # Filter hanya area depan (-90° s.d. +90°)
    azimuth = np.degrees(np.arctan2(y_points, x_points))
    azimuth = (azimuth + 360) % 360
    mask_azimuth = (azimuth >= 270) | (azimuth <= 90)

    x_points = x_points[mask_azimuth]
    y_points = y_points[mask_azimuth]
    z_points = z_points[mask_azimuth]

    # ✅ Batasi area agar cocok dengan ukuran 88x128
    # Asumsikan X: 0-25m (depan), Y: -20m s.d. +20m (samping)
    x_min, x_max = 0.0, 25.0
    y_min, y_max = -20.0, 20.0

    mask = (x_points >= x_min) & (x_points <= x_max) & (y_points >= y_min) & (y_points <= y_max)
    x_points = x_points[mask]
    y_points = y_points[mask]
    z_points = z_points[mask]

    # Mapping ke citra
    x_img = np.floor((y_points - y_min) / (y_max - y_min) * (width - 1)).astype(np.int32)
    y_img = height - 1 - np.floor((x_points - x_min) / (x_max - x_min) * (height - 1)).astype(np.int32)

    channel0 = np.zeros((height, width), dtype=np.uint8)
    channel1 = np.zeros((height, width), dtype=np.uint8)

    for x, y, z in zip(x_img, y_img, z_points):
        if 0 <= x < width and 0 <= y < height:
            if z > ground_thresh:
                channel0[y, x] = min(channel0[y, x] + 1, 255)
            else:
                channel1[y, x] = min(channel1[y, x] + 1, 255)

    return np.stack([channel0, channel1], axis=-1)  # Output shape: (88, 128, 2)

def world_to_local(player_transform, target_transform):
    dx = target_transform.location.x - player_transform.location.x
    dy = target_transform.location.y - player_transform.location.y
    yaw = math.radians(player_transform.rotation.yaw)
    local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
    local_y = dy * math.cos(yaw) - dx * math.sin(yaw)
    return local_x, local_y

def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        blueprint = random.choice(blueprint_list)
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Hancurkan kendaraan sebelumnya jika ada
        if self.player is not None:
            self.destroy()  # Hancurkan kendaraan yang lama
        
        # Ambil spawn points yang tersedia
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            print('No spawn points available in your map/town.')
            sys.exit(1)

        # === Pilih salah satu lokasi spesifik ===
        # === List spawn point straight town1 ===
        custom_spawn_points_town1 = [
            carla.Transform(
                carla.Location(x=392.47, y=163.92, z=0.30),
                carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)
            ),
            carla.Transform(
                carla.Location(x=395.96, y=249.43, z=0.30),
                carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)
            ),
            carla.Transform(
                carla.Location(x=1.51, y=249.43, z=0.30),
                carla.Rotation(pitch=0.0, yaw=-90.0, roll=0.0)
            ),
            carla.Transform(
                carla.Location(x=-1.76, y=119.34, z=0.30),
                carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)
            )
        ]

        custom_spawn_points_town2 = [
            carla.Transform(
                carla.Location(x=131.69, y=105.55, z=0.50),
                carla.Rotation(pitch=0.0, yaw=-180.0, roll=0.0)
            ),
            carla.Transform(
                carla.Location(x=55.41, y=109.40, z=0.50),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)   # yaw = -0.0 sama saja dengan 0.0
            )
        ]

        # Pilih salah satu secara random (atau bisa langsung pakai index 0 / 1)
        if args.task == "straight":
            if args.town == "Town02" :
                spawn_point = random.choice(custom_spawn_points_town2)
            elif args.town == "Town01":
                spawn_point = random.choice(custom_spawn_points_town1)
        else:
            spawn_point = random.choice(spawn_points)

        # Pilih titik spawn acak dari daftar spawn_points
        # spawn_point = random.choice(spawn_points)
        # print(f"spawn point:{spawn_point.location}")

        # Spawn kendaraan baru di titik spawn yang dipilih
        self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        if self.player is None:
            print('Failed to spawn player vehicle.')
            sys.exit(1)
        print(f"[INFO] Ego vehicle spawned with role_name: {self.player.attributes.get('role_name')}")

        # Setel fisika kendaraan setelah di-spawn
        self.modify_vehicle_physics(self.player)

        # Jika synchronous mode diaktifkan
        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        time.sleep(0.05)

        # Tampilkan notifikasi bahwa kendaraan baru telah di-spawn
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Cycle through weather presets"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock, agent=None):
        """Tick the HUD"""
        self.hud.tick(self, clock, agent)

    def render(self, display):
        """Render the HUD"""
        self.hud.render(display)

    def destroy(self):
        """Destroy player vehicle only"""
        if self.player is not None and self.player.is_alive:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================
class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.simulated_fps = 0  # ✅ Tambahkan variabel ini
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.debug_status = ""  # Untuk menyimpan teks debug yang kamu inginkan
        self.noise_history = [0] * 200  # History untuk noise steer
        self.steer_signal_history = [0] * 200
        self.steer_resultant_history = [0] * 200


    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds
        self.simulated_fps = 1.0 / timestamp.delta_seconds

    def tick(self, world, clock, agent=None):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        # colhist = world.collision_sensor.get_collision_history()
        if 'collision_sensor' in world.sensors:
            colhist = world.sensors['collision_sensor'].collision_history
        else:
            colhist = [0] * 100  # default kosong untuk sementara

        # collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        history_len = len(colhist)
        collision = [colhist[(x + self.frame - 200) % history_len] for x in range(0, 200)]

        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        # print('Server:  % 16.0f FPS' % self.server_fps)
        self._info_text = [
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            'Simulated: % 14.0f FPS' % self.simulated_fps,
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y))
        ]

        # Tambahkan blok if di luar list
        if 'gnss_sensor' in world.sensors:
            gnss_data = world.sensors['gnss_sensor']
            gnss_text = 'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (gnss_data.latitude, gnss_data.longitude))
        else:
            gnss_text = 'GNSS: Sensor not available'

        # Append tambahan ke info text
        self._info_text += [
            gnss_text,
            'Height:  % 18.0f m' % transform.location.z
        ]

        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
                    
        if self.debug_status:
            wrapped = self.wrap_text(self.debug_status, 400)  # max 370px agar sesuai HUD baru
            self._info_text.extend(wrapped)

        self._info_text += [
            'Steer control (white), Noise (orange)',
            ', Resultant (green), Speed (red), Throttle (blue)',
            ', Brake (yellow):'
        ]

        self._info_text += [
            ''
        ]

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def wrap_text(self, text, max_width):
        words = text.split(' ')
        lines = []
        current_line = ''
        for word in words:
            test_line = current_line + word + ' '
            if self._font_mono.size(test_line)[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word + ' '
        lines.append(current_line)
        return lines

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((400, 450))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        # --- Gambar bar horizontal ---
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)

                        # --- Tambahkan nilai angka ---
                        val_str = f"{item[1]:.4f}"
                        surface_val = self._font_mono.render(val_str, True, (255, 255, 255))
                        display.blit(surface_val, (bar_h_offset + bar_width + 8, v_offset))
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, world, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        # world = self._parent.get_world()
        self.world = world  # ambil world global, bukan dari actor yang mungkin invalid
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification(f'Collision with {actor_type}')
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))

        # Trigger teleport regardless of intensity
        self.world.collision_detected = True  # Trigger langsung pada setiap collision

        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, world, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        # world = self._parent.get_world()
        self.world = world
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, world, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        # world = self._parent.get_world()
        self.world = world
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================
class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

class DisplayManager:
    def __init__(self, grid_size, window_size):
        # pygame.init()
        # pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()
#EDG-----------------------------------
        # pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, args):
        self.args = args
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
#EDG---------------------------------------
        # Default collision attributes (ensures they always exist)
        self.collision_text = "Collision: No Data"
        self.collision_history = [0] * 100  # Default zero history
#END---------------------------------------
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'GNSS':
            gnss_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
            for key in sensor_options:
                gnss_bp.set_attribute(key, sensor_options[key])  # Apply user-defined options
            gnss = self.world.spawn_actor(gnss_bp, transform, attach_to=attached)
            gnss.listen(self.save_gnss_data)
            return gnss
        elif sensor_type == 'Collision':
            collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
            collision = self.world.spawn_actor(collision_bp, transform, attach_to=attached)
            collision.listen(self.save_collision_data)
            return collision
        elif sensor_type == "Radar":
            radar_bp = self.world.get_blueprint_library().find('sensor.other.radar')
            for key in sensor_options:
                radar_bp.set_attribute(key, sensor_options[key])

            radar = self.world.spawn_actor(radar_bp, transform, attach_to=attached)
            radar.listen(self.save_radar_image)
            return radar
        
        else:
            return None

    def get_sensor(self):
        return self.sensor

    def save_gnss_data(self, gnss_data):
        self.latitude = gnss_data.latitude
        self.longitude = gnss_data.longitude
        self.altitude = gnss_data.altitude

    def save_collision_data(self, event):
        actor_type = event.other_actor.type_id
        impulse = event.normal_impulse
        intensity = (impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5

        self.collision_text = f'Collision: {actor_type} | Intensity: {intensity:.2f}'

        # ✅ Update flag teleport
        self.world.collision_detected = True

        if not hasattr(self, 'collision_history'):
            self.collision_history = [0] * 100  # Keep history for smooth scrolling

        self.collision_history.append(intensity)
        self.collision_history.pop(0)

    def save_radar_image(self, radar_data):
        t_start = self.timer.time()
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_data), 4))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        if self.sensor is not None and self.sensor.is_alive:
            self.sensor.destroy()
        self.sensor = None  # pastikan set ke None agar tidak error saat destroy ulang

def inject_steering_noise(probability, noise_std):
    """
    Inject noise into steering value with given probability.
    steer: original steer value (-1 to 1)
    probability: chance to inject noise per frame (e.g., 0.1 = 10%)
    noise_std: standard deviation of gaussian noise
    """
    if pyrandom.random() < probability:
        noise = pyrandom.gauss(0, noise_std)
        steer_noisy = np.clip(noise, -1.0, 1.0)
        return steer_noisy
    else:
        return 0

def is_in_front(vehicle_transform, target_location, fov_threshold=0.5):
    vehicle_location = vehicle_transform.location
    vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
    
    forward_vector = carla.Vector3D(math.cos(vehicle_yaw), math.sin(vehicle_yaw), 0)
    direction_vector = target_location - vehicle_location
    direction_vector.z = 0
    
    direction_vector_norm = math.sqrt(direction_vector.x ** 2 + direction_vector.y ** 2)
    if direction_vector_norm != 0:
        direction_vector.x /= direction_vector_norm
        direction_vector.y /= direction_vector_norm
    
    dot = forward_vector.x * direction_vector.x + forward_vector.y * direction_vector.y
    
    # hanya anggap "di depan" jika dalam ±60 derajat
    return dot > fov_threshold

def is_vehicle_stuck(world, stuck_seconds=10.0, speed_thresh=0.5):
    current_time = time.time()
    if not hasattr(world, "last_moving_time"):
        world.last_moving_time = current_time

    speed_vec = world.player.get_velocity()
    speed = 3.6 * math.sqrt(speed_vec.x ** 2 + speed_vec.y ** 2 + speed_vec.z ** 2)

    if speed > speed_thresh:
        # Jika mobil bergerak normal, reset timer
        world.last_moving_time = current_time

    elapsed_stuck_time = current_time - world.last_moving_time

    if speed <= speed_thresh and elapsed_stuck_time > stuck_seconds:
        print(f"[DEBUG] Stuck detected for {elapsed_stuck_time:.1f} seconds")
        return True
    
    return False

# Add this global variable to track the current weather index
weather_index = 0
def change_weather(world):
    global weather_index
    weather_presets = [
        (carla.WeatherParameters.ClearNoon, "ClearNoon"),
        (carla.WeatherParameters.ClearSunset, "ClearSunset"),
        (carla.WeatherParameters.WetNoon, "WetNoon"),
        (carla.WeatherParameters.HardRainNoon, "HardRainNoon")
    ]

    # Get the current weather preset
    selected_weather, weather_name = weather_presets[weather_index]

    # Apply weather
    world.world.set_weather(selected_weather)

    # Ambil info FPS dari HUD
    server_fps = world.hud.server_fps
    client_fps = pygame.time.Clock().get_fps()  # alternatif jika ingin realtime
    sim_fps = world.hud.simulated_fps

    # Tampilkan info
    print(f"[INFO] Weather changed to: {weather_name} | Server FPS: {server_fps:.1f} | Client FPS: {client_fps:.1f} | Simulated FPS: {sim_fps:.1f}")

    # Increment index
    weather_index = (weather_index + 1) % len(weather_presets)
    return weather_name

last_weather_change_time = 0  # Initialize the time when the weather was last changed

def spawn_traffic(world, client, tm_port=8000, num_vehicles=30, safe=True):
    traffic_manager = client.get_trafficmanager(tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.set_hybrid_physics_mode(True)
    traffic_manager.set_hybrid_physics_radius(30.0)  # Batasi radius fisika
    traffic_manager.set_synchronous_mode(True)

    blueprints = world.get_blueprint_library().filter('vehicle.*')

    if safe:
        # ✅ hanya pilih kendaraan dengan base_type = 'car'
        blueprints = [x for x in blueprints if x.get_attribute('base_type') == 'car']

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor
    batch = []

    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(blueprints)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        if bp.has_attribute('driver_id'):
            driver_id = random.choice(bp.get_attribute('driver_id').recommended_values)
            bp.set_attribute('driver_id', driver_id)
        bp.set_attribute('role_name', 'autopilot')

        batch.append(
            SpawnActor(bp, spawn_points[i]).then(
                SetAutopilot(FutureActor, True, tm_port)
            )
        )

    client.apply_batch_sync(batch, True)
    print(f"[INFO] ✅ Spawned {len(batch)} safe vehicles.")

def spawn_sync_sensor(world, blueprint, transform, attach_to):
    sensor = world.spawn_actor(blueprint, transform, attach_to=attach_to)
    q = queue.Queue()
    sensor.listen(q.put)
    return sensor, q

def process_lidar(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))

    bev = lidar_to_histogram_bev(points, height=88, width=198)
    lidar_above = bev[:, :, 0]
    lidar_ground = bev[:, :, 1]
    return lidar_above, lidar_ground

def carla_rgb_to_array(img):
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3]  # BGR
    arr = arr[:, :, ::-1]          # BGR -> RGB
    arr = arr[62:, :, :]           # crop top 62 px -> (88, 200, 3)
    return arr

def carla_depth_to_gray(img):
    a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    R = a[:, :, 2].astype(np.float32)
    G = a[:, :, 1].astype(np.float32)
    B = a[:, :, 0].astype(np.float32)
    depth = (R + 256*G + 65536*B) / (256**3 - 1) * 1000.0
    val_clip = 100.0
    depth = np.clip(depth, 0, val_clip)
    gray  = (depth / val_clip * 255).astype(np.uint8)
    gray  = gray[62:, :]           # crop top 62 px -> (88, 200)
    gray  = cv2.medianBlur(gray, 5)
    return gray

class SequentialSensorDisplay:
    def __init__(self, display_manager):
        self.display_man = display_manager

        self.rgb_views = {'left': None, 'center': None, 'right': None}
        self.depth_views = {'left': None, 'center': None, 'right': None}
        self.lidar_bev = None  # tetap boleh, 1 panel saja

        self.grid_positions = {
            'rgb_left':     [0, 0],
            'rgb_center':   [0, 1],
            'rgb_right':    [0, 2],
            'depth_left':   [1, 0],
            'depth_center': [1, 1],
            'depth_right':  [1, 2],
            'lidar':        [2, 1],  # bebas, misalnya di tengah bawah
        }
        display_manager.add_sensor(self)

    def update(self, rgb_views, depth_views, lidar_above=None, lidar_ground=None):
        self.rgb_views = rgb_views
        self.depth_views = depth_views
        if lidar_above is not None and lidar_ground is not None:
            self.lidar_bev = lidar_above + lidar_ground

    def render(self):
        if not self.display_man.render_enabled():
            return

        size = self.display_man.get_display_size()

        for key in ['left', 'center', 'right']:
            # RGB
            rgb = self.rgb_views.get(key)
            if rgb is not None:
                rgb_resized = cv2.resize(rgb, size)
                # rgb_bgr = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)  # ✅ konversi ke BGR
                surface = pygame.surfarray.make_surface(rgb_resized.swapaxes(0, 1))
                offset = self.display_man.get_display_offset(self.grid_positions[f'rgb_{key}'])
                self.display_man.display.blit(surface, offset)

            # Depth
            depth = self.depth_views.get(key)
            if depth is not None:
                depth_resized = cv2.resize(depth, size)
                depth_rgb = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2RGB)
                surface = pygame.surfarray.make_surface(depth_rgb.swapaxes(0, 1))
                offset = self.display_man.get_display_offset(self.grid_positions[f'depth_{key}'])
                self.display_man.display.blit(surface, offset)

        # Optional LiDAR BEV display
        if self.lidar_bev is not None:
            bev = cv2.normalize(self.lidar_bev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            bev_rgb = cv2.cvtColor(bev, cv2.COLOR_GRAY2RGB)
            bev_resized = cv2.resize(bev_rgb, size)
            surface = pygame.surfarray.make_surface(bev_resized.swapaxes(0, 1))
            offset = self.display_man.get_display_offset(self.grid_positions['lidar'])
            self.display_man.display.blit(surface, offset)

    def destroy(self):
        pass  # Optional: tambahkan log untuk debug

class ControlSelectorLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        branches, cmd_onehot = inputs
        branches = tf.stack(branches, axis=1)  # (batch, 4, 3)
        cmd_onehot = tf.expand_dims(cmd_onehot, -1)  # (batch, 4, 1)
        return tf.reduce_sum(branches * cmd_onehot, axis=1)

class ControlPostprocess(tf.keras.layers.Layer):
    def call(self, x):
        import tensorflow as tf
        steer    = tf.tanh(x[:, 0:1])
        throttle = tf.sigmoid(x[:, 1:2])
        brake    = tf.sigmoid(x[:, 2:3])
        return tf.concat([steer, throttle, brake], axis=-1)

    def get_config(self):
        return super().get_config()

class ScaleOffset(tf.keras.layers.Layer):
    def __init__(self, alpha=1.0, beta=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = float(alpha); self.beta = float(beta)
    def call(self, x):
        return x * self.alpha + self.beta
    def get_config(self):
        cfg = super().get_config(); cfg.update({"alpha": self.alpha, "beta": self.beta}); return cfg

class ControlPostprocess(tf.keras.layers.Layer):
    def call(self, x):
        steer    = tf.tanh(x[:, 0:1])
        throttle = tf.sigmoid(x[:, 1:2])
        brake    = tf.sigmoid(x[:, 2:3])
        return tf.concat([steer, throttle, brake], axis=-1)
    def get_config(self):
        return super().get_config()

class RGBOnly(tf.keras.layers.Layer):
    """Ambil 3 channel pertama (RGB) dari input RGBD (H,W,4) -> (H,W,3)."""
    def call(self, x):
        return x[..., :3]
    def get_config(self):
        return super().get_config()

class RGBToGray(tf.keras.layers.Layer):
    def __init__(self, dtype='float32', **kwargs):
        # Compute dtype = float32 → op conv berjalan di GPU fp32 (stabil)
        super().__init__(dtype=dtype, **kwargs)

    def build(self, input_shape):
        # Kernel luminance (BT.601) disimpan sebagai float32
        k = tf.constant([[[[0.2989]], [[0.5870]], [[0.1140]]]], dtype=tf.float32)  # (1,1,3,1)
        self.kernel = self.add_weight(
            name="gray_kernel",
            shape=(1, 1, 3, 1),
            initializer=tf.keras.initializers.Constant(k.numpy()),
            trainable=False,
            dtype=tf.float32,
        )

    def call(self, x):
        # Compute conv di GPU float32, lalu cast balik ke dtype input (fp16)
        x32 = tf.cast(x, tf.float32)
        y32 = tf.nn.conv2d(x32, self.kernel, strides=1, padding='SAME')  # ← GPU op
        return tf.cast(y32, x.dtype)

    def get_config(self):
        return super().get_config()

class DepthOnly(tf.keras.layers.Layer):
    """Ambil channel ke-4 sebagai depth (H,W,4) -> (H,W,1)."""
    def call(self, x):
        return x[..., 3:4]
    def get_config(self):
        return super().get_config()
    
def act_loss(y_true, y_pred):
    w = tf.constant([0.5, 0.45, 0.05], dtype=y_pred.dtype)  # ✅ Sesuaikan dtype
    diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.reduce_sum(w * diff, axis=-1))

class ScalingLayer(tf.keras.layers.Layer):
    """Layer skala terlatih: y = alpha * x"""
    def __init__(self, init_alpha=1.0, name="alpha", **kwargs):
        super().__init__(name=name, **kwargs)
        self.init_alpha = float(init_alpha)

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name=f"{self.name}_scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.init_alpha),
            trainable=True,
            dtype=self.dtype
        )

    def call(self, inputs):
        return self.alpha * inputs

    # ⬇️ Tambahan penting untuk serialisasi
    def get_config(self):
        config = super().get_config()
        config.update({
            "init_alpha": self.init_alpha,
            "name": self.name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# buffer untuk menyimpan riwayat kecepatan (panjang 4: t, t-1, t-2, t-3)
speed_history = []

def update_speed(new_speed):
    """
    Simpan speed baru ke dalam buffer dan jaga panjang buffer = 4.
    Urutan: [t, t-1, t-2, t-3]
    """
    speed_history.insert(0, new_speed)      # masukkan speed terbaru di depan
    if len(speed_history) > 4:              # jaga agar panjang maksimal 4
        speed_history.pop()                 # hapus elemen terakhir

    # pastikan nilai t-1,t-2,t-3 ada (atau None jika belum cukup data)
    t   = speed_history[0]
    t_1 = speed_history[1] if len(speed_history) > 1 else 0
    t_2 = speed_history[2] if len(speed_history) > 2 else 0
    t_3 = speed_history[3] if len(speed_history) > 3 else 0

    return t, t_1, t_2, t_3

# ======= util yang sudah ada =======
def _yaw_deg(tf):
    return float(tf.rotation.yaw)

def _unwraped_deltas_deg(headings_deg):
    deltas = []
    for i in range(1, len(headings_deg)):
        a0 = math.radians(headings_deg[i-1])
        a1 = math.radians(headings_deg[i])
        # hasil atan2(sinΔ, cosΔ) memberi delta sudut ter-unwrapped di [-180, +180]
        d = math.degrees(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0)))
        deltas.append(d)
    return deltas

# ======= versi arah: -1 kiri, +1 kanan, 0 lurus =======
def detect_curve_dir_ahead(route_manager, num_points=5,
                           max_turn_thresh_deg=8.0,
                           cum_turn_thresh_deg=25.0):
    """
    Deteksi arah belok pada N waypoint ke depan.
    Return:
      -1  : menikung kiri
      +1  : menikung kanan
       0  : lurus / tidak mencapai ambang kurva
    """
    plan = route_manager.get_plan() or []

    # Ambil maks num_points item dari plan (aman untuk iterator/deque/list)
    try:
        limited = list(islice(plan, num_points))
        if not limited and isinstance(plan, (list, tuple)):
            limited = plan[:num_points]
    except TypeError:
        limited = list(plan)[:num_points]

    # Ekstrak waypoint (elemen ke-0 dari (waypoint, RoadOption))
    wps = []
    for item in limited:
        if isinstance(item, (list, tuple)) and item and item[0] is not None:
            wps.append(item[0])

    if len(wps) < 3:
        return 0  # tidak cukup titik untuk menyimpulkan kurva

    headings = [_yaw_deg(wp.transform) for wp in wps]
    dhead = _unwraped_deltas_deg(headings)       # delta heading bertanda
    if not dhead:
        return 0

    max_abs = max(abs(d) for d in dhead)         # belokan lokal terbesar (|deg|)
    cum_abs = sum(abs(d) for d in dhead)         # akumulasi |belokan|
    signed_sum = sum(dhead)                      # akumulasi bertanda untuk arah

    # Cek apakah dianggap "kurva" dulu (pakai ambang yang sama seperti versi Anda)
    is_curve = (max_abs >= max_turn_thresh_deg) or (cum_abs >= cum_turn_thresh_deg)
    if not is_curve:
        return 0

    # Tentukan arah dari tanda akumulasi delta heading:
    #   signed_sum > 0  → arah A
    #   signed_sum < 0  → arah B
    # Catatan: jika hasil tampak terbalik (kanan/kiri tertukar),
    # tukar return mapping di bawah (ganti -1 <-> +1).
    if signed_sum > 1e-6:
        return +1  # kiri
    elif signed_sum < -1e-6:
        return -1  # kanan
    else:
        return 0   # sangat kecil/ambiguous
    
def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """
    global frame_count  # ✅ agar sinkron dengan fungsi lain

    print("[INFO] Loading trained model...")
    # === Load model dengan custom_objects ===
    MODEL_PATH = f"/home/edgpc/carla/validation_model/model_{args.model_arch}/{args.model_arch}_run{args.run}/saved_model_at_{args.model_saved}.h5"

    print(f"[INFO] Load model {MODEL_PATH}")
    
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            'act_loss': act_loss,
            'ControlSelectorLayer': ControlSelectorLayer,
            'ControlPostprocess': ControlPostprocess,
        }
    )
 
    pygame.init()
    pygame.font.init()
    world = None
    args.sync = True
    args.filter = 'vehicle.nissan.micra'
    args.loop = True
    args.save_data = False
    display_manager = None  # <- tambahkan ini sebelum try
    process = psutil.Process(os.getpid())

    episode_success = 1
    sum_speed_kmh = 0
    speed_tick_count = 0
    DT_S = 0.1  # durasi 1 tick (s)
    prev_speed_kmh = None

    # Batas untuk menyaring lonjakan tak realistis (opsional)
    ACC_SPIKE_LIMIT = 15.0  # m/s^2

    # ====== STAT PERCEPATAN/DESLERASI ======
    max_accel_mps2 = float("-inf")  # percepatan + terbesar
    min_accel_mps2 = None           # percepatan + terkecil (>0)

    max_decel_mps2 = float("inf")   # deselerasi - terbesar (paling negatif)
    min_decel_mps2 = None           # deselerasi - terkecil (<0, paling mendekati nol)

    # ====== HISTOGRAM SPEED DOMINAN ======
    from collections import defaultdict
    speed_hist = defaultdict(int)
    BIN_KMH = 1.0

    def bin_key(v_kmh, bin_kmh=BIN_KMH):
        return round(v_kmh / bin_kmh) * bin_kmh
    
    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(6)

        traffic_manager = client.get_trafficmanager()
        if (args.town == "Town02"):            
            sim_world = client.load_world('Town02')
        elif (args.town == "Town01"):   
            sim_world = client.load_world('Town01')
        # sim_world = client.load_world('Town02')

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            settings.no_rendering_mode = True #no rendering mode
            sim_world.apply_settings(settings)

            traffic_manager.set_synchronous_mode(True)

        if not args.headless:
            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            display = None  # Headless mode: no GUI window
      
        if not args.headless:
            hud = HUD(args.width, args.height)
            display_manager = DisplayManager(grid_size=[3, 3], window_size=[args.width, args.height])
            seq_display = SequentialSensorDisplay(display_manager)
        else:
            hud = type("FakeHUD", (object,), {
                "notification": lambda self, *args, **kwargs: None,
                "on_world_tick": lambda self, *args, **kwargs: None,
                "tick": lambda self, *args, **kwargs: None,
                "render": lambda self, *args, **kwargs: None,
                "server_fps": 30.0,
                "simulated_fps": 30.0,
                "simulation_time": 0.0,
                "debug_status": ""
            })()
            display_manager = type("FakeDisplay", (object,), {
                "render": lambda self: None,
                "add_sensor": lambda self, sensor: None,
                "get_display_offset": lambda self, pos: (0, 0),
                "get_display_size": lambda self: (1, 1),
                "display": None,
                "render_enabled": lambda self: False,
                "destroy": lambda self: None
            })()
            # ✅ Tambahkan ini
            class DummyDisplay:
                def update(self, *args, **kwargs): pass
                def render(self): pass
                def destroy(self): pass

            seq_display = DummyDisplay()

        world = World(client.get_world(), hud, args)

        # === Set weather sesuai argumen ===
        weather_dict = {
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "ClearSunset": carla.WeatherParameters.ClearSunset,
            "WetNoon": carla.WeatherParameters.WetNoon,
            "WetSunset": carla.WeatherParameters.WetSunset,
            "HardRainNoon": carla.WeatherParameters.HardRainNoon,
            "SoftRainSunset": carla.WeatherParameters.SoftRainSunset
        }
        if args.weather in weather_dict:
            world.world.set_weather(weather_dict[args.weather])
            print(f"[INFO] Weather set to {args.weather}")
        else:
            print(f"[WARNING] Weather '{args.weather}' tidak dikenal, menggunakan default ClearNoon")
            world.world.set_weather(carla.WeatherParameters.ClearNoon)

        weather_name = weather_dict[args.weather]
        last_weather_change_time = world.hud.simulation_time  # Set the initial time
        controller = KeyboardControl(world)

        # Ubah SEMUA lampu menjadi hijau dan bekukan
        # ✅ akses objek carla.World asli
        if args.task != "nav_dynamic":
            for tl in world.world.get_actors().filter("traffic.traffic_light"):
                tl.set_state(carla.TrafficLightState.Green)
                tl.set_green_time(9999.0)
                tl.freeze(True)
        else:
            spawn_traffic(world.world, client, num_vehicles=30, safe=True)

        agent = CustomAgent(world.player, behavior='cautious')

        bp_lib = world.world.get_blueprint_library()

        # ======= PARAMETER UMUM KAMERA =======
        IMG_W = int(args.widthcam)
        IMG_H = int(args.heightcam)
        FOV   = '100'   # tetap seperti kode Anda (ubah jika perlu)

        # Ketinggian & posisi relatif kamera terhadap kendaraan
        CAM_LOC  = carla.Location(x=0.0, y=0.0, z=2.4)
        PITCH    = 2.0

        # Sudut yaw untuk kamera lateral (Codevilla: ±30°)
        YAW_LEFT   = -30.0
        YAW_CENTER = 0.0
        YAW_RIGHT  = +30.0

        # (Opsional) geser sedikit ke samping agar baseline kiri/kanan lebih jelas
        # Nilai kecil (±0.25 m) biasanya cukup dan aman dari clipping body kendaraan
        OFFSET_Y = 0.25

        # Helper untuk set atribut kamera
        def setup_camera_bp(bp, w, h, fov):
            bp.set_attribute('image_size_x', str(w))
            bp.set_attribute('image_size_y', str(h))
            bp.set_attribute('fov', fov)
            bp.set_attribute('sensor_tick', '0.05')  # 20 FPS; bisa 0.1 untuk 10 FPS
            return bp

        # =========================================================
        # ===============  RGB CAMERAS: L/C/R  ====================
        # =========================================================
        rgb_bp = setup_camera_bp(bp_lib.find('sensor.camera.rgb'), IMG_W, IMG_H, FOV)

        # Center RGB
        rgb_center_sensor, rgb_center_queue = spawn_sync_sensor(
            world.world, rgb_bp,
            carla.Transform(
                location=carla.Location(x=CAM_LOC.x, y=0.0,           z=CAM_LOC.z),
                rotation=carla.Rotation(pitch=PITCH, yaw=YAW_CENTER, roll=0.0)
            ),
            world.player
        )

        # =========================================================
        # ==============  DEPTH CAMERAS: L/C/R  ===================
        # =========================================================
        depth_bp = setup_camera_bp(bp_lib.find('sensor.camera.depth'), IMG_W, IMG_H, FOV)
        # rekomendasi: fixed_delta_seconds = 0.05 (20 FPS sim) atau 0.1 (10 FPS sim)
        # kalau fixed_delta_seconds=0.05, set sensor_tick depth = 0.10 (kelipatan 2 frame)
        depth_bp.set_attribute('sensor_tick', '0.05')  # 0.10 detik per frame depth (10 FPS)

        # Center Depth
        depth_center_sensor, depth_center_queue = spawn_sync_sensor(
            world.world, depth_bp,
            carla.Transform(
                location=carla.Location(x=CAM_LOC.x, y=0.0,           z=CAM_LOC.z),
                rotation=carla.Rotation(pitch=PITCH, yaw=YAW_CENTER, roll=0.0)
            ),
            world.player
        )

        # ======= LiDAR Sensor =======
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_bp.set_attribute('rotation_frequency', '10.0')
        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('upper_fov', '10.0')
        lidar_bp.set_attribute('lower_fov', '-25.0')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('sensor_tick', '0.05')
        lidar_sensor, lidar_queue = spawn_sync_sensor(world.world, lidar_bp,
            carla.Transform(carla.Location(x=0.0, z=2.5)),
            world.player)

        route_manager = RouteManager(agent)
        WaypointDisp = WaypointDisplay(agent, world, display_manager, display_pos=[2, 2], route_manager=route_manager)
        steer_graph_display = SteerGraphDisplay(display_manager, [2, 0])

        GNSS_Sens = SensorManager(world.world, display_manager, 'GNSS', carla.Transform(carla.Location(x=1.0, z=2.8)), world.player, {}, display_pos=[2, 0], args=args)
        Collision_Sens = SensorManager(world.world, display_manager, 'Collision', carla.Transform(), world.player, {}, display_pos=[2, 0], args=args)

        sensors = {
            'collision_sensor': Collision_Sens,
            'gnss_sensor': GNSS_Sens,
        }

        world.sensors = sensors  # ✅ tambahkan ini agar HUD bisa akses world.sensors

        # Set the agent destination
        # Ambil semua spawn points
        spawn_points = world.world.get_map().get_spawn_points()
        # Lokasi awal hero
        hero_location = world.player.get_location()
        # Filter spawn points yang jaraknya >= 20m dari posisi hero
        valid_points = [
            sp for sp in spawn_points
            if sp.location.distance(hero_location) >= 50.0
        ]
        # Jika ada kandidat valid, pilih salah satu secara acak
        if valid_points:
            destination = random.choice(valid_points).location
        else:
            # fallback: kalau semua kurang dari 20m, tetap pakai random
            destination = random.choice(spawn_points).location
        # Set tujuan ke agent
        agent.set_destination(destination)        

        clock = pygame.time.Clock()
        # Variabel untuk melacak menit terakhir yang ditampilkan
        last_displayed_minute = -1
        world.agent = agent  

        stat_red = False
        stat_at_traffic_light = False
        collision_counter = 0
        prev_collision_counter = 0
        respawn_counter = 0
        noise_stat = False
        noise_counter = 0
        noise_dir = True
        noise_steer = 0

        frame_skipped = 0
        frame_saved = 0
        prev_frame_counter = 0
        delta_frame_counter = 0
        speed_val = 0

        command_counter = 0
        current_command = 0
        prev_command = 0

        if args.model_arch in ('mul_cil', 'mul_cil_f', 'mul_cil_mb', 'mul_cil_minirs', 'mul_cil_a2', 'mul_cil_a2e'):
            model_type = 1
        else: 
            model_type = 0 # Mencakup 'cil' dan model lainnya yang tidak dikenal
        print(f"[INFO] Arch: {args.model_arch} type: {model_type}")

        if args.model_arch in ('mul_cil_e', 'mul_cil_a2e'):
            extend_type = 1
        else:
            extend_type = 0
        print(f"[INFO] Extend type: {extend_type}")        

        time.sleep(2)
        for _ in range(3):
            clock.tick()
            if args.sync:
                frame_id = world.world.tick()
            else:
                frame_id = world.world.wait_for_tick()

        loop_lock = True
        while loop_lock:
            clock.tick()
            if args.sync:
                frame_id = world.world.tick()
            else:
                frame_id = world.world.wait_for_tick()
            if controller.parse_events():
                return
            
            route_manager.update_plan(frame_id)  # update plan hanya sekali per frame

            simulation_seconds = world.hud.simulation_time
          
            world.tick(clock, agent)            

            rgb_center_image = rgb_center_queue.get(timeout=2.0)
            depth_center_image = depth_center_queue.get(timeout=2.0)
            lidar_data = lidar_queue.get(timeout=2.0)

            rgb_center = carla_rgb_to_array(rgb_center_image)

            depth_center = carla_depth_to_gray(depth_center_image)

            lidar_above, lidar_ground = process_lidar(lidar_data)

            seq_display.update(
                {'center': rgb_center},
                {'center': depth_center},
                lidar_above, lidar_ground
            )

            display_manager.render()
            world.render(display)
            if not args.headless:
                pygame.display.flip()

            colhist = world.sensors['collision_sensor'].collision_history
            if any([c > 0 for c in colhist[-5:]]):
                world.sensors['collision_sensor'].collision_history = [0] * 100
                world.hud.notification("Collision detected! Respawn player...", seconds=3.0)
                collision_counter += 1

            if world.player is None:
                print("[INFO] Player vehicle is destroyed or does not exist.")
                break  # Keluar dari loop jika kendaraan dihancurkan


            velocity = world.player.get_velocity()
            speed_mps = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed_mps * 3.6

            speed_kmh_t, speed_kmh_t_1, speed_kmh_t_2, speed_kmh_t_3 = update_speed(speed_kmh)

            steer_out = 0.0
            throttle_out = 0.0
            brake_out = 0.0
            speed_pred = 0.0

            if collision_counter < 4:
                if not(collision_counter > prev_collision_counter):
                    collision_counter = 0
                prev_collision_counter = collision_counter
                control_agent = agent.run_step()
                control_agent.manual_gear_shift = False

                # Ambil semua traffic light dari dunia
                lights_list = world.world.get_actors().filter("*traffic_light*")
                # Panggil fungsi untuk cek apakah kendaraan sedang terkena lampu merah
                stat_red, tl_actor = affected_by_traffic_light(world.player, lights_list, world.map)
                stat_at_traffic_light = (tl_actor is not None)

                stat_red = bool(stat_red)  # aman jika None
                stat_at_traffic_light = tl_actor is not None and isinstance(tl_actor, carla.TrafficLight)

                if agent is not None and hasattr(agent, 'get_local_planner'):
                    plan = route_manager.get_plan()

                    plan_list = list(plan)
                    if len(plan_list) < 10:
                        world.hud.notification("Target reached,searching for another target.", seconds=4.0)
                        print("Target reached,searching for another target. Quit")
                        loop_lock = False                                     
                    next_road_option = plan[0][1]

                control_cil = carla.VehicleControl()
                if frame_count > 0: 
                    waypoints_local = WaypointDisp.get_last_local_waypoints()
                    flat_waypoints = [coord for wp in waypoints_local for coord in wp]

                    # # === Buat input RGBD ===
                    if model_type == 1:
                        rgb_input   = rgb_center.astype(np.float32) / 255.0
                        depth_input = depth_center.astype(np.float32) / 255.0
                        depth_input = np.expand_dims(depth_input, axis=-1)  # (H, W, 1)

                        # ubah [0,1] -> [-1,1]
                        rgb_input   = rgb_input * 2.0 - 1.0
                        depth_input = depth_input * 2.0 - 1.0

                        rgbd_input  = np.concatenate([rgb_input, depth_input], axis=-1)  # (H, W, 4)
                        rgbd_input  = np.expand_dims(rgbd_input, axis=0)                 # (1, 88, 200, 4)
                    else:
                        rgb_input = rgb_center.astype(np.float32) / 255.0
                        rgb_input = rgb_input * 2.0 - 1.0                    # ke [-1,1]
                        rgb_input = np.expand_dims(rgb_input, axis=0)        # (1,88,200,3)


                    val_normalization_speed = 30.0

                    normalized_speed = (speed_kmh/val_normalization_speed) * 2.0 - 1.0
                    speed_input = np.array([[normalized_speed]], dtype=np.float32)

                    # === Buat input command ===
                    command_map = {'LANEFOLLOW': 0, 'LEFT': 1, 'RIGHT': 2, 'STRAIGHT': 3}
                    command_idx = command_map.get(next_road_option.name.upper(), 0)
                    cmd_input = tf.keras.utils.to_categorical(command_idx, num_classes=4)
                    cmd_input = np.expand_dims(cmd_input, axis=0)
                    
                    # === Inference model ===

                    if model_type == 1:
                        control_output, pred_speed = model.predict([rgbd_input, speed_input, cmd_input], verbose=0)
                    else:
                        control_output = model.predict([rgb_input, speed_input, cmd_input], verbose=0)

                    # =====OUTCLASS======================
                    steer_out = float(control_output[0][0])
                    throttle_out = float(control_output[0][1])
                    brake_out = float(control_output[0][2])

                    THRESH = 0.5
                    
                    
                    if (extend_type == 1):
                        speed_val = (float(pred_speed[0][0]) + 1.0) * 0.5 * val_normalization_speed
                        error_speed = (speed_val - speed_kmh)/30
                        # print(f"speed: {speed_kmh}, pred: {speed_val}, error: {error_speed:.2f}")
                        Kp_break = 1.0
                        Kp_throttle = 0.1
                        if error_speed < 0:
                            brake_out = brake_out + abs(error_speed)*Kp_break
                        else:
                            throttle_out = throttle_out + abs(error_speed)*Kp_throttle

                    # === Buat kontrol kendaraan ===
                    control_cil.steer = np.clip(steer_out, -1.0, 1.0)
                    control_cil.throttle = np.round(np.clip(throttle_out, 0.0, 1.0),4)
                    control_cil.brake = np.round(np.clip(brake_out, 0.0, 1.0),4)     
                    control_cil.manual_gear_shift = False
                    
                    if control_cil.brake > 0.05:
                        control_cil.throttle = 0.0
                    else:
                        control_cil.brake = 0.0

                    steer_graph_display.update(noise_val=0,
                                            control_val=control_cil.steer,
                                            resultant_val=0,
                                            throttle_val=control_cil.throttle,
                                            brake_val=control_cil.brake,
                                            speed_val=speed_kmh)

                    curve_function = detect_curve_dir_ahead(route_manager, num_points=10)
            
                    while len(flat_waypoints) < 10:
                        flat_waypoints.extend([0.0, 0.0])
                    frame_saved += 1
                    world.player.apply_control(control_cil)

                    # Simpan nilai kontrol ke buffer setiap frame
                    control_buffer.append({
                        'frame': frame_count,
                        'throttle': float(control_cil.throttle),
                        'brake': float(control_cil.brake),
                        'speed': speed_kmh,
                    })
                    
                    current_command = command_idx
                    if args.task == "straight":
                        if (command_idx != 0) or stat_red or (curve_function !=0):
                            loop_lock = False
                            print(f"[INFO] Exit command idx {command_idx} or red light {stat_red} or curve {curve_function}")
                    elif args.task == "one_turn":
                        if(prev_command != current_command):
                            command_counter += 1
                        if command_counter >= 2:
                            loop_lock = False
                            print(f"[INFO] Command counter: {command_counter}")
                    else:
                        if(prev_command != current_command):
                            command_counter += 1
                        if command_counter >= 4:
                            loop_lock = False
                            print(f"[INFO] Command counter: {command_counter}")
                    prev_command = current_command
                else:
                    frame_skipped += 1
                    world.player.apply_control(control_agent)
                    print("skipped frame")

                hud.debug_status = (
                    f"frame:{frame_count} saved:{frame_saved} skipped:{frame_skipped} "
                    f"noise:{noise_steer:.2f} len(plan):{len(plan_list)} TL:{stat_at_traffic_light} "
                    f"Red:{stat_red} Roadopt:{next_road_option.name} Coll:{collision_counter} respawn:{respawn_counter} "
                    f"Steer:{steer_out:.3f} Throttle:{throttle_out:.3f} Brake:{brake_out:.3f} SpeedPred:{speed_val:.1f}"
                )

                frame_count += 1
                
                if frame_count > 3000:
                    episode_success = 0
                    loop_lock = False

                # di setiap tick (0.1 s misalnya)
                if speed_kmh != 0:
                    sum_speed_kmh += speed_kmh      # tambah kecepatan saat ini
                    speed_tick_count += 1

                    # speed_kmh = ...  # sudah kamu punya

                    # --- Histogram untuk speed dominan ---
                    speed_hist[bin_key(speed_kmh)] += 1

                    # --- Hitung percepatan ---
                    if prev_speed_kmh is not None:
                        dv_kmh = speed_kmh - prev_speed_kmh
                        dv_mps = dv_kmh * (1000.0 / 3600.0)
                        a_mps2 = dv_mps / DT_S

                        # Saring spike tak realistis (opsional)
                        if abs(a_mps2) <= ACC_SPIKE_LIMIT:
                            if a_mps2 > 0:
                                # Percepatan (positif)
                                if a_mps2 > max_accel_mps2:
                                    max_accel_mps2 = a_mps2
                                if (min_accel_mps2 is None) or (a_mps2 < min_accel_mps2):
                                    min_accel_mps2 = a_mps2
                            elif a_mps2 < 0:
                                # Deselerasi (negatif)
                                if a_mps2 < max_decel_mps2:  # ini mencari yang "paling negatif"
                                    max_decel_mps2 = a_mps2
                                if (min_decel_mps2 is None) or (a_mps2 > min_decel_mps2):
                                    # lebih besar (kurang negatif) = lebih dekat ke nol = deselerasi minimum
                                    min_decel_mps2 = a_mps2

                    prev_speed_kmh = speed_kmh

            else:
                episode_success = 0
                loop_lock = False
                print ("[INFO] Fatal collision")

            if frame_count % 1000 == 0:
                actor_count = len(world.world.get_actors())
                vehicle_count = len([a for a in world.world.get_actors().filter('vehicle.*')])
                print(f"[INFO] Frame: {frame_count} | Vehicles: {vehicle_count} | Actors: {actor_count} | Server FPS: {hud.server_fps:.1f} | Client FPS: {clock.get_fps():.1f}")
                
                # Logging performa setiap 1000 frame
                ram_gb = process.memory_info().rss / (1024**3)
                swap_used = psutil.swap_memory().used / (1024**3)
                print(f"[PERF] Frame: {frame_count} | Server FPS: {hud.server_fps:.1f} | Client FPS: {clock.get_fps():.1f} | RAM: {ram_gb:.2f} GB | Swap: {swap_used:.2f} GB")

                mem = psutil.virtual_memory()
                print(f"MemFree: {mem.free / (1024**3):.2f} GB, MemAvailable: {mem.available / (1024**3):.2f} GB")
                
    finally:
        # ketika episode selesai
        if speed_tick_count > 0:
            avg_speed = sum_speed_kmh / speed_tick_count   # ✅ rata-rata km/h
        else:
            avg_speed = 0.0

        # Speed dominan
        if speed_hist:
            dominant_speed_bin, count = max(speed_hist.items(), key=lambda kv: kv[1])
        else:
            dominant_speed_bin, count = None, 0

        def fmt(x):
            return "N/A" if (x is None or x == float("-inf") or x == float("inf")) else f"{x:.3f}"
        
        csv_file = f"/opt/carla-simulator/PythonAPI/examples/testing/{args.model_arch}_{args.trial_per_mode}_{args.testing_mode}_{args.weather}_{args.town}_{args.task}.csv"

        # Data yang ingin disimpan
        row = [
            fmt(max_accel_mps2),
            fmt(min_accel_mps2),
            fmt(max_decel_mps2),
            fmt(min_decel_mps2),
            dominant_speed_bin if dominant_speed_bin is not None else 'N/A',
            count,
            avg_speed,
            episode_success
        ]

        # Cek apakah file sudah ada
        file_exists = os.path.isfile(csv_file)

        # Buka file dengan append mode
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)

            # Jika file baru dibuat, tuliskan header
            if not file_exists:
                writer.writerow([
                    "MaxAccel", "MinAccel", "MaxDecel", "MinDecel",
                    "DominantSpeedBin", "Count", "AvgSpeed", "EpisodeSuccess"
                ])

            # Tulis baris data
            writer.writerow(row)
            print(f"[INFO] Write result to csv {csv_file}")

        # Simpan buffer ke CSV setelah simulasi selesai
        # output_csv = f"/opt/carla-simulator/PythonAPI/examples/testing/control_log_{args.model_arch}_run{args.run}.csv"
        # with open(output_csv, 'w', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=['frame', 'throttle', 'brake', 'speed'])
        #     writer.writeheader()
        #     writer.writerows(control_buffer)
        # print(f"[INFO] Saved control log to {output_csv} ({len(control_buffer)} entries)")

        if display_manager:
            display_manager.destroy()

        if world is not None:
            # Kembalikan setting simulator
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(False)

            # Hentikan dan hancurkan sensor sinkron (gunakan stop+destroy dalam try)
            for sensor in [rgb_center_sensor, depth_center_sensor, lidar_sensor]:
                try:
                    sensor.stop()
                    sensor.destroy()
                    print(f"[INFO] Destroyed {sensor.type_id} ID {sensor.id}")
                except Exception as e:
                    print(f"[WARNING] Failed to destroy sensor: {e}")

            # Hancurkan semua sensor dari world.sensors (collision, gnss)
            for sensor in world.sensors.values():
                try:
                    sensor.destroy()
                except Exception as e:
                    print(f"[WARNING] Failed to destroy sensor: {e}")

            # Debug status terakhir dari HUD
            if hasattr(world, 'hud') and hasattr(world.hud, 'debug_status'):
                print(f"[DEBUG STATUS] {world.hud.debug_status}")

            world.destroy()
        pygame.quit()

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================

args = None  # variabel global
def main():
    """Main method"""
    global args  # agar args bisa diakses dari fungsi lain
    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument('--crop-top', type=float, default=26.6,
                       help='Percentage to crop from top (default: 16%%)')
    argparser.add_argument('--crop-bottom', type=float, default=0.0,
                       help='Percentage to crop from bottom (default: 16%%)')
    argparser.add_argument(
        '--save-data',
        action='store_true',
        help='Save dataset (images and CSV) during simulation')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        # default='1280x720',
        default='1800x792',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--rescam',
        metavar='WIDTHxHEIGHT',
        # default='330x120',
        default='200x150',
        help='Cam resolution (default: 200x150)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',\
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "Constant"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (no pygame window rendering)')
    argparser.add_argument(
        '--vw',
        action='store_true',
        help='Run in validation mode')
    argparser.add_argument(
        '--vt',
        action='store_true',
        help='Run in validation mode')
    argparser.add_argument(
        '--task',
        default='straight',
        choices=['straight', 'one_turn', 'navigation', 'nav_dynamic'],
        help='task mode: straight | one_turn | navigation | nav_dynamic')
    argparser.add_argument(
        "--town", 
        type=str, 
        default="Town01", 
        help="Nama town CARLA (misal: Town01, Town02)")
    argparser.add_argument(
        "--weather", 
        type=str, 
        default="ClearNoon", 
        help="Cuaca CARLA (misal: ClearNoon, WetNoon, HardRainNoon)")
    argparser.add_argument(
        "--testing_mode",
        type=str,
        default="training_conditions",
        choices=["training_conditions", "new_town", "new_weather", "new_town_weather"],
        help="Pilih skenario evaluasi: training_conditions, new_town, new_weather, atau new_town_weather")
    argparser.add_argument(
        "--run",
        type=int,
        choices=[1, 2, 3, 4, 5],
        required=True,
        help="Nomor run eksperimen (1-5)"
    )
    argparser.add_argument(
        "--trial_per_mode",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Nomor run eksperimen (1-3)"
    )
    argparser.add_argument(
        "--model_saved",
        type=str,
        choices=["100k", "200k", "300k", "400k", "500k"],
        required=True,
        help="Checkpoint model yang akan digunakan (misal: 100k)"
    )
    argparser.add_argument(
        "--model_arch",
        type=str,
        choices=["cil", "mul_cil", "mul_cil_f", "mul_cil_e", "mul_cil_a2", "mul_cil_a2e"],
        required=True,
        help="Arsitektur model yang akan digunakan (cil atau mul_cil)"
    )
    argparser.add_argument(
        '--output',
        type=str,
        default="dataset/dataset_batch_001.h5",  # default kalau tidak diberi dari command-line
        help='Path file HDF5 output (misal dataset/dataset_batch_001.h5)')

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.widthcam, args.heightcam = [int(x) for x in args.rescam.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
