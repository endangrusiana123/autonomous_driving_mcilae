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
import h5py
import cv2  # ✅ Tambahkan ini agar OpenCV bisa digunakan
from itertools import islice
import gc  # tambahkan di atas
import queue
import psutil
import copy  # tambahkan di atas file
import random as pyrandom  # jangan bentrok dengan numpy.random
import fnmatch

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
from agents.navigation.behavior_agent import BehaviorAgent
from agents.navigation.basic_agent import BasicAgent

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
import h5py
from agents.tools.misc import get_trafficlight_trigger_location, is_within_distance

frame_count = 0  # Counter untuk jumlah frame yang sudah diproses

MAX_FRAME_PER_H5 = 200  # Jumlah frame per file
MAX_TOTAL_FRAME = 2_500_000

frame_global_buffer = {
    'rgb': [],
    'depth': [],
    'lidar_above': [],
    'lidar_ground': [],
    'metadata': []
}
global_batch_idx = 0  # Nomor batch file .h5

def get_latest_h5_index(output_dir):
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("cil_batch_") and f.endswith(".h5")]
    if not existing_files:
        return 0
    indices = []
    for fname in existing_files:
        try:
            idx = int(fname.replace("cil_batch_", "").replace(".h5", ""))
            indices.append(idx)
        except:
            pass
    if not indices:
        return 0
    return max(indices) + 1  # lanjut dari file berikutnya

def flush_global_buffer():
    global global_batch_idx, frame_global_buffer

    if len(frame_global_buffer['metadata']) == 0:
        return

    while True:
        filename = os.path.join(h5_dir, f"cil_batch_{global_batch_idx:05d}.h5")
        if not os.path.exists(filename):
            break
        global_batch_idx += 1

    rgb = np.stack(frame_global_buffer['rgb'])
    depth = np.stack(frame_global_buffer['depth'])
    lidar_above = np.stack(frame_global_buffer['lidar_above'])
    lidar_ground = np.stack(frame_global_buffer['lidar_ground'])
    metadata = np.array(frame_global_buffer['metadata'])

    metadata_columns = (
        [
            "frame_id", "speed", "road_option", "steer", "steer_noise", "steer_resultant",
            "throttle", "brake", "red_light", "at_traffic_light", "weather", 
            "vehicle_in_front", "is_curve", "is_junction",
            "speed_kmh_t", "speed_kmh_t_1", "speed_kmh_t_2", "speed_kmh_t_3"
        ]
        + [f"wp{i}_{ax}" for i in range(5) for ax in ("x", "y")]
    )
    
    with h5py.File(filename, "w") as f:
        f.create_dataset("rgb", data=rgb)
        f.create_dataset("depth", data=depth)
        f.create_dataset("lidar_above", data=lidar_above)
        f.create_dataset("lidar_ground", data=lidar_ground)

        for col_idx, col_name in enumerate(metadata_columns):
            col_data = metadata[:, col_idx]
            if col_name in ["road_option", "weather"]:
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset(col_name, data=[str(x) for x in col_data], dtype=dt)
            else:
                f.create_dataset(col_name, data=col_data.astype(np.float32))

    print(f"[H5] ✅ Saved {filename} with {len(metadata)} frames")
    frame_global_buffer = {k: [] for k in frame_global_buffer}
    global_batch_idx += 1

def save_frame_data_buffered(frame_id, rgb_views, depth_views, lidar_above, lidar_ground,
                             world, agent, steer_bycontroller, steer_noise, steer_resultant,
                             stat_red, stat_at_traffic_light, plan, flat_waypoints, weather_name, 
                             vehicle_in_front, is_curve, is_junction,
                             speed_kmh_t, speed_kmh_t_1, speed_kmh_t_2, speed_kmh_t_3):
    global frame_global_buffer

    if rgb_views is None or depth_views is None:
        return False

    speed_vec = world.player.get_velocity()
    speed = 3.6 * math.sqrt(speed_vec.x ** 2 + speed_vec.y ** 2 + speed_vec.z ** 2)
    control = world.player.get_control()
    road_option = plan[0][1].name if plan else "LANEFOLLOW"

    metadata = [
        frame_id, speed, road_option, steer_bycontroller, steer_noise, steer_resultant,
        control.throttle, control.brake, float(stat_red), float(stat_at_traffic_light), weather_name, 
        float(vehicle_in_front), float(is_curve), float(is_junction), 
        speed_kmh_t, speed_kmh_t_1, speed_kmh_t_2, speed_kmh_t_3 
    ] + flat_waypoints

    frame_global_buffer['rgb'].append(rgb_views.copy())    # shape: (3, 88, 200, 3)
    frame_global_buffer['depth'].append(depth_views.copy())  # shape: (3, 88, 200)
    frame_global_buffer['lidar_above'].append(lidar_above.copy())
    frame_global_buffer['lidar_ground'].append(lidar_ground.copy())
    frame_global_buffer['metadata'].append(metadata)

    if len(frame_global_buffer['metadata']) >= MAX_FRAME_PER_H5:
        flush_global_buffer()
    return True

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
        speed_norm_hist = [s / 40.0 for s in self.speed_history]

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
        self.local_waypoints = []  # Untuk menyimpan waypoint lokal (x, y)
        self.route_manager = route_manager
        self.agent = agent
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos

        self.display_enabled = self.display_man.display is not None
        if self.display_enabled:
            self.surface = pygame.Surface(display_man.get_display_size())
            self.display_man.add_sensor(self)

    def update(self):
        """Dapat dipanggil setiap frame — baik headless maupun GUI."""
        self._update_local_waypoints()

    def render(self):
        """Hanya untuk mode GUI"""
        if not self.display_enabled:
            return

        self.surface.fill((0, 0, 0))  # background hitam

        for i, (norm_x, norm_y) in enumerate(self.local_waypoints):
            draw_x = int(self.surface.get_width() / 2 + norm_y * 10)
            draw_y = int(self.surface.get_height() - norm_x * 10)
            color = (255, 0, 0) if i == 0 else (0, 255, 0)
            pygame.draw.circle(self.surface, color, (draw_x, draw_y), 5)

        offset = self.display_man.get_display_offset(self.display_pos)
        self.display_man.display.blit(self.surface, offset)

    def _update_local_waypoints(self):
        self.local_waypoints = []
        player_tf = self.world.player.get_transform()
        desired_waypoint_count = 5

        plan = self.route_manager.get_plan() if self.route_manager else []
        plan_list = list(plan)
        waypoints = [w[0] for w in plan_list[:desired_waypoint_count]]

        while len(waypoints) < desired_waypoint_count:
            if waypoints:
                waypoints.append(waypoints[-1])
            else:
                break

        origin_x, origin_y = None, None
        for i, wp in enumerate(waypoints):
            local_x, local_y = world_to_local(player_tf, wp.transform)
            if i == 0:
                origin_x, origin_y = local_x, local_y
            norm_x = round(local_x - origin_x, 2)
            norm_y = round(local_y - origin_y, 2)
            self.local_waypoints.append((norm_x, norm_y))

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

def lidar_to_histogram_bev(points, height=88, width=88, ground_thresh=0.3):
    # Ambil komponen
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # Filter azimuth: depan (-90° s.d. +90°)
    azimuth = np.degrees(np.arctan2(y_points, x_points))
    azimuth = (azimuth + 360) % 360
    mask_azimuth = (azimuth >= 270) | (azimuth <= 90)
    x_points = x_points[mask_azimuth]
    y_points = y_points[mask_azimuth]
    z_points = z_points[mask_azimuth]

    # Batasi area agar cocok dengan ukuran citra
    x_min, x_max = 0.0, 25.0
    y_min, y_max = -20.0, 20.0
    mask = (x_points >= x_min) & (x_points <= x_max) & (y_points >= y_min) & (y_points <= y_max)
    x_points = x_points[mask]
    y_points = y_points[mask]
    z_points = z_points[mask]

    # Mapping ke koordinat piksel (kolom = sumbu Y, baris = sumbu X)
    x_img = np.floor((y_points - y_min) / (y_max - y_min) * (width - 1)).astype(np.int32)
    y_img = height - 1 - np.floor((x_points - x_min) / (x_max - x_min) * (height - 1)).astype(np.int32)

    # Clip agar semua index valid
    np.clip(x_img, 0, width - 1, out=x_img)
    np.clip(y_img, 0, height - 1, out=y_img)

    # Mask atas-tanah vs tanah
    mask_above = (z_points > ground_thresh)

    # Akumulator int32 (biar tidak overflow saat hitung)
    acc_above = np.zeros((height, width), dtype=np.int32)
    acc_ground = np.zeros((height, width), dtype=np.int32)

    # Tambahkan hitungan secara vektorisasi
    coords = (y_img, x_img)
    np.add.at(acc_above, coords, mask_above.astype(np.int32))
    np.add.at(acc_ground, coords, (~mask_above).astype(np.int32))

    # Saturasi ke 0..255 dan cast ke uint8
    np.minimum(acc_above, 255, out=acc_above)
    np.minimum(acc_ground, 255, out=acc_ground)
    channel0 = acc_above.astype(np.uint8)   # above-ground
    channel1 = acc_ground.astype(np.uint8)  # ground

    return np.stack([channel0, channel1], axis=-1)

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

        # Pilih titik spawn acak dari daftar spawn_points
        spawn_point = random.choice(spawn_points)

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
                # ('Reverse:', control.reverse),
                # ('Hand brake:', control.hand_brake),
                # ('Manual:', control.manual_gear_shift),
                # 'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
                    
        if self.debug_status:
            wrapped = self.wrap_text(self.debug_status, 400)  # max 370px agar sesuai HUD baru
            self._info_text.extend(wrapped)
        
        self._info_text += [
            'Steer control (white), Noise (orange),',
            ' Resultant (green), Speed (red), Throttle (blue),',
            ' Brake (yellow):'
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
            # info_surface = pygame.Surface((320, self.dim[1]))
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
        pygame.init()
        pygame.font.init()
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
    weather_presets = {
        "ClearNoon": carla.WeatherParameters.ClearNoon,
        "ClearSunset": carla.WeatherParameters.ClearSunset,
        "WetNoon": carla.WeatherParameters.WetNoon,
        "HardRainNoon": carla.WeatherParameters.HardRainNoon
    }

    if args.weather is not None:
        weather_name = args.weather
    else:
        preset_keys = list(weather_presets.keys())
        weather_name = random.choice(preset_keys)
        print(f"[INFO] Cuaca tidak ditentukan, dipilih random: {weather_name}")
    selected_weather = weather_presets[weather_name]

    world.world.set_weather(selected_weather)
    print(f"[INFO] Weather changed to: {weather_name}")
    return weather_name

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

def carla_rgb_to_array(img):
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)[:, :, :3]  # BGR
    arr = arr[:, :, ::-1]          # BGR -> RGB
    arr = arr[62:, :, :]           # crop top 32 px -> (88, 200, 3)
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
    gray  = gray[62:, :]           # crop top 32 px -> (88, 200)
    gray  = cv2.medianBlur(gray, 5)
    return gray

def process_lidar(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))

    bev = lidar_to_histogram_bev(points, height=88, width=198)
    lidar_above = bev[:, :, 0]
    lidar_ground = bev[:, :, 1]
    return lidar_above, lidar_ground

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

def is_logging_complete():
    return (global_batch_idx * MAX_FRAME_PER_H5 + len(frame_global_buffer['metadata'])) >= MAX_TOTAL_FRAME

# ================== LEAD VEHICLE (SINUSOIDAL GAP, RED-LIGHT AWARE) ==================
# CARLA 0.9.x — Python 3.8-compatible type hints
import math, random
from typing import Optional
import carla

# ---------- Util konversi ----------
def kmh_to_ms(kmh: float) -> float:
    return kmh / 3.6

def ms_to_kmh(ms: float) -> float:
    return ms * 3.6

# ---------- Parameter utama ----------
LEAD_GAP_M             = 7.0       # jarak dasar (meter)
LEAD_SIN_GAP_AMPL_M    = 16.0       # amplitudo osilasi jarak (lead mendekat/menjauh)
LEAD_MIN_GAP_M         = 0.5       # jarak minimum agar tidak terlalu dekat
LEAD_JUNCTION_BOOST_M  = 10.0       # tambahan jarak saat tikungan/persimpangan
LEAD_HEIGHT_OFFSET     = 0.2        # sedikit naik agar tidak clipping

LEAD_SIN_PERIOD_S      = 20.0       # periode sinus (detik) untuk osilasi jarak
LEAD_SIN_PHASE         = 0.0        # fase awal sinus (radian)

# ---------- Helper waypoint ----------
def _advance_waypoint(map_, wp, distance: float):
    """Maju sepanjang centerline sekitar 'distance' meter.
    Memilih cabang dengan deviasi heading terkecil ketika ada banyak next()."""
    if wp is None:
        return None
    step = 1.5  # m
    travelled = 0.0
    cur = wp
    while travelled < distance:
        nxts = cur.next(step)
        if not nxts:
            return None
        if len(nxts) == 1:
            chosen = nxts[0]
        else:
            # pilih kelanjutan heading terdekat
            def heading_yaw(w):
                return w.transform.rotation.yaw
            cur_yaw = heading_yaw(cur)
            chosen = min(
                nxts,
                key=lambda w: abs((heading_yaw(w) - cur_yaw + 180) % 360 - 180)
            )
        travelled += step
        cur = chosen
    return cur

def _is_ahead_junction_or_turn(hero_wp: carla.Waypoint, lookahead_m: float = 8.0) -> bool:
    """Deteksi cepat apakah di depan ada junction/turn.
    Kita sampling beberapa waypoint ke depan dan cek perubahan heading besar atau masuk junction."""
    if hero_wp is None:
        return False
    yaw0 = hero_wp.transform.rotation.yaw
    total = 0.0
    step = 2.0
    cur = hero_wp
    while total < lookahead_m:
        nxts = cur.next(step)
        if not nxts:
            break
        cur = nxts[0]
        total += step
        if cur.is_junction:
            return True
        yaw = cur.transform.rotation.yaw
        dyaw = abs((yaw - yaw0 + 180) % 360 - 180)
        if dyaw > 20.0:  # belokan cukup tajam
            return True
    return False

# ================== KELAS PENGELOLA LEAD ==================
class LeadVehicleManager:
    def __init__(self, world, hero_actor, gap_m: float = LEAD_GAP_M):
        self.world = world          # type: carla.World
        self.map = world.get_map()
        self.hero = hero_actor      # type: carla.Vehicle
        self.gap_m = gap_m
        self.lead = None            # type: Optional[carla.Vehicle]
        self._bp_filter = "vehicle.*"
        self._last_red_state = False

    # --- RED LIGHT AWARE ---
    def _hero_at_red_light(self) -> bool:
        """True bila hero sedang di traffic light berstatus RED."""
        try:
            if not self.hero.is_at_traffic_light():
                return False
            tl = self.hero.get_traffic_light()
            return (tl is not None) and (tl.get_state() == carla.TrafficLightState.Red)
        except Exception:
            return False

    # --- SPEED LIMIT (m/s) dari jalan yang dilalui ---
    def _current_speed_limit_ms(self, loc: Optional[carla.Location] = None) -> float:
        """Ambil speed limit (m/s) dari waypoint lokasi saat ini,
        fallback ke hero.get_speed_limit(), default 30 km/h bila gagal."""
        try:
            if loc is None:
                loc = self.hero.get_transform().location
            wp = self.map.get_waypoint(loc, project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            if (wp is not None) and hasattr(wp, "speed_limit") and (wp.speed_limit > 0.0):
                return kmh_to_ms(float(wp.speed_limit))  # km/h -> m/s
        except Exception:
            pass
        try:
            return kmh_to_ms(float(self.hero.get_speed_limit()))
        except Exception:
            return kmh_to_ms(30.0)  # default aman

    # --- Spawn lead (sekali, otomatis di depan hero) ---
    def spawn_lead(self, blueprint_filter="vehicle.*", safe=True):
        self._bp_filter = blueprint_filter
        bp_lib = self.world.get_blueprint_library()

        # Filter blueprint: hanya roda 4, bukan truck/van/pickup, blacklist beberapa
        blacklist = [
            "vehicle.carlamotors.",   # semua varian
            "vehicle.ford.ambulance",
            "vehicle.mercedes.sprinter",
            "vehicle.volkswagen.",    # semua varian
            "vehicle.mitsubishi.fusorosa"
        ]
        def is_blacklisted(bp_id: str) -> bool:
            for item in blacklist:
                if item.endswith("."):
                    if bp_id.startswith(item):
                        return True
                else:
                    if bp_id == item:
                        return True
            return False

        bps = [
            bp for bp in bp_lib.filter(blueprint_filter)
            if int(bp.get_attribute('number_of_wheels').as_int()) == 4
            and ('truck' not in bp.id)
            and ('van' not in bp.id)
            and ('pickup' not in bp.id)
            and ('carlacola' not in bp.id)
            and not is_blacklisted(bp.id)
        ]
        if not bps:
            print("[LEAD] Tidak ada blueprint yang valid setelah filter")
            return None

        bp = random.choice(bps)
        bp.set_attribute("role_name", "lead")

        hero_tf = self.hero.get_transform()
        wp = self.map.get_waypoint(hero_tf.location, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        # spawn awal memakai jarak dasar (nanti saat tick → dinamis)
        target_wp = _advance_waypoint(self.map, wp, max(self.gap_m, LEAD_MIN_GAP_M)) or wp
        tf = target_wp.transform
        tf.location.z += LEAD_HEIGHT_OFFSET

        if safe:
            radius = 5.0
            actors = self.world.get_actors().filter("vehicle.*")
            if any(a.id != self.hero.id and a.get_location().distance(tf.location) < radius for a in actors):
                print("[LEAD] Spawn aborted (safe mode) karena area padat")
                return None

        self.lead = self.world.try_spawn_actor(bp, tf)
        if not self.lead:
            print("[LEAD] Gagal spawn lead (posisi terisi)")
            return None

        if hasattr(self.lead, "set_autopilot"):
            self.lead.set_autopilot(False)

        print(f"[LEAD] Spawned {self.lead.type_id} id={self.lead.id}")
        return self.lead

    def _ensure_lead_exists(self):
        if self.lead is None:
            try:
                self.spawn_lead(self._bp_filter)
            except RuntimeError as e:
                print(f"[LEAD] Spawn failed: {e}")

    def tick(self,red_light=False):
        """Panggil setiap frame (mode sync: setelah world.tick())."""
        t_now = self.world.get_snapshot().timestamp.elapsed_seconds

        # 1) Red-light handling: destroy saat RED, respawn lagi setelah bebas
        at_red = red_light
        if at_red:
            if self.lead is not None:
                try:
                    print("[LEAD] Red light → remove lead")
                    self.lead.destroy()
                except Exception:
                    pass
                self.lead = None
            self._last_red_state = True
            return
        else:
            # baru saja keluar dari RED → izinkan respawn normal
            if self._last_red_state:
                self._last_red_state = False

        # 2) Pastikan lead ada
        self._ensure_lead_exists()
        if self.lead is None:
            return

        # 3) Dapatkan waypoint hero terkini
        hero_tf = self.hero.get_transform()
        hero_wp = self.map.get_waypoint(hero_tf.location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        if hero_wp is None:
            # tidak di jalan → berhentikan lead
            self.lead.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            return

        # 4) Hitung jarak target dinamis (SINUSOIDAL RELATIF terhadap hero)
        omega = 2.0 * math.pi / max(LEAD_SIN_PERIOD_S, 1e-6)
        gap_dyn = self.gap_m + LEAD_SIN_GAP_AMPL_M * math.sin(omega * t_now + LEAD_SIN_PHASE)
        gap_dyn = max(gap_dyn, LEAD_MIN_GAP_M)

        # Saat ada belokan / junction di depan, tambahkan boost agar lead "terus maju"
        if _is_ahead_junction_or_turn(hero_wp, lookahead_m=10.0):
            gap_dyn += LEAD_JUNCTION_BOOST_M

        # 5) Ambil target waypoint di depan hero sejauh gap dinamis
        target_wp = _advance_waypoint(self.map, hero_wp, gap_dyn)
        if target_wp is None:
            # tidak ada kelanjutan jalan → berhentikan lead di tempatnya
            self.lead.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            return

        tgt_tf = target_wp.transform
        tgt_tf.location.z += LEAD_HEIGHT_OFFSET

        # 6) Set kecepatan lead mengikuti arah jalan, dengan batas speed limit lokasi target
        vmax = self._current_speed_limit_ms(loc=tgt_tf.location)
        # Trik kecil: biar tidak “mepet” terus, kita kasih kecepatan yang
        # proporsional terhadap gap (semakin jauh → sedikit lebih cepat),
        # namun dibatasi oleh speed limit.
        # v_prop ~ 0..1 untuk gap 0..(gap_m + ampl + boost)
        max_plausible_gap = max(LEAD_MIN_GAP_M, self.gap_m + LEAD_SIN_GAP_AMPL_M + LEAD_JUNCTION_BOOST_M)
        v_prop = min(1.0, max(0.2, gap_dyn / max_plausible_gap))
        v_ms = v_prop * vmax

        # 7) Tether posisi ke centerline di depan hero (relatif), lalu set velocity
        #    (Tether memastikan lead selalu "di depan" hero, tidak nyasar ke lane lain)
        try:
            self.lead.set_transform(tgt_tf)
        except RuntimeError:
            # kemungkinan lead hancur/keluar world → reset agar respawn
            self.lead = None
            return

        yaw_rad = math.radians(tgt_tf.rotation.yaw)
        vx = v_ms * math.cos(yaw_rad)
        vy = v_ms * math.sin(yaw_rad)
        try:
            self.lead.set_target_velocity(carla.Vector3D(x=vx, y=vy, z=0.0))
        except RuntimeError:
            self.lead = None

    def destroy(self):
        if self.lead is not None:
            try:
                self.lead.destroy()
            except Exception:
                pass
            self.lead = None
            print("[LEAD] Destroyed")

# ======== Contoh pemakaian minimal ========
def setup_lead_manager(world, hero, gap: float = LEAD_GAP_M):
    return LeadVehicleManager(world, hero, gap_m=gap)

# --- contoh loop ---
# lead_mgr = setup_lead_manager(world, hero_vehicle, gap=LEAD_GAP_M)
# while True:
#     world.tick()
#     lead_mgr.tick()
#     # ... logging sensor & kontrol hero ...

"""
Sederhana: Oncoming Traffic Manager untuk CARLA 0.9.x (Python 3.8 kompatibel)
- Spawn NPC di lajur berlawanan (berpapasan dengan hero)
- Autopilot via Traffic Manager (tanpa lane change)
- Despawn sederhana jika terlalu jauh di belakang/umum
- Respawn agar jumlah tetap

Catatan:
- Dibuat ringan & longgar, tanpa banyak aturan/cek ketat.
- Type hints disesuaikan untuk Python 3.8 (tanpa operator union "|").
"""
import math
import random
import fnmatch
from typing import Optional, Tuple, List
import carla

# --------------------------- Util sederhana ---------------------------

def _sign(x: int) -> int:
    return 1 if x >= 0 else -1


def _advance(wp: carla.Waypoint, dist_m: float) -> carla.Waypoint:
    step, d, cur = 2.0, 0.0, wp
    while d < dist_m:
        nxt = cur.next(step)
        if not nxt:
            break
        cur = nxt[0]
        d += step
    return cur


def _rewind(wp: carla.Waypoint, dist_m: float) -> carla.Waypoint:
    step, d, cur = 2.0, 0.0, wp
    while d < dist_m:
        prv = cur.previous(step)
        if not prv:
            break
        cur = prv[0]
        d += step
    return cur


def _hero_to_target_angle_deg(hero_tf: carla.Transform, loc: carla.Location) -> float:
    """Sudut 2D antara vektor forward hero dan vektor hero→target (derajat)."""
    hx, hy = hero_tf.get_forward_vector().x, hero_tf.get_forward_vector().y
    tx, ty = loc.x - hero_tf.location.x, loc.y - hero_tf.location.y
    f = math.hypot(hx, hy) or 1.0
    t = math.hypot(tx, ty) or 1.0
    dot = (hx/f)*(tx/t) + (hy/f)*(ty/t)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


# ----------------------- Opposite lane yang simpel -----------------------

def _get_opposite_lane_wp(carla_map: carla.Map, hero_wp: carla.Waypoint) -> Optional[carla.Waypoint]:
    """
    Cari lajur berlawanan yang masih 1 road_id dan lane_type Driving.
    Tidak terlalu ketat: cukup beda tanda lane_id ⇒ dianggap opposite.
    """
    sgn = _sign(hero_wp.lane_id)
    for side in ("left", "right"):
        wp = hero_wp
        while True:
            wp = wp.get_left_lane() if side == "left" else wp.get_right_lane()
            if wp is None or wp.lane_type != carla.LaneType.Driving:
                break
            if wp.road_id != hero_wp.road_id:
                continue
            if _sign(wp.lane_id) != sgn:
                return wp
    return None


# ----------------------- Pemilih blueprint kendaraan ----------------------

def _pick_vehicle_bp(world: carla.World) -> Optional[carla.ActorBlueprint]:
    lib = world.get_blueprint_library()
    deny = ["*truck*", 
            "*van*", 
            "*pickup*", 
            "vehicle.carlacola", 
            "vehicle.mercedes.sprinter",
            "vehicle.carlamotors.*",
            "vehicle.ford.ambulance",
            "vehicle.mercedes.sprinter",
            "vehicle.volkswagen.*",
            "vehicle.mitsubishi.fusorosa"]

    def ok(bp: carla.ActorBlueprint) -> bool:
        if not bp.has_attribute("number_of_wheels"):
            return False
        if int(bp.get_attribute("number_of_wheels").as_int()) != 4:
            return False
        for pat in deny:
            if fnmatch.fnmatch(bp.id, pat):
                return False
        return True

    pool = [bp for bp in lib.filter("vehicle.*") if ok(bp)]
    return random.choice(pool) if pool else None


# ----------------------------- Manager ringkas ----------------------------
class OncomingTrafficManager:
    def __init__(self,
                 world: carla.World,
                 client: carla.Client,
                 tm_port: int,
                 hero_vehicle: carla.Vehicle,
                 target_count: int = 6,
                 start_offset_m: float = 100.0,
                 spacing_m: float = 35.0,
                 speed_kph: float = 35.0,
                 far_despawn_dist_m: float = 160.0,
                 rear_despawn_dist_m: float = 60.0,
                 spawn_fov_deg: float = 160.0):
        self.world = world
        self.client = client
        self.tm_port = tm_port
        self.hero = hero_vehicle
        self.target_count = int(target_count)
        self.start_offset_m = float(start_offset_m)
        self.spacing_m = float(spacing_m)
        self.speed_kph = float(speed_kph)
        self.far_despawn_dist_m = float(far_despawn_dist_m)
        self.rear_despawn_dist_m = float(rear_despawn_dist_m)
        self.spawn_fov_deg = float(spawn_fov_deg)

        self.tm = self.client.get_trafficmanager(self.tm_port)
        try:
            self.tm.set_synchronous_mode(True)
        except Exception:
            pass

        self.actors: List[carla.Vehicle] = []

    # ---- cari anchor opposite lane sederhana ----
    def _get_anchor(self) -> Tuple[Optional[carla.Map], Optional[carla.Waypoint]]:
        m = self.world.get_map()
        hero_tf = self.hero.get_transform()
        hero_wp = m.get_waypoint(hero_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not hero_wp:
            return None, None
        opp = _get_opposite_lane_wp(m, hero_wp)
        return m, opp

    def _spawn_one(self, base_wp: carla.Waypoint, back_offset_m: float) -> Optional[carla.Vehicle]:
        """
        Tempatkan kendaraan di lajur lawan arah, beberapa meter "sejajar-di-depan" hero.
        Sederhana: ambil base_wp (opposite), lalu previous() untuk memindah pos agar muncul di depan hero.
        """
        wp = _rewind(base_wp, back_offset_m)
        tf = wp.transform
        tf.location.z += 0.2

        # Pastikan masih di depan (dengan FOV lebar, default 160°)
        hero_tf = self.hero.get_transform()
        if _hero_to_target_angle_deg(hero_tf, tf.location) > (self.spawn_fov_deg * 0.5):
            return None

        bp = _pick_vehicle_bp(self.world)
        if not bp:
            return None
        bp.set_attribute("role_name", "oncoming")

        veh = self.world.try_spawn_actor(bp, tf)
        if not veh:
            return None

        # Autopilot + setelan ringan
        veh.set_autopilot(True, self.tm_port)
        try:
            self.tm.auto_lane_change(veh, False)
        except AttributeError:
            self.tm.set_auto_lane_change(veh, False)
        self.tm.ignore_lights_percentage(veh, 0)
        self.tm.distance_to_leading_vehicle(veh, 6.0)

        # Kecepatan sederhana: atur selisih dari limit
        diff = int(max(-50, min(50, 50.0 - float(self.speed_kph))))
        self.tm.vehicle_percentage_speed_difference(veh, diff)

        self.actors.append(veh)
        return veh

    # ---- public: spawn awal ----
    def spawn_initial(self) -> None:
        m, opp = self._get_anchor()
        if not opp:
            return
        base = _rewind(opp, self.start_offset_m)
        i, tries = 0, 0
        while len(self.actors) < self.target_count and tries < self.target_count * 5:
            back = i * self.spacing_m
            if self._spawn_one(base, back):
                i += 1
            tries += 1

    # ---- public: panggil setiap tick ----
    def tick(self) -> None:
        hero_tf = self.hero.get_transform()
        hero_loc = hero_tf.location
        fwd = hero_tf.get_forward_vector()

        survivors: List[carla.Vehicle] = []
        for v in self.actors:
            if not v.is_alive:
                continue
            loc = v.get_location()
            d = loc.distance(hero_loc)

            # Sudut negatif (dot < 0) terhadap forward = di belakang
            vx, vy = loc.x - hero_loc.x, loc.y - hero_loc.y
            dot = vx * fwd.x + vy * fwd.y
            is_behind = dot < 0

            # Despawn sederhana
            if d > self.far_despawn_dist_m:
                try:
                    v.destroy()
                except Exception:
                    pass
                continue
            if is_behind and d > self.rear_despawn_dist_m:
                try:
                    v.destroy()
                except Exception:
                    pass
                continue
            survivors.append(v)
        self.actors = survivors

        # Respawn jika kurang
        need = self.target_count - len(self.actors)
        if need <= 0:
            return
        m, opp = self._get_anchor()
        if not opp:
            return
        base = _rewind(opp, self.start_offset_m)
        i, tries = 0, 0
        while i < need and tries < need * 5:
            back = (len(self.actors) + i) * self.spacing_m
            if self._spawn_one(base, back):
                i += 1
            tries += 1

    def destroy(self):
        for v in getattr(self, "actors", []):
            try: v.destroy()
            except: pass
        self.actors = []

def _dot(a, b):
    return a.x*b.x + a.y*b.y + a.z*b.z

def vehicles_in_laser_beam(world, hero, max_dist=18.0, half_width=0.50):
    """
    world      : carla.World
    hero       : actor vehicle hero
    max_dist   : panjang sinar (m)
    half_width : setengah lebar sinar (m). Laser ~ strip selebar 2*half_width
    """
    hero_tf   = hero.get_transform()
    origin    = hero_tf.location
    fwd       = hero_tf.get_forward_vector()
    right     = hero_tf.get_right_vector()

    hits = []
    actors = world.get_actors().filter('vehicle.*')

    for v in actors:
        if v.id == hero.id:
            continue

        # Pusat bbox di world (bbox.location relatif ke aktor → transform ke world)
        t  = v.get_transform()
        bb = v.bounding_box
        bb_world_center = t.transform(bb.location)

        # Vektor relatif dari hero ke pusat bbox
        rel = carla.Vector3D(
            bb_world_center.x - origin.x,
            bb_world_center.y - origin.y,
            bb_world_center.z - origin.z
        )

        # Komponen sepanjang arah maju (longitudinal) dan samping (lateral)
        s_long = _dot(rel, fwd)                    # jarak di depan (+) / belakang (−) sepanjang heading hero
        if s_long <= 0.0 or s_long > max_dist:     # hanya yang benar-benar di depan dan dalam jangkauan
            continue

        # Jarak lateral dari garis tengah hero → koreksi dengan setengah lebar bbox target (extent.y)
        lateral_dist_edge = abs(_dot(rel, right)) - bb.extent.y

        # Dianggap dalam sinar jika "tebal" strip masih mencakup sisi bbox
        if lateral_dist_edge <= half_width:
            hits.append(v)

    return hits

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

def detect_junction_ahead(vehicle, world, max_distance=20.0, step=2.0):
    """
    Deteksi apakah ada junction/percabangan di depan dalam radius tertentu.
    Hanya mengembalikan True/False.
    """
    ego_loc = vehicle.get_location()
    ego_wp = world.get_map().get_waypoint(ego_loc)

    distance = 0.0
    wp = ego_wp
    while distance < max_distance:
        next_wps = wp.next(step)
        if not next_wps:
            break
        wp = next_wps[0]
        distance += step
        if wp.is_junction:
            return True
    return False

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

def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """
    global frame_count  # ✅ agar sinkron dengan fungsi lain
    
    pygame.init()
    pygame.font.init()
    world = None
    args.sync = True
    args.filter = 'vehicle.nissan.micra'
    args.loop = True
    args.save_data = True
    display_manager = None  # <- tambahkan ini sebelum try
    process = psutil.Process(os.getpid())

    try:
        if args.seed:
            random.seed(args.seed)

        client = carla.Client(args.host, args.port)
        client.set_timeout(6)

        traffic_manager = client.get_trafficmanager()
        sim_world = client.load_world('Town01')

        if args.sync:
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            settings.no_rendering_mode = False
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

        if not args.lead_vehicle and args.add_traffic:
            rand_num_vehicle = 70
            # Pilih random dari daftar nilai
            # rand_num_vehicle = random.randint(30, 70)  # inklusif, bisa 30 sampai 70
            # Panggil fungsi spawn_traffic dengan nilai terpilih
            spawn_traffic(world.world, client, num_vehicles=rand_num_vehicle, safe=True)
            print(f"Jumlah kendaraan yang di-spawn: {rand_num_vehicle}")
        else:
            print(f"Tidak ada traffic vehicle")

        weather_name = change_weather(world)
        controller = KeyboardControl(world)

        agent = CustomAgent(world.player, behavior='normal')        
        # Ambil referensi ke lateral controller sekali saja sebelum loop
        lp  = agent.get_local_planner()
        vc  = lp._vehicle_controller
        lat = vc._lat_controller
        lon = vc._lon_controller
        # lat._k_p = 1.95
        # lat._k_i = 0.05
        # lat._k_d = 0.2
        target_speed_kmh = 30.0  # kecepatan target

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
            bp.set_attribute('sensor_tick', '0.1')  # 20 FPS; bisa 0.1 untuk 10 FPS
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

        # Left RGB (geser ke kiri + yaw +30°)
        rgb_left_sensor, rgb_left_queue = spawn_sync_sensor(
            world.world, rgb_bp,
            carla.Transform(
                location=carla.Location(x=CAM_LOC.x, y=+OFFSET_Y,     z=CAM_LOC.z),
                rotation=carla.Rotation(pitch=PITCH, yaw=YAW_LEFT,   roll=0.0)
            ),
            world.player
        )

        # Right RGB (geser ke kanan + yaw -30°)
        rgb_right_sensor, rgb_right_queue = spawn_sync_sensor(
            world.world, rgb_bp,
            carla.Transform(
                location=carla.Location(x=CAM_LOC.x, y=-OFFSET_Y,     z=CAM_LOC.z),
                rotation=carla.Rotation(pitch=PITCH, yaw=YAW_RIGHT,  roll=0.0)
            ),
            world.player
        )

        # =========================================================
        # ==============  DEPTH CAMERAS: L/C/R  ===================
        # =========================================================
        depth_bp = setup_camera_bp(bp_lib.find('sensor.camera.depth'), IMG_W, IMG_H, FOV)
        # rekomendasi: fixed_delta_seconds = 0.05 (20 FPS sim) atau 0.1 (10 FPS sim)
        # kalau fixed_delta_seconds=0.05, set sensor_tick depth = 0.10 (kelipatan 2 frame)
        depth_bp.set_attribute('sensor_tick', '0.1')  # 0.10 detik per frame depth (10 FPS)

        # Center Depth
        depth_center_sensor, depth_center_queue = spawn_sync_sensor(
            world.world, depth_bp,
            carla.Transform(
                location=carla.Location(x=CAM_LOC.x, y=0.0,           z=CAM_LOC.z),
                rotation=carla.Rotation(pitch=PITCH, yaw=YAW_CENTER, roll=0.0)
            ),
            world.player
        )

        # Left Depth
        depth_left_sensor, depth_left_queue = spawn_sync_sensor(
            world.world, depth_bp,
            carla.Transform(
                location=carla.Location(x=CAM_LOC.x, y=+OFFSET_Y,     z=CAM_LOC.z),
                rotation=carla.Rotation(pitch=PITCH, yaw=YAW_LEFT,   roll=0.0)
            ),
            world.player
        )

        # Right Depth
        depth_right_sensor, depth_right_queue = spawn_sync_sensor(
            world.world, depth_bp,
            carla.Transform(
                location=carla.Location(x=CAM_LOC.x, y=-OFFSET_Y,     z=CAM_LOC.z),
                rotation=carla.Rotation(pitch=PITCH, yaw=YAW_RIGHT,  roll=0.0)
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
        lidar_bp.set_attribute('sensor_tick', '0.1')
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
            if sp.location.distance(hero_location) >= 60.0
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
        episode_count = 0
        start_logging = 0

        if args.lead_vehicle:
            # sekali saat init
            lead_mgr = setup_lead_manager(world.world, world.player, gap=20.0)

            tm_port = 8000
            tm = client.get_trafficmanager(tm_port)
            tm.set_synchronous_mode(True)  # cocokkan dengan world sync

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
            world.tick(clock, agent)    

            rgb_left_image = rgb_left_queue.get(timeout=2.0)
            rgb_center_image = rgb_center_queue.get(timeout=2.0)
            rgb_right_image = rgb_right_queue.get(timeout=2.0)

            depth_left_image = depth_left_queue.get(timeout=2.0)
            depth_center_image = depth_center_queue.get(timeout=2.0)
            depth_right_image = depth_right_queue.get(timeout=2.0)

            lidar_data = lidar_queue.get(timeout=2.0)

            # Di game loop, setelah get() dari queue:
            rgb_left   = carla_rgb_to_array(rgb_left_image)
            rgb_center = carla_rgb_to_array(rgb_center_image)
            rgb_right  = carla_rgb_to_array(rgb_right_image)

            depth_left   = carla_depth_to_gray(depth_left_image)
            depth_center = carla_depth_to_gray(depth_center_image)
            depth_right  = carla_depth_to_gray(depth_right_image)

            rgb_views = np.stack([rgb_left, rgb_center, rgb_right], axis=0)     # shape (3, 88, 200, 3)
            depth_views = np.stack([depth_left, depth_center, depth_right], axis=0)  # shape (3, 88, 200)

            lidar_above, lidar_ground = process_lidar(lidar_data)

            seq_display.update(
                {'left': rgb_left, 'center': rgb_center, 'right': rgb_right},
                {'left': depth_left, 'center': depth_center, 'right': depth_right},
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
            # print(f"0:{speed_kmh_t} 1:{speed_kmh_t_1} 2:{speed_kmh_t_2} 3:{speed_kmh_t_3}")
            
            if collision_counter < 1:
                if not(collision_counter > prev_collision_counter):
                    collision_counter = 0
                prev_collision_counter = collision_counter
                control = agent.run_step()
                control_cadangan = control
                control.manual_gear_shift = False
                
                # Ambil semua traffic light dari dunia (tetap sama)
                lights_list = world.world.get_actors().filter("*traffic_light*")

                # Panggil fungsi milik BasicAgent secara langsung
                # NOTE: BehaviorAgent mewarisi BasicAgent, jadi agent pasti punya method ini.
                affected, tl_actor = agent._affected_by_traffic_light(lights_list)

                # Konversi ke flag yang kamu simpan di metadata
                stat_red = bool(affected)
                stat_at_traffic_light = isinstance(tl_actor, carla.TrafficLight)

                # Cek apakah ada kendaraan di depan (model laser)
                vehicles_in_front = vehicles_in_laser_beam(
                    world.world, world.player,
                    max_dist=22.0,      # panjang “laser”
                    half_width=0.50     # setengah lebar strip; 0.3–0.7 biasanya oke
                )
                vehicle_in_front = int(len(vehicles_in_front) > 0)

                # hero_stop_obstacle = vehicles_in_laser_beam(
                #     world.world, world.player,
                #     max_dist=10.0,      # panjang “laser”
                #     half_width=0.50     # setengah lebar strip; 0.3–0.7 biasanya oke
                # )
                # hero_stop = int(len(hero_stop_obstacle) > 0)

                if args.lead_vehicle:
                    lead_mgr.tick(red_light=stat_red)

                if agent is not None and hasattr(agent, 'get_local_planner'):
                    plan = route_manager.get_plan()

                    plan_list = list(plan)
                    if len(plan_list) < 8:
                        # control.brake = 1
                        # world.player.apply_control(control)
                        world.hud.notification("Target reached,searching for another target.", seconds=4.0)
                        print("Target reached,searching for another target. Quit")
                        loop_lock = False
                                      
                    next_road_option = plan[0][1]
                    
                    if not noise_stat:
                        delta_frame_counter = frame_count-prev_frame_counter
                    else:
                        delta_frame_counter = 0

                    if  not noise_stat and abs(control.steer) < 0.1 and delta_frame_counter > random.randint(60, 120):
                        if pyrandom.random() < 0.1:
                            noise_stat = True
                            noise_steer = 0
                            # noise_mag = random.randint(6, 36) #range noise 0.06 to 0.36
                            noise_mag = random.randint(6, 26) #range noise 0.06 to 0.36
                            if pyrandom.random() < 0.5:
                                noise_dir = True
                            else:
                                noise_dir = False

                    if noise_stat and noise_counter< noise_mag:                        
                        if noise_dir:
                            if noise_counter < int(noise_mag / 2):
                                noise_steer += 0.02
                            else:
                                noise_steer -= 0.02
                        else:
                            if noise_counter < int(noise_mag / 2):
                                noise_steer -= 0.02
                            else:
                                noise_steer += 0.02
                        noise_counter += 1

                        prev_frame_counter = frame_count
                    else:
                        noise_stat = False
                        noise_counter = 0
                        noise_steer = 0

                    world.current_noise_steer = noise_steer
                    control_steer_signal = control.steer
                    world.current_steer_signal = control_steer_signal

                    #-------------------------------------------------------------------------------------
                    lat._dt = 0.10
                    lon._dt = 0.10
                    # lat._k_p = 1.95; lat._k_i = 0.1; lat._k_d = 0.1; lat._dt = 0.10
                    lat._k_p = 0.7; lat._k_i = 0.0; lat._k_d = 0.0; lat._dt = 0.10
                    # lon._k_p = 0.07; lon._k_i = 0.1; lon._k_d = 0.05; lon._dt = 0.10
                    # lon._k_p = 1.0; lon._k_i = 0.05; lon._k_d = 0.0; lon._dt = 0.10
                    is_curve = detect_curve_dir_ahead(route_manager, num_points=12)
                    # print(f"curve:{is_curve}")
                    is_junction = detect_junction_ahead(world.player,world.world,max_distance=14)
                    is_junction_for_curve = detect_junction_ahead(world.player,world.world,max_distance=22)
                    # print(f"juction:{is_junction}")
                    
                    # if not (next_road_option.name == "LANEFOLLOW"):
                    #     is_junction = False
                        
                    if (
                        (is_junction_for_curve == True) 
                        or (not (next_road_option.name == "LANEFOLLOW"))
                    ):
                        is_curve = 0

                    if(
                        (is_curve == 0)
                        and (
                            (next_road_option.name == "LANEFOLLOW")
                            or (next_road_option.name == "STRAIGHT")
                        )
                    ):  
                        if(
                            (vehicle_in_front == 0)
                            and (not is_junction)
                        ):
                            if(speed_kmh < 22):
                                agent._local_planner._vehicle_controller.max_throt = 0.75
                                agent._local_planner._vehicle_controller.max_steer_step = 0.1
                                # print("lurus-sprint")
                            else:
                                agent._local_planner._vehicle_controller.max_throt = 0.615
                                agent._local_planner._vehicle_controller.max_steer_step = 0.1
                                # print("lurus-sprint smooth")
                        else:
                            if(speed_kmh < 13):
                                agent._local_planner._vehicle_controller.max_throt = 0.6
                                agent._local_planner._vehicle_controller.max_steer_step = 0.1
                                # print("lurus-slow")
                            else:
                                agent._local_planner._vehicle_controller.max_throt = 0.3
                                agent._local_planner._vehicle_controller.max_steer_step = 0.1
                                # print("lurus-drop")

                    else:
                        if(speed_kmh < 13):
                            if (is_curve == +1) or (next_road_option.name == "RIGHT"):
                                agent._local_planner._vehicle_controller.max_throt = 0.5
                                agent._local_planner._vehicle_controller.max_steer_step = 0.01
                                # print("belok kanan-slow")
                            elif (is_curve == -1) or (next_road_option.name == "LEFT"):
                                agent._local_planner._vehicle_controller.max_throt = 0.6
                                agent._local_planner._vehicle_controller.max_steer_step = 0.009
                                # print("belok kiri-slow")
                        else:
                            agent._local_planner._vehicle_controller.max_throt = 0.3
                            agent._local_planner._vehicle_controller.max_steer_step = 0.1
                            # print("belok-drop")

                    # noise_steer = 0
                    steer_resultant = 0
                    if  (
                        (is_curve == 0)
                        and (not is_junction)
                        and (
                            (next_road_option.name == "LANEFOLLOW")
                            or (next_road_option.name == "STRAIGHT"))
                        ):
                        if control.brake == 0:
                            steer_resultant = control_steer_signal + noise_steer
                            world.current_steer_resultant = steer_resultant
                            control.steer = steer_resultant
                    else:
                        noise_steer = 0
                        steer_resultant = control_steer_signal + noise_steer
                        # steer_resultant = 0
                        world.current_steer_resultant = steer_resultant
                        control.steer = steer_resultant
                    
                    # if (hero_stop
                    #     and (steer_resultant < 0.05)
                    #     and ((next_road_option.name == "LANEFOLLOW")
                    #         or (next_road_option.name == "STRAIGHT")
                    #         or (is_curve == 0))
                    #     ):
                    #     control.brake = 0.5
                    #     print("hero stop")
                    # else:
                    #     print("hero go")

                    if control.brake > 0.1:
                        control.throttle = 0
                    else:
                        control.brake = 0

                    # Update tiap frame
                    steer_graph_display.update(noise_val=noise_steer,
                                            control_val=control_steer_signal ,
                                            resultant_val=steer_resultant,
                                            throttle_val=control.throttle,
                                            brake_val=control.brake,
                                            speed_val=speed_kmh)
                    # print(f"Speed={speed_kmh:.2f}, Target={lp._target_speed:.2f}, Brake={control.brake}, Throttle={control.throttle}")

                    # control.throttle = 0.55
                    world.player.apply_control(control)
                else:
                    print("control cadangan")
                    world.player.apply_control(control_cadangan)
                
                if frame_count > 0:
                    start_logging = 1
                    WaypointDisp.update()
                    waypoints_local = WaypointDisp.get_last_local_waypoints()
                    flat_waypoints = [coord for wp in waypoints_local for coord in wp]
                    # Jika kurang dari 5 waypoints, pad dengan 0
                    while len(flat_waypoints) < 10:
                        flat_waypoints.extend([0.0, 0.0])

                    print(f"vif:{vehicle_in_front} curve:{1 if is_curve != 0 else 0} junction:{1 if is_junction else 0} red_light:{1 if stat_red else 0}")
                    success = save_frame_data_buffered(
                        frame_count, rgb_views, depth_views, lidar_above, lidar_ground,
                        world, agent, control_steer_signal, noise_steer, steer_resultant,
                        (1.0 if stat_red else 0.0), (1.0 if stat_at_traffic_light else 0.0), plan, flat_waypoints, weather_name,
                        (1.0 if len(vehicles_in_front) > 0 else 0.0), (1.0 if (is_curve != 0) else 0.0), (1.0 if(is_junction) else 0.0),
                        speed_kmh_t, speed_kmh_t_1, speed_kmh_t_2, speed_kmh_t_3
                    )

                    if success:
                        frame_saved += 1
                    else:
                        frame_skipped += 1
                else:
                    frame_skipped += 1

                hud.debug_status = (
                    f"weather:{weather_name} frame:{frame_count} saved:{frame_saved} skipped:{frame_skipped} "
                    f"len(plan):{len(plan_list)} TL:{stat_at_traffic_light} "
                    f"Red:{stat_red} Roadopt:{next_road_option.name} VIF:{vehicle_in_front} "
                    f"Respawn:{respawn_counter} logging:{start_logging}"
                )

                frame_count += 1
            else:
                print("Collision occurs. Exit now")
                loop_lock = False

            if is_logging_complete():
                print("✅ Semua command sudah mencapai jumlah batch target. Logging selesai.")
                break
            
            if not loop_lock:
                print(f"🛑 Dibatasi sampai 1 episode. Logging dihentikan.")
                break

            if frame_count % 500 == 0:
                actor_count = len(world.world.get_actors())
                vehicle_count = len([a for a in world.world.get_actors().filter('vehicle.*')])
                print(f"[INFO] Frame: {frame_count} | Vehicles: {vehicle_count} | Actors: {actor_count} | Server FPS: {hud.server_fps:.1f} | Client FPS: {clock.get_fps():.1f}")
                
                # Logging performa setiap 1000 frame
                ram_gb = process.memory_info().rss / (1024**3)
                swap_used = psutil.swap_memory().used / (1024**3)
                print(f"[PERF] Frame: {frame_count} | Server FPS: {hud.server_fps:.1f} | Client FPS: {clock.get_fps():.1f} | RAM: {ram_gb:.2f} GB | Swap: {swap_used:.2f} GB")

                t_gc = time.time()
                collected = gc.collect()
                print(f"[GC] Collected {collected} objects in {time.time() - t_gc:.2f} sec")

                mem = psutil.virtual_memory()
                print(f"MemFree: {mem.free / (1024**3):.2f} GB, MemAvailable: {mem.available / (1024**3):.2f} GB")  

    finally:
        if display_manager:
            display_manager.destroy()

        if world is not None:
            # Kembalikan setting simulator
            settings = world.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            world.world.apply_settings(settings)
            traffic_manager.set_synchronous_mode(True)

            # Hentikan dan hancurkan sensor sinkron (gunakan stop+destroy dalam try)
            for sensor in [rgb_left_sensor, rgb_center_sensor, rgb_right_sensor, depth_left_sensor, depth_center_sensor, depth_right_sensor, lidar_sensor]:
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
        # default='1200x528',
        default='1800x792',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--rescam',
        metavar='WIDTHxHEIGHT',
        # default='330x120',
        default='200x150',
        help='Cam resolution (default: 200x120)')
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
        '--weather',
        type=str,
        choices=['ClearNoon', 'ClearSunset', 'WetNoon', 'HardRainNoon'],
        default=None,
        help='Weather preset to use (e.g., ClearNoon, WetNoon). Default: rotate based on batch.')
    argparser.add_argument(
        '--lead_vehicle',
        action='store_true',
        help='Run in lead vehicle mode')    
    argparser.add_argument(
        '--add_traffic',
        action='store_true',
        help='Run with traffic vehicle')
    args = argparser.parse_args()

    global h5_dir
    h5_dir = os.path.join("dataset")
    os.makedirs(h5_dir, exist_ok=True)

    global global_batch_idx
    global_batch_idx = get_latest_h5_index(h5_dir)

    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.widthcam, args.heightcam = [int(x) for x in args.rescam.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

        # Deteksi alasan keluar: logging complete atau frame limit
        if is_logging_complete():
            print("[EXIT] ✅ Semua data sudah terkumpul. Exit code = 0.")
            sys.exit(0)  # logging selesai, kode keluar 0
        else:
            print("[EXIT] 🛑 Logging berhenti karena dibatasi. Exit code = 2.")
            sys.exit(2)  # frame habis, lanjutkan sesi berikutnya
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        sys.exit(1)  # keyboard interrupt = 1

if __name__ == '__main__':
    main()
