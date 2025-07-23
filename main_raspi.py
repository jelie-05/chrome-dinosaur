import pygame
import os
import random
# import onnxruntime as ort
import tflite_runtime.interpreter as tflite

from queue import Queue, Empty

import pickle
import socket
import time
import numpy as np
import threading
import struct


from ifxAvian import Avian
from src.utils.doppler_avian import DopplerAlgo
from src.utils.common_raspi import do_inference_processing_np
from src.utils.debouncer_time import DebouncerTime

pygame.init()

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("pydino/Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("pydino/Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("pydino/Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("pydino/Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("pydino/Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("pydino/Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join("pydino/Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("pydino/Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("pydino/Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join("pydino/Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("pydino/Assets/Cactus", "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join("pydino/Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("pydino/Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("pydino/Assets/Other", "Cloud.png"))
BG = pygame.image.load(os.path.join("pydino/Assets/Other", "Track.png"))


class Dinosaur:
    X_POS = 80
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5
    DUCK_TIME = 8.0

    def __init__(self):
        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS

        self.duck_time = self.DUCK_TIME

    def update(self, userInput):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if userInput == 'pull' and not (self.dino_jump or self.dino_duck):
            self.dino_jump = True
            self.dino_duck = False
            self.dino_run = False
        elif userInput == 'push' and not (self.dino_jump or self.dino_duck):
            self.dino_duck = True
            self.dino_jump = False
            self.dino_run = False
        elif not self.dino_jump and not self.dino_duck:
            self.dino_run = True

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        if self.dino_duck:
            self.dino_rect.x = self.X_POS
            self.dino_rect.y = self.Y_POS_DUCK
            self.duck_time -= 0.3
        if self.duck_time < -self.DUCK_TIME:
            self.dino_duck = False
            self.duck_time = self.DUCK_TIME
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.X_POS
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 1.5
            self.jump_vel -= 0.3
        if self.jump_vel < - self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type):
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 330


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.type = random.randint(0, 2)
        super().__init__(image, self.type)
        self.rect.y = 315


class Bird(Obstacle):
    def __init__(self, image):
        self.type = 0
        super().__init__(image, self.type)
        self.rect.y = 250
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1


def softmax_np(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class PyGameInference:
    def __init__(self, observation_length, num_classes):
        self.num_classes = num_classes
        self.observation_length = observation_length

        self.debouncer = DebouncerTime(memory_length=self.observation_length)
        # self.interpreter = tflite.Interpreter(model_path="runs/trained_models/train_0613.tflite")
        # self.interpreter.allocate_tensors()
        # self.input_details = self.interpreter.get_input_details()
        # self.output_details = self.interpreter.get_output_details()

        with open('runs/trained_models/train_0606-idx_mapping.pkl', 'rb') as f:
            self.idx_to_class = pickle.load(f)

        self.label = 'nothing'
        self.label_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.last_print_time = -1.0
        self.cooldown_time = 0.0
        self.label_queue = Queue(maxsize=1)

        self.label_buffer = []
        self.buffer_size = 1

    def get_class_name(self, label):
        return self.idx_to_class[label]
    
    def _recv_all(self, sock, size):
        data = b''
        while len(data) < size:
            more = sock.recv(size - len(data))
            if not more:
                raise EOFError("Connection closed")
            data += more
        return data

    def run(self):
        # HOST = '192.168.1.2' # Raspi: HOST = '192.168.1.1'
        HOST = '127.0.0.1'
        PORT = 5005
        SERVER_HOST = '0.0.0.0'
        SERVER_PORT = 5006
        num_beams = 27         # number of beams
        max_angle_degrees = 40 # maximum angle, angle ranges from -40 to +40 degrees

        config = Avian.DeviceConfig(
            sample_rate_Hz = 2500000,       # 1MHZ
            rx_mask = 7,                      # activate RX1 and RX3
            tx_mask = 1,                      # activate TX1
            if_gain_dB = 25,                  # gain of 33dB
            tx_power_level = 31,              # TX power level of 31
            start_frequency_Hz = 57569828864,        # 60GHz 
            end_frequency_Hz = 63930171392,        # 61.5GHz
            num_chirps_per_frame = 256,       # 128 chirps per frame
            num_samples_per_chirp = 128,       # 64 samples per chirp
            chirp_repetition_time_s = 0.0004112379392609, # 0.5ms
            frame_repetition_time_s = 0.10526315867900848,   # 0.15s, frame_Rate = 6.667Hz
            mimo_mode = 'off'                 # MIMO disabled
        )

        with Avian.Device() as device, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock, socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            # Sender
            sock.connect((HOST, PORT))

            # Receiver
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((SERVER_HOST, SERVER_PORT))
            server.listen(1)

            num_rx_antennas = 3 #device.get_sensor_information()["num_rx_antennas"]
            device.set_config(config)

            algo = DopplerAlgo(device.get_config(), num_rx_antennas)

            while not self.stop_event.is_set():
                frame_data = device.get_next_frame()
                data_all_antennas = []
                for i_ant in range(num_rx_antennas):
                    mat = frame_data[i_ant, :, :]
                    dfft_dbfs = algo.compute_doppler_map(mat, i_ant)
                    data_all_antennas.append(dfft_dbfs)

                range_doppler = do_inference_processing_np(data_all_antennas)
                self.debouncer.add_scan_np(range_doppler)

                dtm, rtm = self.debouncer.get_scans()

                rtm_np = np.stack(rtm, axis=1)
                dtm_np = np.stack(dtm, axis=1)
                rtm_np = np.squeeze(rtm_np, axis=2)
                dtm_np = np.squeeze(dtm_np, axis=2)
                rdtm_np = np.stack([rtm_np, dtm_np], axis=1)
                rdtm_np = np.expand_dims(rdtm_np, axis=0)
                rdtm_np = rdtm_np.transpose(0, 2, 1, 3)

                if rdtm_np.shape[3] >= self.observation_length:
                    # Send data to TCP server
                    print("Sending data to TCP server...")
                    rdtm_np = rdtm_np.astype(np.float32)
                    sent_data = rdtm_np.tobytes()
                    # sent_data = pickle.dumps(rdtm_np)
                    frame_size = len(sent_data)
                    sock.sendall(frame_size.to_bytes(4, 'big'))
                    sock.sendall(sent_data)

                    # Receive data from TCP server
                    print("Receiving data from TCP server...")
                    conn, addr = server.accept()

                    with conn:
                        try:
                            len_data = self._recv_all(conn, 4)  # Read the first 4 bytes for length

                            # Unpack as a network-ordered (big-endian) unsigned integer
                            payload_size = struct.unpack('!I', len_data)[0]  # Unpack the length

                            payload = self._recv_all(conn, payload_size)
                            num_floats = payload_size // 4  # Each float is 4 bytes
                            floats = struct.unpack(f'<{num_floats}f', payload)
                        except EOFError as e:
                            print(f"Client disconnected: {e}")

                    # Convert floats to numpy array
                    floats = np.array(floats, dtype=np.float32) 
                    output = floats.reshape(1, -1)  # shape: (1, 4)
                    
                    if output.ndim == 1:
                        output = output[None, :]

                    exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
                    softmax_output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
                    prediction = softmax_output.squeeze(0)
                    max_idx = np.argmax(softmax_output, axis=1).item()

                    if time.time() - self.last_print_time > self.cooldown_time:
                        detected_label = self.get_class_name(max_idx)
                        confidence = prediction[max_idx]

                        with self.label_lock:
                            threshold_pull = 0.8
                            threshold_push = 0.8

                            if detected_label == 'pull' and confidence >= threshold_pull:
                                self.label_buffer = ['pull'] * self.buffer_size
                            elif detected_label == 'push' and confidence >= threshold_push:
                                self.label_buffer = ['push'] * self.buffer_size
                            elif self.label_buffer:
                                self.label_buffer.pop(0)
                            if not self.label_buffer:
                                self.label_buffer = ['nothing']

                            self.label = self.label_buffer[0]
                            while not self.label_queue.empty():
                                try:
                                    self.label_queue.get_nowait()
                                except Empty:
                                    break
                            self.label_queue.put_nowait(self.label)

                        print(f"Detected: {detected_label} | Prob: {confidence * 100:.2f}% → Used: {self.label}")
                        self.last_print_time = time.time()

    def main(self):
        global game_speed, x_pos_bg, y_pos_bg, points, obstacles
        run = True
        clock = pygame.time.Clock()
        player = Dinosaur()
        cloud = Cloud()
        game_speed = 5
        x_pos_bg = 0
        y_pos_bg = 380
        points = 0
        font = pygame.font.Font('freesansbold.ttf', 20)
        obstacles = []
        death_count = 0
        current_label = 'nothing'

        def score():
            global points, game_speed
            points += 1
            text = font.render("Points: " + str(points), True, (0, 0, 0))
            textRect = text.get_rect()
            textRect.center = (1000, 40)
            SCREEN.blit(text, textRect)

        def background():
            global x_pos_bg, y_pos_bg
            image_width = BG.get_width()
            SCREEN.blit(BG, (x_pos_bg, y_pos_bg))
            SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
            if x_pos_bg <= -image_width:
                SCREEN.blit(BG, (image_width + x_pos_bg, y_pos_bg))
                x_pos_bg = 0
            x_pos_bg -= game_speed

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            SCREEN.fill((255, 255, 255))

            try:
                current_label = self.label_queue.get_nowait()
            except Empty:
                pass

            player.draw(SCREEN)
            player.update(current_label)

            if len(obstacles) == 0:
                choice = random.randint(0, 2)
                if choice == 0:
                    obstacles.append(SmallCactus(SMALL_CACTUS))
                elif choice == 1:
                    obstacles.append(LargeCactus(LARGE_CACTUS))
                else:
                    obstacles.append(Bird(BIRD))

            for obstacle in obstacles:
                obstacle.draw(SCREEN)
                obstacle.update()
                if player.dino_rect.colliderect(obstacle.rect):
                    pygame.time.delay(2000)
                    death_count += 1
                    menu(death_count)

            background()
            cloud.draw(SCREEN)
            cloud.update()
            score()

            clock.tick(30)
            pygame.display.update()


inference = None
inference_thread = None

def menu(death_count):
    global inference, inference_thread

    observation_length = 10
    num_classes = 4

    if inference is None:
        inference = PyGameInference(observation_length=observation_length, num_classes=num_classes)
        inference_thread = threading.Thread(target=inference.run, daemon=True)
        inference_thread.start()

    global points
    run = True
    while run:
        SCREEN.fill((255, 255, 255))
        font = pygame.font.Font('freesansbold.ttf', 30)

        if death_count == 0:
            text = font.render("Press any Key to Start", True, (0, 0, 0))
        else:
            text = font.render("Press any Key to Restart", True, (0, 0, 0))
            score_text = font.render("Your Score: " + str(points), True, (0, 0, 0))
            scoreRect = score_text.get_rect()
            scoreRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50)
            SCREEN.blit(score_text, scoreRect)

        textRect = text.get_rect()
        textRect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        SCREEN.blit(text, textRect)
        SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 - 140))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                run = False
            if event.type == pygame.KEYDOWN:
                inference.main()


if __name__ == "__main__":
    menu(death_count=0)