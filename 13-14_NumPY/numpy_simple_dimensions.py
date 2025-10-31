import numpy as np


heartbeat = np.array([72, 75, 78, 74, 71, 73, 76, 79])
print(f"Forma: {heartbeat.shape}")  # (8,)
print(f"Puls mediu: {np.mean(heartbeat)} BPM")


room_temp = np.array([
    [22.5, 23.1],
    [22.7, 24.2],
    [22.4, 23.8]
])
print(f"Forma: {room_temp.shape}")  # (3, 2)
print(f"Temperatura medie: {np.mean(room_temp)}°C")