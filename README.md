# QPSK Simulation using Python

## Aim
To simulate Quadrature Phase Shift Keying (QPSK) modulation and demodulation.

## Tools Required
- Python
- NumPy
- Matplotlib

## Steps
1. Generate binary data.
2. Map bits to I and Q components.
3. Modulate and combine signals.
4. Plot In-phase, Quadrature, and Combined waveforms.

## code
```
# qpsk_simulation.py

import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_symbols = 8
T = 0.5
fs = 200.0
t = np.arange(0, T, 1/fs)

# Generate random bits
bits = np.random.randint(0, 2, num_symbols * 2)

# I and Q bits
i_bits = bits[0::2]
q_bits = bits[1::2]

# Map bits to -1 and +1
i_values = 2 * i_bits - 1
q_values = 2 * q_bits - 1

i_signal, q_signal, combined_signal = np.array([]), np.array([]), np.array([])
symbol_times = []

for i in range(num_symbols):
    i_carrier = i_values[i] * np.cos(2 * np.pi * 2 / T * t)
    q_carrier = q_values[i] * np.sin(2 * np.pi * 2 / T * t)

    symbol_times.append(i * T)
    i_signal = np.concatenate((i_signal, i_carrier))
    q_signal = np.concatenate((q_signal, q_carrier))
    combined_signal = np.concatenate((combined_signal, i_carrier + q_carrier))

t_total = np.arange(0, num_symbols * T, 1/fs)

# Plot
plt.figure(figsize=(14, 9))

plt.subplot(3, 1, 1)
plt.plot(t_total, i_signal, label='In-phase (cos)', color='blue')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, np.max(i_signal) * 0.8, f'{i_bits[i]}', fontsize=12)
plt.title('In-phase Component')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t_total, q_signal, label='Quadrature (sin)', color='orange')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, np.max(q_signal) * 0.8, f'{q_bits[i]}', fontsize=12)
plt.title('Quadrature Component')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_total, combined_signal, label='QPSK Signal = I + Q', color='green')
for i, symbol_time in enumerate(symbol_times):
    plt.axvline(symbol_time, color='red', linestyle='--', linewidth=0.5)
    plt.text(symbol_time + T/4, np.max(combined_signal) * 0.8, f'{i_bits[i]}{q_bits[i]}', fontsize=12)
plt.title('Combined QPSK Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

```
## output
![image](https://github.com/user-attachments/assets/7cf16155-263c-4e38-8b11-57f52d467478)

## Result
The QPSK modulation and demodulation simulation works correctly under ideal (noiseless) conditions.
