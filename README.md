# Automated PSCAD Waveform & Z-Tool Analysis GUI

Python GUI for automated PSCAD waveform analysis and Z-tool based stability analysis of a 2-Level VSC system.

---

## Overview

This project provides a unified Python-based interface for automating PSCAD simulation analysis and frequency-domain stability analysis.

The GUI integrates PSCAD through the MHI Python interface and provides two independent workflows:

- Automated Waveform Analyzer
- Z-Tool Analysis for 2-Level VSC

---

## Features

### Waveform Analyzer

- Automated PSCAD simulation execution
- Automatic `.out` and `.inf` file detection
- Automatic extraction of available simulation channels
- Selective output/channel selection for visualization
- Automated waveform plotting
- Reduced manual post-processing

### Z-Tool Analysis

- 2-Level VSC frequency-domain analysis
- Automated AC Scan execution
- D-axis and Q-axis perturbation injection
- Automated frequency sweep
- VSC and grid admittance extraction
- Nyquist stability analysis
- Encirclement calculation
- Dominant mode and frequency identification
- Integrated simulation-status terminal

---

## System Flow

```text
                    Python GUI
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
   Waveform Analyzer            Z-Tool Analysis
          │                           │
          ▼                           ▼
       PSCAD                    2-Level VSC
     Simulation                    AC Scan
          │                           │
          ▼                           ▼
     .OUT / .INF               D-Axis Injection
          │                           │
          ▼                      Q-Axis Injection
    Channel Selection                │
          │                           ▼
          ▼                    Frequency Sweep
   Waveform Plots                     │
                                      ▼
                              Admittance Extraction
                                      │
                                      ▼
                              Nyquist Stability
                                      │
                                      ▼
                                 Final Results
```
## Technologies Used

- **Python 3.11** – Core development and automation
- **Tkinter** – GUI development
- **NumPy** – Numerical computation and matrix operations
- **Matplotlib** – Waveform and Nyquist plot visualization
- **PSCAD** – Electromagnetic transient simulation
- **MHI Python Interface** – Python–PSCAD automation and control
- **Z-Tool** – Frequency-domain impedance/admittance and stability analysis


## Applications

- **Automated PSCAD waveform analysis** and visualization
- **Time-domain simulation post-processing** for power-electronic systems
- **2-Level VSC frequency-domain analysis** using Z-Tool
- **VSC-grid interaction and admittance analysis**
- **Nyquist-based stability assessment** of converter-dominated systems
- **Automated simulation monitoring and result analysis**
