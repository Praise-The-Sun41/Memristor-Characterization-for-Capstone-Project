# Memristor Measurement and Simulation

This repository contains Python scripts for memristor device measurement,
data plotting, and offline convolution simulation.

## Project Structure

```text
.
├── memristor_measure/       # Keithley 2450 measurement and plotting scripts
├── memristor_simulation/    # Offline memristor convolution simulation
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## Main Scripts

- `memristor_measure/forming.py`: progressive forming workflow for memristor devices.
- `memristor_measure/IV_Sweep.py`: I-V sweep measurement with configurable compliance current.
- `memristor_measure/Fixed_V_Read.py`: fixed-voltage current monitoring.
- `memristor_measure/modulation.py`: pulse-based LTP/LTD modulation experiment.
- `memristor_measure/resistance_switch_retention.py`: SET/RESET resistance switching and retention test.
- `memristor_measure/pulse_resistance_modulation.py`: pulse-by-pulse resistance modulation test.
- `memristor_measure/bulkplot.py`: bulk plotting utility for I-V CSV results.
- `memristor_simulation/iv_convolution_sim.py`: offline convolution simulation using measured I-V data.

## Environment

Recommended Python version:

```bash
python --version
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Measurement scripts require a VISA-compatible instrument setup, such as a
Keithley 2450 SMU, plus a working PyVISA backend.

## Usage

Run measurement scripts from their folder or from the repository root:

```bash
python memristor_measure/IV_Sweep.py
python memristor_measure/forming.py
python memristor_measure/Fixed_V_Read.py
python memristor_measure/modulation.py
python memristor_measure/resistance_switch_retention.py --dev DEV-1-1
python memristor_measure/pulse_resistance_modulation.py --dev DEV-1-1
```

Run the offline simulation:

```bash
python memristor_simulation/iv_convolution_sim.py
```

List simulation device data:

```bash
python memristor_simulation/iv_convolution_sim.py --list-devices
```

## Data

Generated measurement and simulation outputs are ignored by git by default to
keep the repository lightweight. If specific CSV, PNG, or PDF result files need
to be published, update `.gitignore` and add only the selected files.

## TODO

- Add project background and research objective.
- Add hardware setup details.
- Add device stack and fabrication notes.
- Add example figures or selected result files.
- Add citation or license information.
