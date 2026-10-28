# Project 5.4 – Wafer Spin-Coater Acceleration Control

## 1. Installation
System requirements: Python 3.10 or higher. Open the terminal in the root directory and run:
`pip install -r requirements.txt`

## 2. Usage
*   **Run all cases:** `python main.py` (The program will automatically compute 26 cases and update the `summary.csv` file).
*   **Run and print results for a single case (For testing/Q&A):** `python main.py --case data/inputs/case_300mm_scurve_1000.json`
*   **Generate plots:** `python plots.py` (Extracts data from the outputs folder to generate plots in the visualizations folder).

## 3. Directory Structure
*   `main.py`: Contains the main execution flow (CLI), manages inputs, and exports CSV data.
*   `physics_model.py`: Contains kinematic and dynamic physics differential equations.
*   `solvers.py`: Integrates differential equation solving algorithms (RK45).
*   `profile_segments.py`: Defines acceleration profile functions (Linear, S-curve, Exponential).
*   `error_checker.py`: Module for checking integration errors and energy conservation.
*   `plots.py`: Handles data visualization and plotting.
*   `data/`: Contains configuration files `inputs/` (JSON) and output data sequences `outputs/` (CSV).
*   `tests/`: Contains configuration files with erroneous parameters used for system exception testing.

## 4. Output Data Columns (CSV)
The `summary.csv` file aggregates the maximum parameters for each case: acceleration time (`t_ramp_s`), acceleration revolutions (`revolutions_ramp`), maximum jerk (`jerk_max`), maximum tangential acceleration (`a_t_max`), maximum centripetal acceleration (`a_c_max`), maximum torque (`torque_max`), and energy conservation error (`energy_err_J`).

## 5. Adding New Cases
To add a new case, create a JSON file in `data/inputs/` containing the physical parameters of the chuck, wafer, and the `steps` array. Physical parameters must be positive numbers. The steps must ensure velocity continuity (the next step starts with the target velocity of the previous step), and the `duration` must match the corresponding kinematic expressions.
