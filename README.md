# HMC-Research
HMC-Research Project

Current environment: HMC-Research

requirements.txt generated Feb 16, 2026 from CHMC_FALL_2025. 


To Do High Level:
0) Currently refactoring code into modular parts with namedtuples for better debugging

1) Smaller To Do list found in ```CHMC_notes.txt```.
2) List of potential issues in ```CHMC_debts.txt```.


seven main files: 
- ```datatypes.py``` containts types for easier debugging
- ```core.py``` contains core hamiltonian structures
- ```target.py``` generates $\pi(q)$ pdfs
- ```integrator.py``` contains numerical integrators for hamiltonians
- ```sampler.py``` contains sampler logic
- ```metrics.py``` contains analytics of chains
- ```tests.py``` contains coding tests for numerical integrators, hamiltonians, pdfs, and samplers,
- ```plotting.py``` contains plotting functionality for metrics