# RTL Folder

This folder contains starter RTL for later local Verilator testing.

Current file:

- `mac_unit.sv`: signed 8-bit multiply-accumulate block with a tiny valid/ready style control path.

This is not connected to the Python compiler yet. It is a placeholder for the next milestone:

```text
Python scheduled op -> tiled primitive op -> RTL MAC/dot-product block
```

## Later local tools

Install in WSL2 Ubuntu:

```bash
sudo apt update
sudo apt install -y verilator gtkwave make g++
```

Then add a C++ or cocotb testbench.
