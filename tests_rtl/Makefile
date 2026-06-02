# OpenTSP RTL verification: signed INT8 MAC unit
# Run from repo root with: make -C tests_rtl

SIM ?= verilator
TOPLEVEL_LANG ?= verilog

# tests_rtl is one level below the repo root.
REPO_ROOT := $(abspath $(CURDIR)/..)
VERILOG_SOURCES := $(REPO_ROOT)/rtl/mac_unit.sv

TOPLEVEL := mac_unit
MODULE := test_mac_unit

# Keep generated simulator files inside tests_rtl/sim_build.
SIM_BUILD ?= sim_build

# Verilator accepts the SystemVerilog syntax used in rtl/mac_unit.sv.
EXTRA_ARGS += --trace

include $(shell cocotb-config --makefiles)/Makefile.sim
