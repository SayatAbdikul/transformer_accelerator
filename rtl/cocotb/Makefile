# cocotb test Makefile for TACCEL Phase 1
#
# Prerequisites:
#   pip install cocotb cocotb-bus
#   iverilog >= 11  OR  verilator >= 5.0  OR  cadence/synopsys VCS
#
# Usage:
#   make                  — run all Phase 1 cocotb tests (Icarus Verilog)
#   make SIM=verilator    — run with Verilator
#   make clean

TOPLEVEL_LANG ?= verilog
SIM           ?= icarus     # icarus | verilator | vcs | xsim

TOPLEVEL      := taccel_top
MODULE        := test_fetch_decode,test_dma,test_systolic

SRC_DIR       := ../src
INCLUDE_DIR   := $(SRC_DIR)/include

# All SV source files for Phase 2 (order matters for Icarus)
VERILOG_SOURCES := \
  $(INCLUDE_DIR)/taccel_pkg.sv \
  $(SRC_DIR)/decode_unit.sv \
  $(SRC_DIR)/fetch_unit.sv \
  $(SRC_DIR)/control_unit.sv \
  $(SRC_DIR)/memory/sram_dp.sv \
  $(SRC_DIR)/memory/register_file.sv \
  $(SRC_DIR)/memory/sram_subsystem.sv \
  $(SRC_DIR)/systolic/systolic_pe.sv \
  $(SRC_DIR)/systolic/systolic_array.sv \
  $(SRC_DIR)/systolic/systolic_controller.sv \
  $(SRC_DIR)/dma_engine.sv \
  $(SRC_DIR)/taccel_top.sv

# Include paths
COMPILE_ARGS += -I$(INCLUDE_DIR)
COMPILE_ARGS += -I$(SRC_DIR)/systolic

# Verilator-specific flags
ifeq ($(SIM), verilator)
  COMPILE_ARGS += --timing -Wall -Wno-UNUSED -Wno-UNDRIVEN
endif

# Icarus-specific: use SystemVerilog mode
ifeq ($(SIM), icarus)
  COMPILE_ARGS += -g2012
endif

# cocotb makefile magic
include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: clean
clean::
	rm -rf sim_build __pycache__ results.xml *.vcd dump.* \
	       $(TOPLEVEL).vvp iverilog_dump.v
