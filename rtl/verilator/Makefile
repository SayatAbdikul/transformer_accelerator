# Verilator build & test Makefile for TACCEL Phase 1
#
# Prerequisites:
#   verilator >= 5.0   (brew install verilator  OR  apt install verilator)
#   g++ / clang++ with C++17
#
# Usage:
#   make           — build + run all Phase 1 tests
#   make test_decode   — decode unit tests only
#   make test_control  — control unit / full-chip tests only
#   make clean     — remove generated objects

VERILATOR     ?= verilator
CXX           ?= g++
CXXFLAGS      := -std=c++17 -O2 -Wall -Wno-unused-variable
VFLAGS        := --cc --exe --build --timing \
                 -Wall -Wno-UNUSED -Wno-UNDRIVEN \
                                  --x-assign fast --x-initial fast

SRC_DIR       := ../src
INCLUDE_DIR   := $(SRC_DIR)/include
TB_DIR        := $(SRC_DIR)/tb
BUILD_DIR     := build

# Source file lists
PKG           := $(INCLUDE_DIR)/taccel_pkg.sv
DECODE_SV     := $(PKG) $(SRC_DIR)/decode_unit.sv $(TB_DIR)/tb_decode_unit.sv
CONTROL_SV    := $(PKG) \
                 $(SRC_DIR)/decode_unit.sv \
                 $(SRC_DIR)/fetch_unit.sv \
                 $(SRC_DIR)/control_unit.sv \
                 $(SRC_DIR)/memory/register_file.sv \
                 $(SRC_DIR)/memory/sram_dp.sv \
                 $(SRC_DIR)/memory/sram_subsystem.sv \
                 $(SRC_DIR)/taccel_top.sv

.PHONY: all test_decode test_control clean

all: test_decode test_control

# -----------------------------------------------------------------------
# test_decode: verilate tb_decode_unit + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_decode/Vtb_decode_unit: $(DECODE_SV) test_decode.cpp
	mkdir -p $(BUILD_DIR)/test_decode
	$(VERILATOR) $(VFLAGS) \
	  -I$(INCLUDE_DIR) \
	  --top-module tb_decode_unit \
	  --Mdir $(BUILD_DIR)/test_decode \
	  $(DECODE_SV) \
	  test_decode.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_decode -f Vtb_decode_unit.mk

test_decode: $(BUILD_DIR)/test_decode/Vtb_decode_unit
	@echo "--- Running decode unit tests ---"
	$(BUILD_DIR)/test_decode/Vtb_decode_unit

# -----------------------------------------------------------------------
# test_control: verilate taccel_top + compile + run
# -----------------------------------------------------------------------
$(BUILD_DIR)/test_control/Vtaccel_top: $(CONTROL_SV) test_control.cpp
	mkdir -p $(BUILD_DIR)/test_control
	$(VERILATOR) $(VFLAGS) \
	  -I$(INCLUDE_DIR) \
	  --top-module taccel_top \
	  --Mdir $(BUILD_DIR)/test_control \
	  $(CONTROL_SV) \
	  test_control.cpp \
	  -CFLAGS "-I$(CURDIR)/include $(CXXFLAGS)"
	$(MAKE) -C $(BUILD_DIR)/test_control -f Vtaccel_top.mk

test_control: $(BUILD_DIR)/test_control/Vtaccel_top
	@echo "--- Running control unit tests ---"
	$(BUILD_DIR)/test_control/Vtaccel_top

# -----------------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
