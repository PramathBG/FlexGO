VITIS_HLS_DIR := /tools/software/xilinx/Vitis_HLS/2023.1
BUILD_DIR := ../src

IFLAG += -I "${BUILD_DIR}"

IFLAG += -I "${VITIS_HLS_DIR}/include"
IFLAG += -I "${VITIS_HLS_DIR}/include/etc"
IFLAG += -I "${VITIS_HLS_DIR}/include/utils"

IFLAG += -I "${VITIS_HLS_DIR}/lnx64/tools/fpo_v7_1"
IFLAG += -I "${VITIS_HLS_DIR}/lnx64/tools/fft_v9_1"
IFLAG += -I "${VITIS_HLS_DIR}/lnx64/tools/fir_v7_0"
IFLAG += -I "${VITIS_HLS_DIR}/lnx64/tools/dds_v6_0"
IFLAG += -I "${VITIS_HLS_DIR}/lnx64/lib/csim"

CFLAG += -L"${VITIS_HLS_DIR}/lnx64/lib/csim" -lhlsmc++-CLANG39 -Wl,-rpath,"${VITIS_HLS_DIR}/lnx64/lib/csim" -Wl,-rpath,"${VITIS_HLS_DIR}/lnx64/tools/fpo_v7_1"
                                            
# for C sim
IFLAG += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__

IFLAG += -g
CFLAG += -fPIC -O3 -Wno-unused-result
CFLAG += -lm
CFLAG += -std=c++11
# TODO: add support for faster runtime using -ffast-math and -march=native
# CFLAG += -ffast-math -march=native

CXX = ${VITIS_HLS_DIR}/lnx64/tools/clang-3.9/bin/clang++
# TODO: add support for system compilers
#CXX = g++


result: ${BUILD_DIR}/conv_layer.cc ${BUILD_DIR}/finalize.cc ${BUILD_DIR}/GNN_compute.cc ${BUILD_DIR}/globals.cc ${BUILD_DIR}/linear.cc ${BUILD_DIR}/load_inputs.cc ${BUILD_DIR}/node_embedding.cc  ${BUILD_DIR}/message_passing.cc ${BUILD_DIR}/main.cc ${BUILD_DIR}/load.cc ${BUILD_DIR}/dcl.h ${BUILD_DIR}/conv_layer.h ${BUILD_DIR}/finalize.h ${BUILD_DIR}/linear.h ${BUILD_DIR}/load_inputs.h ${BUILD_DIR}/node_embedding.h ${BUILD_DIR}/message_passing.h ${BUILD_DIR}/util.h ${BUILD_DIR}/testbench.h
	$(CXX) ${BUILD_DIR}/conv_layer.cc ${BUILD_DIR}/finalize.cc ${BUILD_DIR}/GNN_compute.cc ${BUILD_DIR}/globals.cc ${BUILD_DIR}/linear.cc ${BUILD_DIR}/load_inputs.cc ${BUILD_DIR}/node_embedding.cc  ${BUILD_DIR}/message_passing.cc ${BUILD_DIR}/main.cc ${BUILD_DIR}/load.cc -o result $(CFLAG) $(IFLAG)

all: ./result

run: ./result
	./result

clean:
	rm -f *.o result