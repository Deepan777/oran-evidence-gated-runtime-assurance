#!/usr/bin/env bash
set -euo pipefail

cp /patch/flexible_o_du_impl.cpp \
  /src/apps/units/flexible_o_du/split_helpers/flexible_o_du_impl.cpp
cp /patch/radio_zmq_tx_channel.cpp \
  /src/lib/radio/zmq/radio_zmq_tx_channel.cpp
cp /patch/lower_phy_baseband_processor.cpp \
  /src/lib/phy/lower/lower_phy_baseband_processor.cpp
cp /patch/lower_phy_baseband_processor.h \
  /src/lib/phy/lower/lower_phy_baseband_processor.h
cp /patch/pdxch_processor_impl.cpp \
  /src/lib/phy/lower/processors/downlink/pdxch/pdxch_processor_impl.cpp
cp /patch/mac_cell_processor.cpp \
  /src/lib/mac/mac_dl/mac_cell_processor.cpp
cp /patch/mac_ue_creation_procedure.cpp \
  /src/lib/mac/mac_ctrl/ue_creation_procedure.cpp
cp /patch/mac_ul_processor.cpp \
  /src/lib/mac/mac_ul/mac_ul_processor.cpp
cp /patch/ocudu_scheduler_adapter.cpp \
  /src/lib/mac/mac_sched/ocudu_scheduler_adapter.cpp
grep -q 'return launch_no_op_task();' /src/lib/mac/mac_dl/mac_cell_processor.cpp
grep -q 'async_no_op_task.h' /src/lib/mac/mac_dl/mac_cell_processor.cpp || \
  sed -i '/#include "ocudu\/support\/async\/async_timer.h"/i #include "ocudu/support/async/async_no_op_task.h"' \
    /src/lib/mac/mac_dl/mac_cell_processor.cpp
grep -q 'Blocking ZeroMQ request/reply mode can quiesce cell_exec' /src/lib/mac/mac_dl/mac_cell_processor.cpp
grep -q 'MAC_CREATE_DIAGNOSTIC ul-dl-begin' /src/lib/mac/mac_ctrl/ue_creation_procedure.cpp
grep -q 'acknowledge it immediately' /src/lib/mac/mac_sched/ocudu_scheduler_adapter.cpp
cmake --build /src/build --target gnb -- -j2
cp /src/build/apps/gnb/gnb /out/gnb
sha256sum /out/gnb
