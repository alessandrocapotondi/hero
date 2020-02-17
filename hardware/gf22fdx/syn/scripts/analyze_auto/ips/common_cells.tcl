puts "${Green}Analyzing common_cells ${NC}"

puts "${Green}--> compile common_cells_all${NC}"
analyze -format sv -work work \
    -define { \
        TARGET_SYNOPSYS \
        TARGET_SYNTHESIS \
    } \
    [list \
        "${IPS_PATH}/common_cells/src/addr_decode.sv" \
        "${IPS_PATH}/common_cells/src/cdc_2phase.sv" \
        "${IPS_PATH}/common_cells/src/cf_math_pkg.sv" \
        "${IPS_PATH}/common_cells/src/clk_div.sv" \
        "${IPS_PATH}/common_cells/src/delta_counter.sv" \
        "${IPS_PATH}/common_cells/src/edge_propagator_tx.sv" \
        "${IPS_PATH}/common_cells/src/exp_backoff.sv" \
        "${IPS_PATH}/common_cells/src/fifo_v3.sv" \
        "${IPS_PATH}/common_cells/src/graycode.sv" \
        "${IPS_PATH}/common_cells/src/lfsr.sv" \
        "${IPS_PATH}/common_cells/src/lfsr_16bit.sv" \
        "${IPS_PATH}/common_cells/src/lfsr_8bit.sv" \
        "${IPS_PATH}/common_cells/src/lzc.sv" \
        "${IPS_PATH}/common_cells/src/mv_filter.sv" \
        "${IPS_PATH}/common_cells/src/onehot_to_bin.sv" \
        "${IPS_PATH}/common_cells/src/plru_tree.sv" \
        "${IPS_PATH}/common_cells/src/popcount.sv" \
        "${IPS_PATH}/common_cells/src/rr_arb_tree.sv" \
        "${IPS_PATH}/common_cells/src/rstgen_bypass.sv" \
        "${IPS_PATH}/common_cells/src/serial_deglitch.sv" \
        "${IPS_PATH}/common_cells/src/shift_reg.sv" \
        "${IPS_PATH}/common_cells/src/spill_register.sv" \
        "${IPS_PATH}/common_cells/src/stream_demux.sv" \
        "${IPS_PATH}/common_cells/src/stream_filter.sv" \
        "${IPS_PATH}/common_cells/src/stream_fork.sv" \
        "${IPS_PATH}/common_cells/src/stream_join.sv" \
        "${IPS_PATH}/common_cells/src/stream_mux.sv" \
        "${IPS_PATH}/common_cells/src/sub_per_hash.sv" \
        "${IPS_PATH}/common_cells/src/sync.sv" \
        "${IPS_PATH}/common_cells/src/sync_wedge.sv" \
        "${IPS_PATH}/common_cells/src/unread.sv" \
        "${IPS_PATH}/common_cells/src/cb_filter.sv" \
        "${IPS_PATH}/common_cells/src/cdc_fifo_2phase.sv" \
        "${IPS_PATH}/common_cells/src/cdc_fifo_gray.sv" \
        "${IPS_PATH}/common_cells/src/counter.sv" \
        "${IPS_PATH}/common_cells/src/edge_detect.sv" \
        "${IPS_PATH}/common_cells/src/id_queue.sv" \
        "${IPS_PATH}/common_cells/src/max_counter.sv" \
        "${IPS_PATH}/common_cells/src/rstgen.sv" \
        "${IPS_PATH}/common_cells/src/stream_delay.sv" \
        "${IPS_PATH}/common_cells/src/stream_fifo.sv" \
        "${IPS_PATH}/common_cells/src/stream_fork_dynamic.sv" \
        "${IPS_PATH}/common_cells/src/fall_through_register.sv" \
        "${IPS_PATH}/common_cells/src/mem_to_stream.sv" \
        "${IPS_PATH}/common_cells/src/stream_arbiter_flushable.sv" \
        "${IPS_PATH}/common_cells/src/stream_register.sv" \
        "${IPS_PATH}/common_cells/src/stream_arbiter.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/clock_divider_counter.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/find_first_one.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/generic_LFSR_8bit.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/generic_fifo.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/prioarbiter.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/pulp_sync.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/pulp_sync_wedge.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/rrarbiter.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/clock_divider.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/fifo_v2.sv" \
        "${IPS_PATH}/common_cells/src/deprecated/fifo_v1.sv" \
        "${IPS_PATH}/common_cells/src/edge_propagator.sv" \
        "${IPS_PATH}/common_cells/src/edge_propagator_rx.sv" \
    ]
