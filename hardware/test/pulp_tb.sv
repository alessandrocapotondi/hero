// Copyright 2019 ETH Zurich and University of Bologna.
// All rights reserved.
//
// This code is under development and not yet released to the public.
// Until it is released, the code is under the copyright of ETH Zurich and
// the University of Bologna, and may contain confidential and/or unpublished
// work. Any reuse/redistribution is strictly forbidden without written
// permission from ETH Zurich.

`include "axi/assign.svh"
`include "axi/typedef.svh"

`define wait_for(signal) \
  do \
    @(posedge clk); \
  while (!signal);

module pulp_tb #(
  // TB Parameters
  parameter time          CLK_PERIOD = 1000ps,
  parameter bit           RUN_DM_TESTS = 0,
  // SoC Parameters
  parameter int unsigned  N_CLUSTERS = 1,
  parameter int unsigned  AXI_DW = 128,
  parameter int unsigned  L2_N_AXI_PORTS = 1
);

  timeunit 1ps;
  timeprecision 1ps;

  localparam int unsigned AXI_IW = pulp_pkg::axi_iw_sb_oup(N_CLUSTERS);
  localparam int unsigned AXI_SW = AXI_DW/8;  // width of strobe
  typedef pulp_pkg::addr_t      axi_addr_t;
  typedef logic [AXI_DW-1:0]    axi_data_t;
  typedef logic [AXI_IW-1:0]    axi_id_t;
  typedef logic [AXI_SW-1:0]    axi_strb_t;
  typedef pulp_pkg::user_t      axi_user_t;
  `AXI_TYPEDEF_AW_CHAN_T(       axi_aw_t,     axi_addr_t, axi_id_t, axi_user_t);
  `AXI_TYPEDEF_W_CHAN_T(        axi_w_t,      axi_data_t, axi_strb_t, axi_user_t);
  `AXI_TYPEDEF_B_CHAN_T(        axi_b_t,      axi_id_t, axi_user_t);
  `AXI_TYPEDEF_AR_CHAN_T(       axi_ar_t,     axi_addr_t, axi_id_t, axi_user_t);
  `AXI_TYPEDEF_R_CHAN_T(        axi_r_t,      axi_data_t, axi_id_t, axi_user_t);
  `AXI_TYPEDEF_REQ_T(           axi_req_t,    axi_aw_t, axi_w_t, axi_ar_t);
  `AXI_TYPEDEF_RESP_T(          axi_resp_t,   axi_b_t, axi_r_t);

  typedef pulp_pkg::lite_addr_t axi_lite_addr_t;
  typedef pulp_pkg::lite_data_t axi_lite_data_t;
  typedef pulp_pkg::lite_strb_t axi_lite_strb_t;
  `AXI_LITE_TYPEDEF_AW_CHAN_T(  axi_lite_aw_t,    axi_lite_addr_t);
  `AXI_LITE_TYPEDEF_W_CHAN_T(   axi_lite_w_t,     axi_lite_data_t, axi_lite_strb_t);
  `AXI_LITE_TYPEDEF_B_CHAN_T(   axi_lite_b_t);
  `AXI_LITE_TYPEDEF_AR_CHAN_T(  axi_lite_ar_t,    axi_lite_addr_t);
  `AXI_LITE_TYPEDEF_R_CHAN_T(   axi_lite_r_t,     axi_lite_data_t);
  `AXI_LITE_TYPEDEF_REQ_T(      axi_lite_req_t,   axi_lite_aw_t, axi_lite_w_t, axi_lite_ar_t);
  `AXI_LITE_TYPEDEF_RESP_T(     axi_lite_resp_t,  axi_lite_b_t, axi_lite_r_t);

  logic clk,
        rst_n;

  logic [N_CLUSTERS-1:0]  cl_busy,
                          cl_eoc,
                          cl_fetch_en;

  axi_req_t   from_pulp_req,
              to_pulp_req;
  axi_resp_t  from_pulp_resp,
              to_pulp_resp;

  logic jtag_tck;
  logic jtag_trst_n;
  logic jtag_tdi;
  logic jtag_tms;
  logic jtag_tdo;

  clk_rst_gen #(
    .CLK_PERIOD     (CLK_PERIOD),
    .RST_CLK_CYCLES (10)
  ) i_clk_gen (
    .clk_o  (clk),
    .rst_no (rst_n)
  );

  pulp #(
    .N_CLUSTERS     (N_CLUSTERS),
    .AXI_DW         (AXI_DW),
    .L2_N_AXI_PORTS (L2_N_AXI_PORTS),
    .axi_req_t      (axi_req_t),
    .axi_resp_t     (axi_resp_t),
    .axi_lite_req_t (axi_lite_req_t),
    .axi_lite_resp_t(axi_lite_resp_t)
  ) dut (
    .clk_i          (clk),
    .rst_ni         (rst_n),

    .cl_fetch_en_i  (cl_fetch_en),
    .cl_eoc_o       (cl_eoc),
    .cl_busy_o      (cl_busy),

    .ext_req_o      (from_pulp_req),
    .ext_resp_i     (from_pulp_resp),
    .ext_req_i      (to_pulp_req),
    .ext_resp_o     (to_pulp_resp),

    .jtag_tck_i     (jtag_tck),
    .jtag_trst_ni   (jtag_trst_n),
    .jtag_tdi_i     (jtag_tdi),
    .jtag_tms_i     (jtag_tms),
    .jtag_tdo_o     (jtag_tdo)
  );

  // AXI Node for Memory (slave 0) and Peripherals (slave 1)
  AXI_BUS #(
    .AXI_ADDR_WIDTH (pulp_pkg::AXI_AW),
    .AXI_DATA_WIDTH (AXI_DW),
    .AXI_ID_WIDTH   (AXI_IW),
    .AXI_USER_WIDTH (pulp_pkg::AXI_UW)
  ) from_pulp[0:0] ();
  AXI_BUS #(
    .AXI_ADDR_WIDTH (pulp_pkg::AXI_AW),
    .AXI_DATA_WIDTH (AXI_DW),
    .AXI_ID_WIDTH   (AXI_IW+1),
    .AXI_USER_WIDTH (pulp_pkg::AXI_UW)
  ) from_xbar[1:0] ();
  AXI_BUS #(
    .AXI_ADDR_WIDTH (pulp_pkg::AXI_AW),
    .AXI_DATA_WIDTH (64),
    .AXI_ID_WIDTH   (AXI_IW+1),
    .AXI_USER_WIDTH (pulp_pkg::AXI_UW)
  ) to_periphs ();
  `AXI_ASSIGN_FROM_REQ(from_pulp[0], from_pulp_req);
  `AXI_ASSIGN_TO_RESP (from_pulp_resp, from_pulp[0]);

  localparam int unsigned N_RULES = 1;
  axi_pkg::xbar_rule_32_t [N_RULES-1:0] addr_map;
  assign addr_map[0] = '{
    idx:        1,
    start_addr: 32'h1a10_0000,
    end_addr:   32'h1a11_0000
  };
  `ifndef VERILATOR
    initial begin
      assert (pulp_pkg::AXI_AW == 32)
        else $fatal("Address map is only defined for 32-bit addresses!");
    end
  `endif
  localparam int unsigned MAX_TXNS_PER_CLUSTER =  pulp_cluster_cfg_pkg::N_CORES +
                                                  pulp_cluster_cfg_pkg::DMA_MAX_N_TXNS;
  localparam axi_pkg::xbar_cfg_t xbar_cfg = '{
    NoSlvPorts:         1,
    NoMstPorts:         2,
    MaxMstTrans:        MAX_TXNS_PER_CLUSTER,
    MaxSlvTrans:        N_CLUSTERS * MAX_TXNS_PER_CLUSTER,
    FallThrough:        1'b0,
    LatencyMode:        axi_pkg::NO_LATENCY,
    AxiIdWidthSlvPorts: AXI_IW,
    AxiIdUsedSlvPorts:  AXI_IW,
    AxiAddrWidth:       pulp_pkg::AXI_AW,
    AxiDataWidth:       AXI_DW,
    NoAddrRules:        N_RULES
  };
  axi_xbar_intf #(
    .AXI_USER_WIDTH (pulp_pkg::AXI_UW),
    .Cfg            (xbar_cfg),
    .rule_t         (axi_pkg::xbar_rule_32_t)
  ) i_axi_xbar (
    .clk_i                  (clk),
    .rst_ni                 (rst_n),
    .test_i                 (1'b0),
    .slv_ports              (from_pulp),
    .mst_ports              (from_xbar),
    .addr_map_i             (addr_map),
    .en_default_mst_port_i  ({1{1'b1}}), // enable default master port for all slave ports
    .default_mst_port_i     ({1{1'b0}})  // use slave 0 as default master port
  );

  // Emulate infinite memory with AXI slave port.
  initial begin
    automatic logic [7:0] mem[axi_addr_t];
    automatic axi_ar_t ar_queue[$];
    automatic axi_aw_t aw_queue[$];
    automatic axi_b_t b_queue[$];
    automatic shortint unsigned r_cnt = 0, w_cnt = 0;
    from_xbar[0].aw_ready = 1'b0;
    from_xbar[0].w_ready = 1'b0;
    from_xbar[0].b_id = '0;
    from_xbar[0].b_resp = '0;
    from_xbar[0].b_user = '0;
    from_xbar[0].b_valid = 1'b0;
    from_xbar[0].ar_ready = 1'b0;
    from_xbar[0].r_id = '0;
    from_xbar[0].r_data = '0;
    from_xbar[0].r_resp = '0;
    from_xbar[0].r_last = 1'b0;
    from_xbar[0].r_user = '0;
    from_xbar[0].r_valid = 1'b0;
    wait (rst_n);
    @(posedge clk);
    fork
      // AW
      forever begin
        from_xbar[0].aw_ready = 1'b1;
        if (from_xbar[0].aw_valid) begin
          automatic axi_aw_t aw;
          `AXI_SET_TO_AW(aw, from_xbar[0]);
          aw_queue.push_back(aw);
        end
        @(posedge clk);
      end
      // W
      forever begin
        if (aw_queue.size() != 0) begin
          from_xbar[0].w_ready = 1'b1;
          if (from_xbar[0].w_valid) begin
            automatic axi_pkg::size_t size = aw_queue[0].size;
            automatic axi_addr_t addr = axi_pkg::beat_addr(aw_queue[0].addr, size, w_cnt);
            for (shortint unsigned
                i_byte = axi_pkg::beat_lower_byte(addr, size, AXI_SW, w_cnt);
                i_byte <= axi_pkg::beat_upper_byte(addr, size, AXI_SW, w_cnt);
                i_byte++) begin
              if (from_xbar[0].w_strb[i_byte]) begin
                automatic axi_addr_t byte_addr = (addr / AXI_SW) * AXI_SW + i_byte;
                mem[byte_addr] = from_xbar[0].w_data[i_byte*8+:8];
              end
            end
            if (w_cnt == aw_queue[0].len) begin
              automatic axi_b_t b_beat = '0;
              assert (from_xbar[0].w_last) else $error("Expected last beat of W burst!");
              b_beat.id = aw_queue[0].id;
              b_beat.resp = axi_pkg::RESP_OKAY;
              b_queue.push_back(b_beat);
              w_cnt = 0;
              void'(aw_queue.pop_front());
            end else begin
              assert (!from_xbar[0].w_last) else $error("Did not expect last beat of W burst!");
              w_cnt++;
            end
          end
        end else begin
          from_xbar[0].w_ready = 1'b0;
        end
        @(posedge clk);
      end
      // B
      forever begin
        if (b_queue.size() != 0) begin
          `AXI_SET_FROM_B(from_xbar[0], b_queue[0]);
          from_xbar[0].b_valid = 1'b1;
          @(posedge clk);
          if (from_xbar[0].b_ready) begin
            void'(b_queue.pop_front());
          end
        end else begin
          @(posedge clk);
        end
        from_xbar[0].b_valid = 1'b0;
      end
      // AR
      forever begin
        from_xbar[0].ar_ready = 1'b1;
        if (from_xbar[0].ar_valid) begin
          automatic axi_ar_t ar;
          `AXI_SET_TO_AR(ar, from_xbar[0]);
          ar_queue.push_back(ar);
        end
        @(posedge clk);
      end
      // R
      forever begin
        if (ar_queue.size() != 0) begin
          automatic axi_pkg::size_t size = ar_queue[0].size;
          automatic axi_addr_t addr = axi_pkg::beat_addr(ar_queue[0].addr, size, r_cnt);
          automatic axi_r_t r_beat = '0;
          r_beat.data = 'x;
          r_beat.id = ar_queue[0].id;
          r_beat.resp = axi_pkg::RESP_OKAY;
          for (shortint unsigned
              i_byte = axi_pkg::beat_lower_byte(addr, size, AXI_SW, r_cnt);
              i_byte <= axi_pkg::beat_upper_byte(addr, size, AXI_SW, r_cnt);
              i_byte++) begin
            automatic axi_addr_t byte_addr = (addr / AXI_SW) * AXI_SW + i_byte;
            if (!mem.exists(byte_addr)) begin
              $warning("Access to non-initialized byte at address 0x%016x by ID 0x%x.", byte_addr,
                  r_beat.id);
              r_beat.data[i_byte*8+:8] = 'x;
            end else begin
              r_beat.data[i_byte*8+:8] = mem[byte_addr];
            end
          end
          if (r_cnt == ar_queue[0].len) begin
            r_beat.last = 1'b1;
          end
          `AXI_SET_FROM_R(from_xbar[0], r_beat);
          from_xbar[0].r_valid = 1'b1;
          @(posedge clk);
          if (from_xbar[0].r_ready) begin
            if (r_beat.last) begin
              r_cnt = 0;
              void'(ar_queue.pop_front());
            end else begin
              r_cnt++;
            end
          end
        end else begin
          @(posedge clk);
        end
        from_xbar[0].r_valid = 1'b0;
      end
    join
  end

  // AXI Write
  task write_axi(input axi_addr_t addr, input axi_data_t data);
    @(posedge clk);
    to_pulp_req.aw.addr  = addr;
    to_pulp_req.aw.size  = 3'h2;
    to_pulp_req.aw_valid = 1'b1;
    to_pulp_req.aw.burst = axi_pkg::BURST_INCR;
    `wait_for(to_pulp_resp.aw_ready)
    to_pulp_req.aw_valid = 1'b0;
    to_pulp_req.aw       = '0;
    to_pulp_req.w.data   = data;
    to_pulp_req.w.strb   = '1;
    to_pulp_req.w.last   = 1'b1;
    to_pulp_req.w_valid  = 1'b1;
    `wait_for(to_pulp_resp.w_ready)
    to_pulp_req.w_valid  = 1'b0;
    to_pulp_req.w        = '0;
    to_pulp_req.b_ready  = 1'b1;
    `wait_for(to_pulp_resp.b_valid)
    to_pulp_req.b_ready  = 1'b0;
  endtask



  // Simulation control
  initial begin
    automatic dm_test::dm_if dm_if = new;
    automatic logic error;

    cl_fetch_en = '0;
    to_pulp_req = '0;
    // Wait for reset.
    wait (rst_n);
    @(posedge clk);

    // run dm tests instead of regular software tests
    if (RUN_DM_TESTS) begin
      // TODO: multiple write_axi seem to get stuck.
      // make cores loop on boot addr
      // disable icache for debug module test
      // write_axi(32'h1020_1400, 128'h00000000_00000000_00000000_00000000);
      // 1c008080
      //write_axi(32'h1c00_8080, 128'h00000000_00000000_00000000_1c008080);
      // enable all cores (TODO: what to do when multiple clusters)
      //write_axi(32'h1020_0008, 128'h00000000_000000ff_00000000_00000000);
      // TODO: remove this workaround when write_axi is fixed
      write_axi(32'h1020_0008, 128'h00000000_00000001_00000000_00000000);

      dm_if.run_dm_tests(N_CLUSTERS, pulp_cluster_cfg_pkg::N_CORES, error,
                           jtag_tck, jtag_tms, jtag_trst_n, jtag_tdi, jtag_tdo);
      $finish();
    end

    // Start cluster 0.
    // cl_fetch_en[0] = 1'b1;
    // Only start core 0. --> AXI write request
    write_axi(32'h1020_0008, 128'h00000000_00000001_00000000_00000000);

    // Wait for EOC of cluster 0 before terminating the simulation.
    wait (cl_eoc[0]);
    #1us;
    $finish();
  end

  // Fill TCDM memory.
  for (genvar iCluster = 0; iCluster < N_CLUSTERS; iCluster++) begin: gen_fill_tcdm_cluster
    for (genvar iBank = 0; iBank < 16; iBank++) begin: gen_fill_tcdm_bank
      initial begin
        automatic string slm_path;
        automatic int ignore;
        slm_path = "../test/slm_files"; // default path
        ignore = $value$plusargs("slm_path=%s", slm_path);
        $readmemh($sformatf("%s/l1_0_%01d.slm", slm_path, iBank),
          dut.gen_clusters[iCluster].gen_cluster_sync.i_cluster.i_ooc.i_bound.gen_tcdm_banks[iBank].i_mem.mem);
      end
    end
  end

  // Fill L2 memory.
  // Unroll parameters for VCS
  // ! If the parameters in the L2 memory are changed, the changes will not automatically be applied here
  // localparam N_SER_CUTS = dut.gen_l2_ports[0].i_l2_mem.N_SER_CUTS; // both same on all ports
  // localparam N_PAR_CUTS = dut.gen_l2_ports[0].i_l2_mem.N_PAR_CUTS;
  localparam N_PAR_CUTS = AXI_DW / 32;
  localparam N_SER_CUTS = (pulp_cluster_cfg_pkg::L2_SIZE/L2_N_AXI_PORTS) / (N_PAR_CUTS * (32 * 1024) / 8);
  for (genvar iPort = 0; iPort < L2_N_AXI_PORTS; iPort++) begin: gen_fill_l2_ports
    for (genvar iRow = 0; iRow < N_SER_CUTS; iRow++) begin: gen_fill_l2_rows
      for (genvar iCol = 0; iCol < N_PAR_CUTS; iCol++) begin: gen_fill_l2_cols
        int unsigned file_ser_idx = iPort*N_SER_CUTS + iRow;
        initial begin
          automatic string slm_path;
          automatic int ignore;
          slm_path = "../test/slm_files"; // default path
          ignore = $value$plusargs("slm_path=%s", slm_path);
          $readmemh($sformatf("%s/l2_%01d_%01d.slm", slm_path, file_ser_idx, iCol),
            dut.gen_l2_ports[iPort].i_l2_mem.gen_rows[iRow].gen_cols[iCol].i_mem_cut.mem);
        end
      end
    end
  end

  // Observe SoC bus for errors.
  for (genvar iCluster = 0; iCluster < N_CLUSTERS; iCluster++) begin: gen_assert_cluster
    assert property (@(posedge dut.clk_i) dut.rst_ni && dut.cl_oup[iCluster].r_valid
        |-> dut.cl_oup[iCluster].r_resp != 2'b11)
      else $warning("R resp decode error at cl_oup[%01d]", iCluster);

    assert property (@(posedge dut.clk_i) dut.rst_ni && dut.cl_oup[iCluster].b_valid
        |-> dut.cl_oup[iCluster].b_resp != 2'b11)
      else $warning("B resp decode error at cl_oup[%01d]", iCluster);
  end

endmodule
