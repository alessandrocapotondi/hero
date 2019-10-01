// Copyright 2019 ETH Zurich and University of Bologna.
// All rights reserved.
//
// This code is under development and not yet released to the public.
// Until it is released, the code is under the copyright of ETH Zurich and
// the University of Bologna, and may contain confidential and/or unpublished
// work. Any reuse/redistribution is strictly forbidden without written
// permission from ETH Zurich.

module axi_rab_wrap #(
  // L1 Configuration
  parameter int unsigned  L1NumSlicesPulp = 0,
  parameter int unsigned  L1NumSlicesHost = 0,
  // L2 Configuration
  parameter bit           L2Enable = 1'b0,
  parameter int unsigned  L2NumSets = 0,
  parameter int unsigned  L2NumSetEntries = 0,
  parameter int unsigned  L2NumParVaRams = 0,
  // Miss Handler FIFO Configuration
  parameter int unsigned  MhFifoDepth = 0,
  // AXI Configuration
  parameter int unsigned  AxiAddrWidth = 0,
  parameter int unsigned  AxiDataWidth = 0,
  parameter int unsigned  AxiIdWidth = 0,
  parameter int unsigned  AxiUserWidth = 0,
  parameter type          axi_req_t = logic,
  parameter type          axi_resp_t = logic
) (
  input  logic clk_i,
  input  logic rst_ni,

  // Transactions coming from PULP and going to Host
  input  axi_req_t  from_pulp_req_i,
  output axi_resp_t from_pulp_resp_o,
  output logic      from_pulp_miss_irq_o,
  output logic      from_pulp_multi_irq_o,
  output logic      from_pulp_prot_irq_o,
  output axi_req_t  to_host_req_o,
  input  axi_resp_t to_host_resp_i,

  // Transactions coming from Host and going to PULP
  input  axi_req_t  from_host_req_i,
  output axi_resp_t from_host_resp_o,
  output logic      from_host_miss_irq_o,
  output logic      from_host_multi_irq_o,
  output logic      from_host_prot_irq_o,
  output axi_req_t  to_pulp_req_o,
  input  axi_resp_t to_pulp_resp_i,

  output logic mh_fifo_full_irq_o
);

  axi_rab_top #(
    .N_PORTS              (2),
    .N_L1_SLICES          ('{0, 0, L1NumSlicesPulp, L1NumSlicesHost}),
    .N_L1_SLICES_MAX      (L1NumSlicesPulp > L1NumSlicesHost ? L1NumSlicesPulp : L1NumSlicesHost),
    .EN_ACP               (1'b0),
    .ENABLE_L2TLB         ('{1'b0, 1'b0, L2Enable, 1'b0}),
    .N_L2_SETS            (L2NumSets),
    .N_L2_SET_ENTRIES     (L2NumSetEntries),
    .N_L2_PAR_VA_RAMS     (L2NumParVaRams),
    .AXI_DATA_WIDTH       (AxiDataWidth),
    .AXI_S_ADDR_WIDTH     (AxiAddrWidth),
    .AXI_M_ADDR_WIDTH     (AxiAddrWidth),
    .AXI_LITE_DATA_WIDTH  (64/* TODO */),
    .AXI_LITE_ADDR_WIDTH  (32/* TODO */),
    .AXI_ID_WIDTH         (AxiIdWidth),
    .AXI_USER_WIDTH       (AxiUserWidth),
    .MH_FIFO_DEPTH        (MhFifoDepth)
  ) i_rab (
    .Clk_CI         (clk_i),
    .NonGatedClk_CI (clk_i),
    .Rst_RBI        (rst_ni),

    // AXI4 Slave {{{
    .s_axi4_awid      ({from_pulp_req_i.aw.id,      from_host_req_i.aw.id}),
    .s_axi4_awaddr    ({from_pulp_req_i.aw.addr,    from_host_req_i.aw.addr}),
    .s_axi4_awvalid   ({from_pulp_req_i.aw_valid,   from_host_req_i.aw_valid}),
    .s_axi4_awready   ({from_pulp_resp_o.aw_ready,  from_host_resp_o.aw_ready}),
    .s_axi4_awlen     ({from_pulp_req_i.aw.len,     from_host_req_i.aw.len}),
    .s_axi4_awsize    ({from_pulp_req_i.aw.size,    from_host_req_i.aw.size}),
    .s_axi4_awburst   ({from_pulp_req_i.aw.burst,   from_host_req_i.aw.burst}),
    .s_axi4_awlock    ({from_pulp_req_i.aw.lock,    from_host_req_i.aw.lock}),
    .s_axi4_awprot    ({from_pulp_req_i.aw.prot,    from_host_req_i.aw.prot}),
    .s_axi4_awatop    ({from_pulp_req_i.aw.atop,    from_host_req_i.aw.atop}),
    .s_axi4_awcache   ({from_pulp_req_i.aw.cache,   from_host_req_i.aw.cache}),
    .s_axi4_awregion  ({from_pulp_req_i.aw.region,  from_host_req_i.aw.region}),
    .s_axi4_awqos     ({from_pulp_req_i.aw.qos,     from_host_req_i.aw.qos}),
    .s_axi4_awuser    ({from_pulp_req_i.aw.user,    from_host_req_i.aw.user}),

    .s_axi4_wdata     ({from_pulp_req_i.w.data,     from_host_req_i.w.data}),
    .s_axi4_wvalid    ({from_pulp_req_i.w_valid,    from_host_req_i.w_valid}),
    .s_axi4_wready    ({from_pulp_resp_o.w_ready,   from_host_resp_o.w_ready}),
    .s_axi4_wstrb     ({from_pulp_req_i.w.strb,     from_host_req_i.w.strb}),
    .s_axi4_wlast     ({from_pulp_req_i.w.last,     from_host_req_i.w.last}),
    .s_axi4_wuser     ({from_pulp_req_i.w.user,     from_host_req_i.w.user}),

    .s_axi4_bid       ({from_pulp_resp_o.b.id,      from_host_resp_o.b.id}),
    .s_axi4_bresp     ({from_pulp_resp_o.b.resp,    from_host_resp_o.b.resp}),
    .s_axi4_bvalid    ({from_pulp_resp_o.b_valid,   from_host_resp_o.b_valid}),
    .s_axi4_buser     ({from_pulp_resp_o.b.user,    from_host_resp_o.b.user}),
    .s_axi4_bready    ({from_pulp_req_i.b_ready,    from_host_req_i.b_ready}),

    .s_axi4_arid      ({from_pulp_req_i.ar.id,      from_host_req_i.ar.id}),
    .s_axi4_araddr    ({from_pulp_req_i.ar.addr,    from_host_req_i.ar.addr}),
    .s_axi4_arvalid   ({from_pulp_req_i.ar_valid,   from_host_req_i.ar_valid}),
    .s_axi4_arready   ({from_pulp_resp_o.ar_ready,  from_host_resp_o.ar_ready}),
    .s_axi4_arlen     ({from_pulp_req_i.ar.len,     from_host_req_i.ar.len}),
    .s_axi4_arsize    ({from_pulp_req_i.ar.size,    from_host_req_i.ar.size}),
    .s_axi4_arburst   ({from_pulp_req_i.ar.burst,   from_host_req_i.ar.burst}),
    .s_axi4_arlock    ({from_pulp_req_i.ar.lock,    from_host_req_i.ar.lock}),
    .s_axi4_arprot    ({from_pulp_req_i.ar.prot,    from_host_req_i.ar.prot}),
    .s_axi4_arcache   ({from_pulp_req_i.ar.cache,   from_host_req_i.ar.cache}),
    .s_axi4_arregion  ({from_pulp_req_i.ar.region,  from_host_req_i.ar.region}),
    .s_axi4_arqos     ({from_pulp_req_i.ar.qos,     from_host_req_i.ar.qos}),
    .s_axi4_aruser    ({from_pulp_req_i.ar.user,    from_host_req_i.ar.user}),

    .s_axi4_rid       ({from_pulp_resp_o.r.id,      from_host_resp_o.r.id}),
    .s_axi4_rdata     ({from_pulp_resp_o.r.data,    from_host_resp_o.r.data}),
    .s_axi4_rresp     ({from_pulp_resp_o.r.resp,    from_host_resp_o.r.resp}),
    .s_axi4_rvalid    ({from_pulp_resp_o.r_valid,   from_host_resp_o.r_valid}),
    .s_axi4_rready    ({from_pulp_req_i.r_ready,    from_host_req_i.r_ready}),
    .s_axi4_rlast     ({from_pulp_resp_o.r.last,    from_host_resp_o.r.last}),
    .s_axi4_ruser     ({from_pulp_resp_o.r.user,    from_host_resp_o.r.user}),
    // }}}

    // AXI4 Master 0 {{{
    .m0_axi4_awid     ({to_host_req_o.aw.id,        to_pulp_req_o.aw.id}),
    .m0_axi4_awaddr   ({to_host_req_o.aw.addr,      to_pulp_req_o.aw.addr}),
    .m0_axi4_awvalid  ({to_host_req_o.aw_valid,     to_pulp_req_o.aw_valid}),
    .m0_axi4_awready  ({to_host_resp_i.aw_ready,    to_pulp_resp_i.aw_ready}),
    .m0_axi4_awlen    ({to_host_req_o.aw.len,       to_pulp_req_o.aw.len}),
    .m0_axi4_awsize   ({to_host_req_o.aw.size,      to_pulp_req_o.aw.size}),
    .m0_axi4_awburst  ({to_host_req_o.aw.burst,     to_pulp_req_o.aw.burst}),
    .m0_axi4_awlock   ({to_host_req_o.aw.lock,      to_pulp_req_o.aw.lock}),
    .m0_axi4_awprot   ({to_host_req_o.aw.prot,      to_pulp_req_o.aw.prot}),
    .m0_axi4_awatop   ({to_host_req_o.aw.atop,      to_pulp_req_o.aw.atop}),
    .m0_axi4_awcache  ({to_host_req_o.aw.cache,     to_pulp_req_o.aw.cache}),
    .m0_axi4_awregion ({to_host_req_o.aw.region,    to_pulp_req_o.aw.region}),
    .m0_axi4_awqos    ({to_host_req_o.aw.qos,       to_pulp_req_o.aw.qos}),
    .m0_axi4_awuser   ({to_host_req_o.aw.user,      to_pulp_req_o.aw.user}),

    .m0_axi4_wdata    ({to_host_req_o.w.data,       to_pulp_req_o.w.data}),
    .m0_axi4_wvalid   ({to_host_req_o.w_valid,      to_pulp_req_o.w_valid}),
    .m0_axi4_wready   ({to_host_resp_i.w_ready,     to_pulp_resp_i.w_ready}),
    .m0_axi4_wstrb    ({to_host_req_o.w.strb,       to_pulp_req_o.w.strb}),
    .m0_axi4_wlast    ({to_host_req_o.w.last,       to_pulp_req_o.w.last}),
    .m0_axi4_wuser    ({to_host_req_o.w.user,       to_pulp_req_o.w.user}),

    .m0_axi4_bid      ({to_host_resp_i.b.id,        to_pulp_resp_i.b.id}),
    .m0_axi4_bresp    ({to_host_resp_i.b.resp,      to_pulp_resp_i.b.resp}),
    .m0_axi4_bvalid   ({to_host_resp_i.b_valid,     to_pulp_resp_i.b_valid}),
    .m0_axi4_buser    ({to_host_resp_i.b.user,      to_pulp_resp_i.b.user}),
    .m0_axi4_bready   ({to_host_req_o.b_ready,      to_pulp_req_o.b_ready}),

    .m0_axi4_arid     ({to_host_req_o.ar.id,        to_pulp_req_o.ar.id}),
    .m0_axi4_araddr   ({to_host_req_o.ar.addr,      to_pulp_req_o.ar.addr}),
    .m0_axi4_arvalid  ({to_host_req_o.ar_valid,     to_pulp_req_o.ar_valid}),
    .m0_axi4_arready  ({to_host_resp_i.ar_ready,    to_pulp_resp_i.ar_ready}),
    .m0_axi4_arlen    ({to_host_req_o.ar.len,       to_pulp_req_o.ar.len}),
    .m0_axi4_arsize   ({to_host_req_o.ar.size,      to_pulp_req_o.ar.size}),
    .m0_axi4_arburst  ({to_host_req_o.ar.burst,     to_pulp_req_o.ar.burst}),
    .m0_axi4_arlock   ({to_host_req_o.ar.lock,      to_pulp_req_o.ar.lock}),
    .m0_axi4_arprot   ({to_host_req_o.ar.prot,      to_pulp_req_o.ar.prot}),
    .m0_axi4_arcache  ({to_host_req_o.ar.cache,     to_pulp_req_o.ar.cache}),
    .m0_axi4_arregion ({to_host_req_o.ar.region,    to_pulp_req_o.ar.region}),
    .m0_axi4_arqos    ({to_host_req_o.ar.qos,       to_pulp_req_o.ar.qos}),
    .m0_axi4_aruser   ({to_host_req_o.ar.user,      to_pulp_req_o.ar.user}),

    .m0_axi4_rid      ({to_host_resp_i.r.id,        to_pulp_resp_i.r.id}),
    .m0_axi4_rdata    ({to_host_resp_i.r.data,      to_pulp_resp_i.r.data}),
    .m0_axi4_rresp    ({to_host_resp_i.r.resp,      to_pulp_resp_i.r.resp}),
    .m0_axi4_rvalid   ({to_host_resp_i.r_valid,     to_pulp_resp_i.r_valid}),
    .m0_axi4_rready   ({to_host_req_o.r_ready,      to_pulp_req_o.r_ready}),
    .m0_axi4_rlast    ({to_host_resp_i.r.last,      to_pulp_resp_i.r.last}),
    .m0_axi4_ruser    ({to_host_resp_i.r.user,      to_pulp_resp_i.r.user}),
    // }}}

    // AXI4 Master 1 {{{
    .m1_axi4_awid     (/* unused */),
    .m1_axi4_awaddr   (/* unused */),
    .m1_axi4_awvalid  (/* unused */),
    .m1_axi4_awready  ('0),
    .m1_axi4_awlen    (/* unused */),
    .m1_axi4_awsize   (/* unused */),
    .m1_axi4_awburst  (/* unused */),
    .m1_axi4_awlock   (/* unused */),
    .m1_axi4_awprot   (/* unused */),
    .m1_axi4_awatop   (/* unused */),
    .m1_axi4_awcache  (/* unused */),
    .m1_axi4_awregion (/* unused */),
    .m1_axi4_awqos    (/* unused */),
    .m1_axi4_awuser   (/* unused */),

    .m1_axi4_wdata    (/* unused */),
    .m1_axi4_wvalid   (/* unused */),
    .m1_axi4_wready   ('0),
    .m1_axi4_wstrb    (/* unused */),
    .m1_axi4_wlast    (/* unused */),
    .m1_axi4_wuser    (/* unused */),

    .m1_axi4_bid      ('0),
    .m1_axi4_bresp    ('0),
    .m1_axi4_bvalid   ('0),
    .m1_axi4_buser    ('0),
    .m1_axi4_bready   (/* unused */),

    .m1_axi4_arid     (/* unused */),
    .m1_axi4_araddr   (/* unused */),
    .m1_axi4_arvalid  (/* unused */),
    .m1_axi4_arready  ('0),
    .m1_axi4_arlen    (/* unused */),
    .m1_axi4_arsize   (/* unused */),
    .m1_axi4_arburst  (/* unused */),
    .m1_axi4_arlock   (/* unused */),
    .m1_axi4_arprot   (/* unused */),
    .m1_axi4_arcache  (/* unused */),
    .m1_axi4_arregion (/* unused */),
    .m1_axi4_arqos    (/* unused */),
    .m1_axi4_aruser   (/* unused */),

    .m1_axi4_rid      ('0),
    .m1_axi4_rdata    ('0),
    .m1_axi4_rresp    ('0),
    .m1_axi4_rvalid   ('0),
    .m1_axi4_rready   (/* unused */),
    .m1_axi4_rlast    ('0),
    .m1_axi4_ruser    ('0),
    // }}}

    // AXI4 Lite Slave (Configuration Interface) {{{
    .s_axi4lite_awaddr  ('0/* TODO */),
    .s_axi4lite_awvalid ('0/* TODO */),
    .s_axi4lite_awready (/* TODO */),

    .s_axi4lite_wdata   ('0/* TODO */),
    .s_axi4lite_wvalid  ('0/* TODO */),
    .s_axi4lite_wready  (/* TODO */),
    .s_axi4lite_wstrb   ('0/* TODO */),

    .s_axi4lite_bresp   (/* TODO */),
    .s_axi4lite_bvalid  (/* TODO */),
    .s_axi4lite_bready  ('0/* TODO */),

    .s_axi4lite_araddr  ('0/* TODO */),
    .s_axi4lite_arvalid ('0/* TODO */),
    .s_axi4lite_arready (/* TODO */),

    .s_axi4lite_rdata   (/* TODO */),
    .s_axi4lite_rresp   (/* TODO */),
    .s_axi4lite_rvalid  (/* TODO */),
    .s_axi4lite_rready  ('0/* TODO */),
    // }}}

    .int_miss     ({from_pulp_miss_irq_o,   from_host_miss_irq_o}),
    .int_multi    ({from_pulp_multi_irq_o,  from_host_multi_irq_o}),
    .int_prot     ({from_pulp_prot_irq_o,   from_host_prot_irq_o}),
    .int_mhf_full (mh_fifo_full_irq_o)
  );

endmodule

// vim: foldmethod=marker
