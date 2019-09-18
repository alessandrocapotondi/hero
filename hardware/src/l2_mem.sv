// Copyright 2019 ETH Zurich and University of Bologna.
// All rights reserved.
//
// This code is under development and not yet released to the public.
// Until it is released, the code is under the copyright of ETH Zurich and
// the University of Bologna, and may contain confidential and/or unpublished
// work. Any reuse/redistribution is strictly forbidden without written
// permission from ETH Zurich.

module l2_mem #(
  parameter int unsigned  AXI_AW = 0,   // [bit], must be a power of 2
  parameter int unsigned  AXI_DW = 0,   // [bit], must be a power of 2
  parameter int unsigned  AXI_IW = 0,   // [bit]
  parameter int unsigned  AXI_UW = 0,   // [bit]
  // Memory
  parameter int unsigned  N_BYTES = 0   // [B], must be a power of 2
) (
  input  logic  clk_i,
  input  logic  rst_ni,
  AXI_BUS.Slave slv
);

  // Properties of one memory cut, keep synchronized with instantiated macro.
  localparam int unsigned CUT_DW = 64;          // [bit], must be a power of 2 and >=8
  localparam int unsigned CUT_N_WORDS = 8192;   // must be a power of 2
  localparam int unsigned CUT_N_BITS = CUT_DW * CUT_N_WORDS;

  // Derived properties of memory array
  localparam int unsigned N_PAR_CUTS = AXI_DW / CUT_DW;
  localparam int unsigned PAR_CUTS_N_BYTES = N_PAR_CUTS * CUT_N_BITS / 8;
  localparam int unsigned N_SER_CUTS = N_BYTES / PAR_CUTS_N_BYTES;

  // Types for entire memory array
  typedef logic   [AXI_AW-1:0] arr_addr_t;
  typedef logic   [AXI_DW-1:0] arr_data_t;
  typedef logic [AXI_DW/8-1:0] arr_strb_t;

  // Types for one memory cut
  typedef logic [$clog2(CUT_N_WORDS)-1:0] cut_addr_t;
  typedef logic [CUT_DW-1:0]              cut_data_t;
  typedef logic [CUT_DW/8-1:0]            cut_strb_t;

  // Interface from AXI to memory array
  logic       req, we;
  arr_addr_t  addr;
  arr_data_t  wdata, rdata;
  arr_strb_t  be;

  axi_mem_if #(
    .AXI_ID_WIDTH   (AXI_IW),
    .AXI_ADDR_WIDTH (AXI_AW),
    .AXI_DATA_WIDTH (AXI_DW),
    .AXI_USER_WIDTH (AXI_UW)
  ) i_axi_if (
    .clk_i,
    .rst_ni,
    .slave  (slv),
    .req_o  (req),
    .we_o   (we),
    .addr_o (addr),
    .be_o   (be),
    .data_o (wdata),
    .data_i (rdata)
  );

  // Interface from memory array to memory cuts
  localparam int unsigned WORD_IDX_OFF = $clog2(AXI_DW/8);
  localparam int unsigned WORD_IDX_WIDTH = $clog2(CUT_N_WORDS);
  localparam int unsigned ROW_IDX_OFF = WORD_IDX_OFF + WORD_IDX_WIDTH;
  localparam int unsigned ROW_IDX_WIDTH = $clog2(N_SER_CUTS);
  logic       [N_SER_CUTS-1:0]                  cut_req;
  cut_addr_t                                    cut_addr_d, cut_addr_q;
  cut_data_t  [N_SER_CUTS-1:0][N_PAR_CUTS-1:0]  cut_rdata;
  cut_data_t                  [N_PAR_CUTS-1:0]  cut_wdata;
  cut_strb_t                  [N_PAR_CUTS-1:0]  cut_be;

  assign cut_addr_d = req ? addr[ROW_IDX_OFF-1:WORD_IDX_OFF] : cut_addr_q;
  if (ROW_IDX_WIDTH > 0) begin: gen_row_idx
    logic [ROW_IDX_WIDTH-1:0]row_idx_d, row_idx_q;
    assign row_idx_d = req ? addr[ROW_IDX_OFF+:ROW_IDX_WIDTH] : row_idx_q;
    always_comb begin
      cut_req = '0;
      cut_req[row_idx_d] = req;
    end
    assign rdata = cut_rdata[row_idx_d];
    always_ff @(posedge clk_i, negedge rst_ni) begin
      if (!rst_ni) begin
        row_idx_q <= '0;
      end else begin
        row_idx_q <= row_idx_d;
      end
    end
  end else begin: gen_no_row_idx
    assign cut_req = req;
    assign rdata = cut_rdata;
  end
  assign cut_wdata = wdata;
  assign cut_be = be;

  for (genvar iRow = 0; iRow < N_SER_CUTS; iRow++) begin: gen_rows
    for (genvar iCol = 0; iCol < N_PAR_CUTS; iCol++) begin: gen_cols
      sram #(
        .DATA_WIDTH (CUT_DW),
        .N_WORDS    (CUT_N_WORDS)
      ) i_mem_cut (
        .clk_i,
        .rst_ni,
        .req_i    (cut_req[iRow]),
        .we_i     (we),
        .addr_i   (cut_addr_d),
        .wdata_i  (cut_wdata[iCol]),
        .be_i     (cut_be[iCol]),
        .rdata_o  (cut_rdata[iRow][iCol])
      );
    end
  end

  always_ff @(posedge clk_i, negedge rst_ni) begin
    if (!rst_ni) begin
      cut_addr_q <= '0;
    end else begin
      cut_addr_q <= cut_addr_d;
    end
  end

  // Validate parameters and properties.
  // pragma translate_off
  initial begin
    assert (AXI_AW > 0);
    assert (AXI_AW % (2**$clog2(AXI_AW)) == 0);
    assert (AXI_DW > 0);
    assert (AXI_DW % (2**$clog2(AXI_DW)) == 0);
    assert (N_BYTES > 0);
    assert (N_BYTES % (2**$clog2(N_BYTES)) == 0);
    assert (CUT_DW % (2**$clog2(CUT_DW)) == 0);
    assert (CUT_DW >= 8);
    assert (AXI_DW >= CUT_DW);
    assert (CUT_N_WORDS % 2**$clog2(CUT_N_WORDS) == 0);
    assert (N_BYTES % PAR_CUTS_N_BYTES == 0);
  end
  // pragma translate_on

endmodule