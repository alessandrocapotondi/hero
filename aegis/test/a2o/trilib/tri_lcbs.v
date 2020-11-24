// © IBM Corp. 2020
// This softcore is licensed under and subject to the terms of the CC-BY 4.0
// license (https://creativecommons.org/licenses/by/4.0/legalcode). 
// Additional rights, including the right to physically implement a softcore 
// that is compliant with the required sections of the Power ISA 
// Specification, will be available at no cost via the OpenPOWER Foundation. 
// This README will be updated with additional information when OpenPOWER's 
// license is available.

`timescale 1 ns / 1 ns

// *!****************************************************************
// *! FILENAME    : tri_lcbs.v
// *! DESCRIPTION : Wrapper for slat LCB
// *!****************************************************************

`include "tri_a2o.vh"

module tri_lcbs(
   vd,
   gd,
   delay_lclkr,
   nclk,
   force_t,
   thold_b,
   dclk,
   lclk
);
   inout      vd;
   inout      gd;
   input      delay_lclkr;
   input[0:`NCLK_WIDTH-1]  nclk;
   input      force_t;
   input      thold_b;
   output     dclk;
   output[0:`NCLK_WIDTH-1]  lclk;

   //  tri_lcbs

   (* analysis_not_referenced="true" *)
   wire       unused;

   assign unused = vd | gd | delay_lclkr | force_t;

   // No scan chain in this methodology
   assign dclk = thold_b;
   assign lclk = nclk;
endmodule
