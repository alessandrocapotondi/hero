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
// *! FILENAME    : tri_aoi22.v
// *! DESCRIPTION : AOI22 gate
// *!
// *!****************************************************************

`include "tri_a2o.vh"

module tri_aoi22(
   y,
   a0,
   a1,
   b0,
   b1
);
   parameter                      WIDTH = 1;
   parameter                      BTR = "AOI22_X2M_NONE";  //Specify full BTR name, else let tool select
   output [0:WIDTH-1]  y;
   input [0:WIDTH-1]   a0;
   input [0:WIDTH-1]   a1;
   input [0:WIDTH-1]   b0;
   input [0:WIDTH-1]   b1;

   // tri_aoi22
   genvar 	       i;
   wire [0:WIDTH-1]    outA;
   wire [0:WIDTH-1]    outB;

   generate
      begin : t
	 for (i = 0; i < WIDTH; i = i + 1)
	   begin : w

	      and I0(outA[i], a0[i], a1[i]);
	      and I1(outB[i], b0[i], b1[i]);
	      nor I2(y[i], outA[i], outB[i]);


	   end // block: w
      end

   endgenerate
endmodule
