// © IBM Corp. 2020
// This softcore is licensed under and subject to the terms of the CC-BY 4.0
// license (https://creativecommons.org/licenses/by/4.0/legalcode). 
// Additional rights, including the right to physically implement a softcore 
// that is compliant with the required sections of the Power ISA 
// Specification, will be available at no cost via the OpenPOWER Foundation. 
// This README will be updated with additional information when OpenPOWER's 
// license is available.

`timescale 1 ns / 1 ns

   `include "tri_a2o.vh"


module tri_fu_csa22_h2(
   a,
   b,
   car,
   sum
);
   input   a;
   input   b;
   output  car;
   output  sum;

   wire    car_b;
   wire    sum_b;

   assign car_b = (~(a & b));
   assign sum_b = (~(car_b & (a | b)));		// this is equiv to an xnor
   assign car = (~car_b);
   assign sum = (~sum_b);

endmodule
