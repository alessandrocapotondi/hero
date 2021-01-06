// © IBM Corp. 2020
// This softcore is licensed under and subject to the terms of the CC-BY 4.0
// license (https://creativecommons.org/licenses/by/4.0/legalcode). 
// Additional rights, including the right to physically implement a softcore 
// that is compliant with the required sections of the Power ISA 
// Specification, will be available at no cost via the OpenPOWER Foundation. 
// This README will be updated with additional information when OpenPOWER's 
// license is available.

`timescale 1 ns / 1 ns

////############################################################################
////#####  FU_GEST_loa.VHDL                                           #########
////#####  side pipe for graphics estimates                            #########
////#####  flogefp, fexptefp                                           #########
////#####                                                              #########
////############################################################################

module fu_gst_loa(
   a,
   shamt
);
   `include "tri_a2o.vh"


   input [1:19] a;

   output [0:4] shamt;


   wire         unused;

   assign unused = a[19];

   //@@ ESPRESSO TABLE START @@
   // ##################################################################################################

   // ##################################################################################################
   // .i 19
   // .o 5
   // .ilb a(01) a(02) a(03) a(04) a(05) a(06) a(07) a(08) a(09) a(10) a(11) a(12) a(13) a(14) a(15) a(16) a(17) a(18) a(19)
   // .ob  shamt(0) shamt(1) shamt(2) shamt(3) shamt(4)

   // .type fr
   ////#######################
   //
   // 0000000000000000001     10011
   // 000000000000000001-     10010
   // 00000000000000001--     10001
   // 0000000000000001---     10000
   // 000000000000001----     01111
   // 00000000000001-----     01110
   // 0000000000001------     01101
   // 000000000001-------     01100
   // 00000000001--------     01011
   // 0000000001---------     01010
   // 000000001----------     01001
   // 00000001-----------     01000
   // 0000001------------     00111
   // 000001-------------     00110
   // 00001--------------     00101
   // 0001---------------     00100
   // 001----------------     00011
   // 01-----------------     00010
   // 1------------------     00001
   // 0000000000000000000     00000

   // ###############################################################################
   // .e
   //@@ ESPRESSO TABLE END @@

   //@@ ESPRESSO LOGIC START @@
   // logic generated on: Tue Dec  4 13:14:17 2007
   assign shamt[0] = ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & (~a[08]) & (~a[09]) & (~a[10]) & (~a[11]) & (~a[12]) & (~a[13]) & (~a[14]) & (~a[15]) & a[19]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & (~a[08]) & (~a[09]) & (~a[10]) & (~a[11]) & (~a[12]) & (~a[13]) & (~a[14]) & (~a[15]) & a[18]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & (~a[08]) & (~a[09]) & (~a[10]) & (~a[11]) & (~a[12]) & (~a[13]) & (~a[14]) & (~a[15]) & a[17]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & (~a[08]) & (~a[09]) & (~a[10]) & (~a[11]) & (~a[12]) & (~a[13]) & (~a[14]) & (~a[15]) & a[16]);

   assign shamt[1] = ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & a[15]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & a[14]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & a[13]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & a[12]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & a[11]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & a[10]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & a[09]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[04]) & (~a[05]) & (~a[06]) & (~a[07]) & a[08]);

   assign shamt[2] = ((~a[01]) & (~a[02]) & (~a[03]) & (~a[08]) & (~a[09]) & (~a[10]) & (~a[11]) & a[15]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[08]) & (~a[09]) & (~a[10]) & (~a[11]) & a[14]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[08]) & (~a[09]) & (~a[10]) & (~a[11]) & a[13]) | ((~a[01]) & (~a[02]) & (~a[03]) & (~a[08]) & (~a[09]) & (~a[10]) & (~a[11]) & a[12]) | ((~a[01]) & (~a[02]) & (~a[03]) & a[07]) | ((~a[01]) & (~a[02]) & (~a[03]) & a[06]) | ((~a[01]) & (~a[02]) & (~a[03]) & a[05]) | ((~a[01]) & (~a[02]) & (~a[03]) & a[04]);

   assign shamt[3] = ((~a[01]) & (~a[04]) & (~a[05]) & (~a[08]) & (~a[09]) & (~a[12]) & (~a[13]) & (~a[16]) & (~a[17]) & a[19]) | ((~a[01]) & (~a[04]) & (~a[05]) & (~a[08]) & (~a[09]) & (~a[12]) & (~a[13]) & (~a[16]) & (~a[17]) & a[18]) | ((~a[01]) & (~a[04]) & (~a[05]) & (~a[08]) & (~a[09]) & (~a[12]) & (~a[13]) & a[15]) | ((~a[01]) & (~a[04]) & (~a[05]) & (~a[08]) & (~a[09]) & (~a[12]) & (~a[13]) & a[14]) | ((~a[01]) & (~a[04]) & (~a[05]) & (~a[08]) & (~a[09]) & a[11]) | ((~a[01]) & (~a[04]) & (~a[05]) & (~a[08]) & (~a[09]) & a[10]) | ((~a[01]) & (~a[04]) & (~a[05]) & a[07]) | ((~a[01]) & (~a[04]) & (~a[05]) & a[06]) | ((~a[01]) & a[03]) | ((~a[01]) & a[02]);

   assign shamt[4] = ((~a[02]) & (~a[04]) & (~a[06]) & (~a[08]) & (~a[10]) & (~a[12]) & (~a[14]) & (~a[16]) & (~a[18]) & a[19]) | ((~a[02]) & (~a[04]) & (~a[06]) & (~a[08]) & (~a[10]) & (~a[12]) & (~a[14]) & (~a[16]) & a[17]) | ((~a[02]) & (~a[04]) & (~a[06]) & (~a[08]) & (~a[10]) & (~a[12]) & (~a[14]) & a[15]) | ((~a[02]) & (~a[04]) & (~a[06]) & (~a[08]) & (~a[10]) & (~a[12]) & a[13]) | ((~a[02]) & (~a[04]) & (~a[06]) & (~a[08]) & (~a[10]) & a[11]) | ((~a[02]) & (~a[04]) & (~a[06]) & (~a[08]) & a[09]) | ((~a[02]) & (~a[04]) & (~a[06]) & a[07]) | ((~a[02]) & (~a[04]) & a[05]) | ((~a[02]) & a[03]) | (a[01]);

endmodule