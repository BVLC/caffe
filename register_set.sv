/*
 *  register_set.sv -- register set file
 *  ETRI <SW-SoC AI Deep Learning HW Accelerator RTL Design> course material
 *
 *  first draft by Junyoung Park
 */

`timescale 1ns / 1ps

module register_set
(
  // clock and resetn from domain a
  input  wire         clk_a,
  input  wire         arstz_aq,

  // memory mapped register interface
  cnnip_mem_if.slave  mem_if,

  // controller register interface ---------------------------------------------
  input  wire         CMD_DONE,
  input  wire         CMD_DONE_VALID,

  output wire         CMD_START,
  output wire   [7:0] MODE_KERNEL_SIZE,
  output wire   [7:0] MODE_KERNEL_NUMS,
  output wire   [1:0] MODE_STRIDE,
  output wire         MODE_PADDING
  // ---------------------------------------------------------------------------
  );

  // ---------------------------------------------------------------------------
  // slv_reg[0] |                            [31:0]                            |
  //            |                           CMD_START                          |
  // ---------------------------------------------------------------------------
  // slv_reg[1] |                            [31:0]                            |
  //            |                           CMD_DONE                           |
  // ---------------------------------------------------------------------------
  // slv_reg[2] |   [31:24]   |      [23:16]     |  [15:8]  |       [7:0]      |
  //            |   reserved  | MODE_KERNEL_SIZE | reserved | MODE_KERNEL_NUMS |
  // ---------------------------------------------------------------------------
  // slv_reg[3] |    [31:17]    |      [16]    |    [15:2]    |     [1:0]      |
  //            |    reserved   | MODE_PADDING |   reserved   |  MODE_STRIDE   |
  // ---------------------------------------------------------------------------
  reg [31:0] slv_reg[3:0];

  // memory-mapped read
  wire [1:0] data_addr = mem_if.addr[3:2];
  assign mem_if.dout   = slv_reg[data_addr];
  assign mem_if.valid  = (mem_if.en == 1'b1) && (|mem_if.we == 1'b0);

  // configuration read
  assign CMD_START        = &slv_reg[0][31:0];
  assign MODE_KERNEL_NUMS =  slv_reg[2][7:0];
  assign MODE_KERNEL_SIZE =  slv_reg[2][23:16];
  assign MODE_STRIDE      =  slv_reg[3][1:0];
  assign MODE_PADDING     =  slv_reg[3][16];

  // disable byte-addressable write & hard-coded!
  always @(posedge clk_a, negedge arstz_aq)
    if (arstz_aq == 1'b0)               slv_reg[0] <= 32'b0;
    else if ( (mem_if.en  == 1'b1) &&
              (|mem_if.we == 1'b1) &&
              (data_addr  == 2'b00))    slv_reg[0] <= mem_if.din;

  always @(posedge clk_a, negedge arstz_aq)
    if (arstz_aq == 1'b0)               slv_reg[1] <= 32'b0;
    else if (CMD_DONE_VALID)            slv_reg[1] <= {32{CMD_DONE}};
    else if ( (mem_if.en  == 1'b1) &&
              (|mem_if.we == 1'b1) &&
              (data_addr  == 2'b00))    slv_reg[1] <= mem_if.din;

  always @(posedge clk_a, negedge arstz_aq)
    if (arstz_aq == 1'b0)               slv_reg[2] <= 32'b0;
    else if ( (mem_if.en  == 1'b1) &&
              (|mem_if.we == 1'b1) &&
              (data_addr  == 2'b10))    slv_reg[2] <= mem_if.din;

  always @(posedge clk_a, negedge arstz_aq)
    if (arstz_aq == 1'b0)               slv_reg[3] <= 32'b0;
    else if ( (mem_if.en  == 1'b1) &&
              (|mem_if.we == 1'b1) &&
              (data_addr  == 2'b11))    slv_reg[3] <= mem_if.din;

endmodule
