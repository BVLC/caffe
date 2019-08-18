/*
 *  register_set.sv -- register set file
 *  ETRI <SW-SoC AI Deep Learning HW Accelerator RTL Design> course material
 *
 *  first draft by Junyoung Park
 */

`timescale 1 ns / 1 ps
`include "cnnip_mem_if.sv"

module cnnip_v1_0 #
(
  // Users to add parameters here

  // User parameters ends
  // Do not modify the parameters beyond this line


  // Parameters of Axi Slave Bus Interface S00_AXI
  parameter integer C_S00_AXI_DATA_WIDTH	= 32,
  parameter integer C_S00_AXI_ADDR_WIDTH	= 16
)
(
  // Users to add ports here

  // User ports ends
  // Do not modify the ports beyond this line


  // Ports of Axi Slave Bus Interface S00_AXI
  input wire  s00_axi_aclk,
  input wire  s00_axi_aresetn,
  input wire [C_S00_AXI_ADDR_WIDTH-1 : 0] s00_axi_awaddr,
  input wire [2 : 0] s00_axi_awprot,
  input wire  s00_axi_awvalid,
  output wire  s00_axi_awready,
  input wire [C_S00_AXI_DATA_WIDTH-1 : 0] s00_axi_wdata,
  input wire [(C_S00_AXI_DATA_WIDTH/8)-1 : 0] s00_axi_wstrb,
  input wire  s00_axi_wvalid,
  output wire  s00_axi_wready,
  output wire [1 : 0] s00_axi_bresp,
  output wire  s00_axi_bvalid,
  input wire  s00_axi_bready,
  input wire [C_S00_AXI_ADDR_WIDTH-1 : 0] s00_axi_araddr,
  input wire [2 : 0] s00_axi_arprot,
  input wire  s00_axi_arvalid,
  output wire  s00_axi_arready,
  output wire [C_S00_AXI_DATA_WIDTH-1 : 0] s00_axi_rdata,
  output wire [1 : 0] s00_axi_rresp,
  output wire  s00_axi_rvalid,
  input wire  s00_axi_rready
);


  // system clock and reset from AXI4-Lite
  wire clk_a;
  wire arstz_aq;

  assign clk_a    = s00_axi_aclk;
  assign arstz_aq = s00_axi_aresetn;

  // interfaces
  cnnip_mem_if reg_set_interconnect();

  cnnip_mem_if input_mem_ext();
  cnnip_mem_if weight_mem_ext();
  cnnip_mem_if feature_mem_ext();

  cnnip_mem_if input_mem_int();
  cnnip_mem_if weight_mem_int();
  cnnip_mem_if feature_mem_int();

  cnnip_mem_if axi4l_int();

  // configuration signals
  wire 		 	 CMD_DONE;
  wire 			 CMD_DONE_VALID;
  wire 			 CMD_START;
  wire [7:0] MODE_KERNEL_SIZE;
  wire [7:0] MODE_KERNEL_NUMS;
  wire [1:0] MODE_STRIDE;
  wire 			 MODE_PADDING;

// Instantiation of Axi Bus Interface S00_AXI
  cnnip_v1_0_S00_AXI # (
    .C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
    .C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
  ) cnnip_v1_0_S00_AXI_inst (
    .S_AXI_ACLK(s00_axi_aclk),
    .S_AXI_ARESETN(s00_axi_aresetn),
    .S_AXI_AWADDR(s00_axi_awaddr),
    .S_AXI_AWPROT(s00_axi_awprot),
    .S_AXI_AWVALID(s00_axi_awvalid),
    .S_AXI_AWREADY(s00_axi_awready),
    .S_AXI_WDATA(s00_axi_wdata),
    .S_AXI_WSTRB(s00_axi_wstrb),
    .S_AXI_WVALID(s00_axi_wvalid),
    .S_AXI_WREADY(s00_axi_wready),
    .S_AXI_BRESP(s00_axi_bresp),
    .S_AXI_BVALID(s00_axi_bvalid),
    .S_AXI_BREADY(s00_axi_bready),
    .S_AXI_ARADDR(s00_axi_araddr),
    .S_AXI_ARPROT(s00_axi_arprot),
    .S_AXI_ARVALID(s00_axi_arvalid),
    .S_AXI_ARREADY(s00_axi_arready),
    .S_AXI_RDATA(s00_axi_rdata),
    .S_AXI_RRESP(s00_axi_rresp),
    .S_AXI_RVALID(s00_axi_rvalid),
    .S_AXI_RREADY(s00_axi_rready),
    .int_mem_if(axi4l_int)
  );

  addr_gen #(
    .ADDR_WIDTH(12),
    .DATA_WIDTH(32)
  ) i_addr_gen (
    .clk_a     (clk_a),
    .arstz_aq  (arstz_aq),

    // signals from the external interfaces
    .from_axi4l_mem_if(axi4l_int),

    // signals to the internal interfaces
    .to_register_if(reg_set_interconnect),
    .to_input_mem_if(input_mem_ext),
    .to_weight_mem_if(weight_mem_ext),
    .to_feature_mem_if(feature_mem_ext)
  );

  register_set i_register_set (
    .clk_a     (clk_a),
    .arstz_aq  (arstz_aq),
    .mem_if    (reg_set_interconnect),

    .MODE_KERNEL_SIZE(MODE_KERNEL_SIZE),
    .MODE_KERNEL_NUMS(MODE_KERNEL_NUMS),
    .MODE_STRIDE(MODE_STRIDE),
    .MODE_PADDING(MODE_PADDING),
    .CMD_START(CMD_START),
    
    .CMD_DONE(CMD_DONE),
    .CMD_DONE_VALID(CMD_DONE_VALID)
  );

  // true dual-port memories
  blk_mem_wrapper #(
    .READ_LATENCY(2),
    .ADDR_WIDTH(8)
  ) input_mem (
    .clk_a(clk_a),
    .arstz_aq(arstz_aq),
    .mem_if_a(input_mem_ext),

    .clk_b(clk_a),
    .arstz_bq(arstz_aq),
    .mem_if_b(input_mem_int)
  );

  blk_mem_wrapper #(
    .READ_LATENCY(2),
    .ADDR_WIDTH(8)
  ) weight_memory (
    .clk_a(clk_a),
    .arstz_aq(arstz_aq),
    .mem_if_a(weight_mem_ext),

    .clk_b(clk_a),
    .arstz_bq(arstz_aq),
    .mem_if_b(weight_mem_int)
  );

  blk_mem_wrapper #(
    .READ_LATENCY(2),
    .ADDR_WIDTH(8)
  ) feature_memory (
    .clk_a(clk_a),
    .arstz_aq(arstz_aq),
    .mem_if_a(feature_mem_ext),

    .clk_b(clk_a),
    .arstz_bq(arstz_aq),
    .mem_if_b(feature_mem_int)
  );

  cnnip_ctrlr i_cnnip_ctrlr
  (
    .clk_a(clk_a),
    .arstz_aq(arstz_aq),

    .to_input_mem(input_mem_int),
    .to_weight_mem(weight_mem_int),
    .to_feature_mem(feature_mem_int),

    .CMD_START(CMD_START),
    .MODE_KERNEL_SIZE(MODE_KERNEL_SIZE),
    .MODE_KERNEL_NUMS(MODE_KERNEL_NUMS),
    .MODE_STRIDE(MODE_STRIDE),
    .MODE_PADDING(MODE_PADDING),

    .CMD_DONE(CMD_DONE),
    .CMD_DONE_VALID(CMD_DONE_VALID)
  );

  // User logic ends

endmodule
