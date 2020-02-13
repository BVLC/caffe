/*
 *  addr_gen.sv -- address generator for multi-bank memories incl. registers
 *  ETRI <SW-SoC AI Deep Learning HW Accelerator RTL Design> course material
 *
 *  first draft by Junyoung Park
 */

`timescale 1ns / 1ps

module addr_gen #(
  parameter integer ADDR_WIDTH = 12,
  parameter integer DATA_WIDTH = 32
  ) (
  // clock and resetn from domain a
  input  wire clk_a,
  input  wire arstz_aq,

  // signals from the external interfaces
  cnnip_mem_if.slave  from_axi4l_mem_if,

  // signals to the internal interfaces
  cnnip_mem_if.master to_register_if,
  cnnip_mem_if.master to_input_mem_if,
  cnnip_mem_if.master to_weight_mem_if,
  cnnip_mem_if.master to_feature_mem_if
  );

  wire to_register;
  wire to_input_mem;
  wire to_weight_mem;
  wire to_feature_mem;

  wire [3:0] select;

  // address multiplexer
  assign to_register    = (from_axi4l_mem_if.addr[11:8] == 4'h0);
  assign to_input_mem   = (from_axi4l_mem_if.addr[11:8] == 4'h1);
  assign to_weight_mem  = (from_axi4l_mem_if.addr[11:8] == 4'h2);
  assign to_feature_mem = (from_axi4l_mem_if.addr[11:8] == 4'h3);

  assign select = { to_feature_mem,
                    to_weight_mem,
                    to_input_mem,
                    to_register };

  assign to_register_if.en   = (to_register) ? from_axi4l_mem_if.en   : 'b0;
  assign to_register_if.we   = (to_register) ? from_axi4l_mem_if.we   : 'b0;
  assign to_register_if.addr = (to_register) ? from_axi4l_mem_if.addr : 'b0;
  assign to_register_if.din  = (to_register) ? from_axi4l_mem_if.din  : 'b0;

  assign to_input_mem_if.en   = (to_input_mem) ? from_axi4l_mem_if.en   : 'b0;
  assign to_input_mem_if.we   = (to_input_mem) ? from_axi4l_mem_if.we   : 'b0;
  assign to_input_mem_if.addr = (to_input_mem) ? from_axi4l_mem_if.addr : 'b0;
  assign to_input_mem_if.din  = (to_input_mem) ? from_axi4l_mem_if.din  : 'b0;

  assign to_weight_mem_if.en   = (to_weight_mem) ? from_axi4l_mem_if.en   : 'b0;
  assign to_weight_mem_if.we   = (to_weight_mem) ? from_axi4l_mem_if.we   : 'b0;
  assign to_weight_mem_if.addr = (to_weight_mem) ? from_axi4l_mem_if.addr : 'b0;
  assign to_weight_mem_if.din  = (to_weight_mem) ? from_axi4l_mem_if.din  : 'b0;

  assign to_feature_mem_if.en   = (to_feature_mem) ? from_axi4l_mem_if.en   : 'b0;
  assign to_feature_mem_if.we   = (to_feature_mem) ? from_axi4l_mem_if.we   : 'b0;
  assign to_feature_mem_if.addr = (to_feature_mem) ? from_axi4l_mem_if.addr : 'b0;
  assign to_feature_mem_if.din  = (to_feature_mem) ? from_axi4l_mem_if.din  : 'b0;

  always_comb
  begin
    // unfortunately, the current vivado does not support unique case syntax
    // unique case(select)
    from_axi4l_mem_if.dout = 0;
    case(select)
              // 0: from_axi4l_mem_if.dout = 0;
      1'b1 << 0: from_axi4l_mem_if.dout = to_register_if.dout;
      1'b1 << 1: from_axi4l_mem_if.dout = to_input_mem_if.dout;
      1'b1 << 2: from_axi4l_mem_if.dout = to_weight_mem_if.dout;
      1'b1 << 3: from_axi4l_mem_if.dout = to_feature_mem_if.dout;
      // {4{1'bx}}: from_axi4l_mem_if.dout = 0;
    endcase
  end

  always_comb
  begin
    // unique case(select)
    from_axi4l_mem_if.valid = 0;
    case(select)
              // 0: from_axi4l_mem_if.valid = 0;
      1'b1 << 0: from_axi4l_mem_if.valid = to_register_if.valid;
      1'b1 << 1: from_axi4l_mem_if.valid = to_input_mem_if.valid;
      1'b1 << 2: from_axi4l_mem_if.valid = to_weight_mem_if.valid;
      1'b1 << 3: from_axi4l_mem_if.valid = to_feature_mem_if.valid;
      // {4{1'bx}}: from_axi4l_mem_if.valid = 0;
    endcase
  end

endmodule
