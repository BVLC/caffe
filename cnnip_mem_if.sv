/*
 *  cnnip_mem_if.v -- block memory with parameterized read latency
 *  ETRI <SW-SoC AI Deep Learning HW Accelerator RTL Design> course material
 *
 *  first draft by Junyoung Park
 */
`timescale 1ns / 1ps

interface cnnip_mem_if #(
  parameter ADDR_WIDTH = 12,
  parameter DATA_WIDTH = 32
);
  logic                         en;
  logic [((DATA_WIDTH-1)>>3):0] we;
  logic [ADDR_WIDTH-1:0]        addr;
  logic [DATA_WIDTH-1:0]        din;
  logic [DATA_WIDTH-1:0]        dout;
  logic                         valid;

  modport slave (
    input en,
    input we,
    input addr,
    input din,
    output dout,
    output valid
  );

  modport master (
    output en,
    output we,
    output addr,
    output din,
    input dout,
    input valid
  );
endinterface
