/*
 *  blk_mem_wrapper.v -- true 2-port block ram with parameterized read latency
 *  ETRI <SW-SoC AI Deep Learning HW Accelerator RTL Design> course material
 *
 *  first draft by Junyoung Park
 */
`timescale 1ns / 1ps

module blk_mem_wrapper #(
  parameter ADDR_WIDTH   = 12,
  // parameter DATA_WIDTH   = 31,
  parameter READ_LATENCY = 3
  )(
  // clock and resetn from domain a
  input wire                   clk_a,
  input wire                   arstz_aq,

  // block memory control signals for port a
  cnnip_mem_if.slave           mem_if_a,

  // clock and resetn from domain b
  input wire                   clk_b,
  input wire                   arstz_bq,
  // block memory control signals for port b
  cnnip_mem_if.slave           mem_if_b
  );

  // FSM for read operation -- WRITE is omitted
  enum { IDLE, WAIT, READ } state_aq, next_state_a, state_bq, next_state_b;

  // read control for port a ---------------------------------------------------
  wire read_request_a;
  wire read_wait_done_a;

  reg [1:0] wait_counter_a;
  reg [1:0] wait_counter_next_a;

  reg en_to_blkmem_a;
  reg valid_to_ext_a;

  // transition condition calculation
  assign read_request_a   = (mem_if_a.en == 1'b1 && mem_if_a.we == 1'b0);
  assign read_wait_done_a = (wait_counter_a == READ_LATENCY-1);

  // state transition
  always @(posedge clk_a, negedge arstz_aq)
    if (arstz_aq == 1'b0) state_aq <= IDLE;
    else state_aq <= next_state_a;

  always @(*)
  begin
    next_state_a = state_aq;
    case (state_aq)
      IDLE:
      begin
        if (read_request_a)
        begin
          if (READ_LATENCY == 1) next_state_a = READ;
          else                   next_state_a = WAIT;
        end
      end

      WAIT:
      begin
        if (read_wait_done_a) next_state_a = READ;
      end

      READ:
      begin
        next_state_a = IDLE;
      end
    endcase
  end

  // Clock count for WAIT state
  always @(posedge clk_a, negedge arstz_aq)
    if (arstz_aq == 1'b0) wait_counter_a <= 0;
    else wait_counter_a <= wait_counter_next_a;

  always @(*)
  begin
    wait_counter_next_a = wait_counter_a;
    case (state_aq)
      IDLE: wait_counter_next_a = 2'b1;
      WAIT: wait_counter_next_a = wait_counter_a + 1'b1;
      default: wait_counter_next_a = 0;
    endcase
  end

  // output signals
  always @(*)
  begin
    en_to_blkmem_a = mem_if_a.en;
    valid_to_ext_a = 0;

    case (state_aq)
      IDLE:
      begin
        en_to_blkmem_a = mem_if_a.en;
        valid_to_ext_a = 0;
      end

      WAIT:
      begin
        en_to_blkmem_a = 1;
        valid_to_ext_a = 0;
      end

      READ:
      begin
        en_to_blkmem_a = 0;
        valid_to_ext_a = 1;
      end
    endcase
  end

  assign mem_if_a.valid = valid_to_ext_a;

  // read control for port b ---------------------------------------------------
  wire read_request_b;
  wire read_wait_done_b;

  reg [1:0] wait_counter_b;
  reg [1:0] wait_counter_next_b;

  reg en_to_blkmem_b;
  reg valid_to_ext_b;

  // transition condition calculation
  assign read_request_b   = (mem_if_b.en == 1'b1 && mem_if_b.we == 1'b0);
  assign read_wait_done_b = (wait_counter_b == READ_LATENCY-1);

  always @(posedge clk_b, negedge arstz_bq)
    if (arstz_bq == 1'b0) state_bq <= IDLE;
    else state_bq <= next_state_b;

  always @(*)
  begin
    next_state_b = state_bq;
    case (state_bq)
      IDLE:
      begin
        if (read_request_b)
        begin
          if (READ_LATENCY == 1) next_state_b = READ;
          else                   next_state_b = WAIT;
        end
      end

      WAIT:
      begin
        if (read_wait_done_b) next_state_b = READ;
      end

      READ:
      begin
        next_state_b = IDLE;
      end
    endcase
  end

  // Clock count for WAIT state
  always @(posedge clk_b, negedge arstz_bq)
    if (arstz_bq == 1'b0) wait_counter_b <= 0;
    else wait_counter_b <= wait_counter_next_b;

  always @(*)
  begin
    wait_counter_next_b = wait_counter_b;
    case (state_bq)
      IDLE: wait_counter_next_b = 2'b1;
      WAIT: wait_counter_next_b = wait_counter_b + 1'b1;
      default: wait_counter_next_b = 0;
    endcase
  end

  // output signals
  always @(*)
  begin
    en_to_blkmem_b = mem_if_b.en;
    valid_to_ext_b = 0;

    case (state_bq)
      IDLE:
      begin
        en_to_blkmem_b = mem_if_b.en;
        valid_to_ext_b = 0;
      end

      WAIT:
      begin
        en_to_blkmem_b = 1;
        valid_to_ext_b = 0;
      end

      READ:
      begin
        en_to_blkmem_b = 0;
        valid_to_ext_b = 1;
      end
    endcase
  end

  assign mem_if_b.valid = valid_to_ext_b;

  // native block memory connections -------------------------------------------
  blk_mem_gen_0 i_blk_mem (
    .clka(clk_a),
    .ena(en_to_blkmem_a),
    .wea(mem_if_a.we),
    .addra(mem_if_a.addr[ADDR_WIDTH-1:2]),
    .dina(mem_if_a.din),
    .douta(mem_if_a.dout),

    .clkb(clk_b),
    .enb(en_to_blkmem_b),
    .web(mem_if_b.we),
    .addrb(mem_if_b.addr[ADDR_WIDTH-1:2]),
    .dinb(mem_if_b.din),
    .doutb(mem_if_b.dout)
  );
  // ---------------------------------------------------------------------------

endmodule
