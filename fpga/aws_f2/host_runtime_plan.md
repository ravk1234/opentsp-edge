# Host Runtime Plan

Host communicates through AXI-Lite registers.

Existing mapping:

0x00 CONTROL
0x08 PROGRAM_LEN
0x10 INSTR_ADDR
0x14 INSTR_WORD
0x18 DATA_ADDR
0x1C DATA_BANK
0x20 DATA_WORD

Readback:

0x30 C00
0x34 C01
0x38 C10
0x3C C11

Workflow:

1. Load instructions
2. Load matrices
3. Start engine
4. Poll status
5. Read outputs