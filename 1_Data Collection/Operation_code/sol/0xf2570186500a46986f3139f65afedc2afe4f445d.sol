PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>PUSH4 0xffffffff<br>PUSH1 0xe0<br>PUSH1 0x02<br>EXP<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0xb535c741<br>DUP2<br>EQ<br>PUSH1 0x23<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH1 0x2d<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x3c<br>PUSH4 0xffffffff<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH1 0x3e<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH1 0x00<br>JUMPDEST<br>DUP2<br>PUSH4 0xffffffff<br>AND<br>DUP2<br>PUSH4 0xffffffff<br>AND<br>LT<br>ISZERO<br>PUSH1 0xd8<br>JUMPI<br>PUSH1 0x00<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH1 0x40<br>MLOAD<br>PUSH32 0x67697665426c6f636b5265776172642829000000000000000000000000000000<br>DUP2<br>MSTORE<br>PUSH1 0x11<br>ADD<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>SHA3<br>PUSH1 0xe0<br>PUSH1 0x02<br>EXP<br>SWAP1<br>DIV<br>PUSH1 0x40<br>MLOAD<br>DUP2<br>PUSH4 0xffffffff<br>AND<br>PUSH1 0xe0<br>PUSH1 0x02<br>EXP<br>MUL<br>DUP2<br>MSTORE<br>PUSH1 0x04<br>ADD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>PUSH1 0x00<br>DUP8<br>PUSH2 0x646e<br>GAS<br>SUB<br>CALL<br>POP<br>POP<br>POP<br>POP<br>JUMPDEST<br>PUSH1 0x01<br>ADD<br>PUSH1 0x41<br>JUMP<br>JUMPDEST<br>JUMPDEST<br>POP<br>POP<br>JUMP<br>STOP<br>