PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0061<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x3f7e2120<br>DUP2<br>EQ<br>PUSH2 0x0063<br>JUMPI<br>DUP1<br>PUSH4 0x80f8ea60<br>EQ<br>PUSH2 0x0076<br>JUMPI<br>DUP1<br>PUSH4 0xbea948c8<br>EQ<br>PUSH2 0x008a<br>JUMPI<br>DUP1<br>PUSH4 0xdac47a71<br>EQ<br>PUSH2 0x0092<br>JUMPI<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x006e<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0061<br>PUSH2 0x00a8<br>JUMP<br>JUMPDEST<br>PUSH2 0x0061<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x010d<br>JUMP<br>JUMPDEST<br>PUSH2 0x0061<br>PUSH2 0x019d<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x009d<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0061<br>PUSH1 0x04<br>CALLDATALOAD<br>PUSH2 0x0200<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>DUP1<br>ISZERO<br>PUSH2 0x00d0<br>JUMPI<br>POP<br>PUSH1 0x01<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>ISZERO<br>ISZERO<br>JUMPDEST<br>ISZERO<br>PUSH2 0x010b<br>JUMPI<br>PUSH1 0x01<br>DUP1<br>SLOAD<br>PUSH21 0xff0000000000000000000000000000000000000000<br>NOT<br>AND<br>PUSH21 0x010000000000000000000000000000000000000000<br>OR<br>SWAP1<br>SSTORE<br>JUMPDEST<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH21 0x010000000000000000000000000000000000000000<br>SWAP1<br>DIV<br>PUSH1 0xff<br>AND<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x013f<br>JUMPI<br>POP<br>PUSH8 0x0de0b6b3a7640000<br>CALLVALUE<br>GT<br>JUMPDEST<br>DUP1<br>PUSH2 0x0153<br>JUMPI<br>POP<br>PUSH1 0x00<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>ISZERO<br>JUMPDEST<br>ISZERO<br>PUSH2 0x019a<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>CALLER<br>DUP2<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>NOT<br>SWAP3<br>DUP4<br>AND<br>OR<br>SWAP1<br>SWAP3<br>SSTORE<br>PUSH1 0x01<br>DUP1<br>SLOAD<br>SWAP3<br>DUP5<br>AND<br>SWAP3<br>SWAP1<br>SWAP2<br>AND<br>SWAP2<br>SWAP1<br>SWAP2<br>OR<br>SWAP1<br>SSTORE<br>TIMESTAMP<br>PUSH1 0x02<br>SSTORE<br>JUMPDEST<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>SLOAD<br>TIMESTAMP<br>GT<br>ISZERO<br>PUSH2 0x01fb<br>JUMPI<br>PUSH1 0x01<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>ISZERO<br>PUSH2 0x01fb<br>JUMPI<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH2 0x08fc<br>ADDRESS<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>BALANCE<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x01fb<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x010b<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>SLOAD<br>TIMESTAMP<br>GT<br>ISZERO<br>PUSH2 0x0227<br>JUMPI<br>PUSH1 0x00<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>ISZERO<br>PUSH2 0x0227<br>JUMPI<br>PUSH1 0x02<br>DUP2<br>SWAP1<br>SSTORE<br>JUMPDEST<br>PUSH2 0x019a<br>JUMP<br>STOP<br>