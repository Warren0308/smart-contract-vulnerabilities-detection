PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0082<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x06fdde03<br>DUP2<br>EQ<br>PUSH2 0x0084<br>JUMPI<br>DUP1<br>PUSH4 0x2986c0e5<br>EQ<br>PUSH2 0x010e<br>JUMPI<br>DUP1<br>PUSH4 0x47c05c22<br>EQ<br>PUSH2 0x0135<br>JUMPI<br>DUP1<br>PUSH4 0x5fd8c710<br>EQ<br>PUSH2 0x0155<br>JUMPI<br>DUP1<br>PUSH4 0x6b88f4ae<br>EQ<br>PUSH2 0x016a<br>JUMPI<br>DUP1<br>PUSH4 0x95d89b41<br>EQ<br>PUSH2 0x0182<br>JUMPI<br>DUP1<br>PUSH4 0xf2fde38b<br>EQ<br>PUSH2 0x0197<br>JUMPI<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0090<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0099<br>PUSH2 0x01b8<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH1 0x20<br>DUP1<br>DUP3<br>MSTORE<br>DUP4<br>MLOAD<br>DUP2<br>DUP4<br>ADD<br>MSTORE<br>DUP4<br>MLOAD<br>SWAP2<br>SWAP3<br>DUP4<br>SWAP3<br>SWAP1<br>DUP4<br>ADD<br>SWAP2<br>DUP6<br>ADD<br>SWAP1<br>DUP1<br>DUP4<br>DUP4<br>PUSH1 0x00<br>JUMPDEST<br>DUP4<br>DUP2<br>LT<br>ISZERO<br>PUSH2 0x00d3<br>JUMPI<br>DUP2<br>DUP2<br>ADD<br>MLOAD<br>DUP4<br>DUP3<br>ADD<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH2 0x00bb<br>JUMP<br>JUMPDEST<br>POP<br>POP<br>POP<br>POP<br>SWAP1<br>POP<br>SWAP1<br>DUP2<br>ADD<br>SWAP1<br>PUSH1 0x1f<br>AND<br>DUP1<br>ISZERO<br>PUSH2 0x0100<br>JUMPI<br>DUP1<br>DUP3<br>SUB<br>DUP1<br>MLOAD<br>PUSH1 0x01<br>DUP4<br>PUSH1 0x20<br>SUB<br>PUSH2 0x0100<br>EXP<br>SUB<br>NOT<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>JUMPDEST<br>POP<br>SWAP3<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x011a<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0123<br>PUSH2 0x01ef<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>SWAP2<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0141<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0082<br>PUSH1 0x04<br>DUP1<br>CALLDATALOAD<br>PUSH1 0x24<br>DUP2<br>ADD<br>SWAP2<br>ADD<br>CALLDATALOAD<br>PUSH2 0x01f5<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0161<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0082<br>PUSH2 0x02ff<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0176<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0082<br>PUSH1 0x04<br>CALLDATALOAD<br>PUSH2 0x0354<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x018e<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0099<br>PUSH2 0x0370<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x01a3<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0082<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x03a7<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>DUP1<br>DUP3<br>ADD<br>SWAP1<br>SWAP2<br>MSTORE<br>PUSH1 0x1c<br>DUP2<br>MSTORE<br>PUSH32 0xe286932053656520436f6465204f662054686520436f6e747261637400000000<br>PUSH1 0x20<br>DUP3<br>ADD<br>MSTORE<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>CALLER<br>EQ<br>PUSH2 0x020d<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH3 0x0f4240<br>EQ<br>ISZERO<br>PUSH2 0x021f<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH1 0x01<br>SLOAD<br>JUMPDEST<br>DUP2<br>DUP2<br>LT<br>ISZERO<br>PUSH2 0x02fa<br>JUMPI<br>DUP3<br>DUP3<br>DUP3<br>DUP2<br>DUP2<br>LT<br>PUSH2 0x0238<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x20<br>SWAP1<br>SWAP3<br>MUL<br>SWAP4<br>SWAP1<br>SWAP4<br>ADD<br>CALLDATALOAD<br>AND<br>SWAP2<br>PUSH1 0x00<br>SWAP2<br>POP<br>PUSH2 0x0309<br>SWAP1<br>DUP3<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP4<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>POP<br>DUP3<br>DUP3<br>DUP3<br>DUP2<br>DUP2<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x0273<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>SWAP1<br>POP<br>PUSH1 0x20<br>MUL<br>ADD<br>CALLDATALOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH1 0x00<br>PUSH32 0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef<br>PUSH2 0x0309<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>LOG3<br>PUSH1 0x00<br>NOT<br>DUP3<br>ADD<br>DUP2<br>EQ<br>ISZERO<br>PUSH2 0x02df<br>JUMPI<br>PUSH3 0x0f4240<br>PUSH1 0x01<br>SSTORE<br>PUSH2 0x02fa<br>JUMP<br>JUMPDEST<br>PUSH2 0xc350<br>GAS<br>GT<br>PUSH2 0x02f2<br>JUMPI<br>PUSH1 0x01<br>DUP2<br>SWAP1<br>SSTORE<br>PUSH2 0x02fa<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>ADD<br>PUSH2 0x0224<br>JUMP<br>JUMPDEST<br>POP<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>CALLER<br>EQ<br>PUSH2 0x0316<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>SWAP2<br>AND<br>SWAP2<br>ADDRESS<br>BALANCE<br>DUP1<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP3<br>SWAP1<br>SWAP2<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x0351<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>CALLER<br>EQ<br>PUSH2 0x036b<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>SSTORE<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>DUP1<br>DUP3<br>ADD<br>SWAP1<br>SWAP2<br>MSTORE<br>PUSH1 0x08<br>DUP2<br>MSTORE<br>PUSH32 0x436f646520e29c93000000000000000000000000000000000000000000000000<br>PUSH1 0x20<br>DUP3<br>ADD<br>MSTORE<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>CALLER<br>EQ<br>PUSH2 0x03be<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>NOT<br>AND<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP3<br>SWAP1<br>SWAP3<br>AND<br>SWAP2<br>SWAP1<br>SWAP2<br>OR<br>SWAP1<br>SSTORE<br>JUMP<br>STOP<br>