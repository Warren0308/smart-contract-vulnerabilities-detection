PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x006d<br>JUMPI<br>PUSH1 0x00<br>CALLDATALOAD<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>SWAP1<br>DIV<br>PUSH4 0xffffffff<br>AND<br>DUP1<br>PUSH4 0x110ba0f9<br>EQ<br>PUSH2 0x0072<br>JUMPI<br>DUP1<br>PUSH4 0x8da5cb5b<br>EQ<br>PUSH2 0x00db<br>JUMPI<br>DUP1<br>PUSH4 0xb863bd37<br>EQ<br>PUSH2 0x0132<br>JUMPI<br>DUP1<br>PUSH4 0xc8eab537<br>EQ<br>PUSH2 0x0173<br>JUMPI<br>DUP1<br>PUSH4 0xe72f5f63<br>EQ<br>PUSH2 0x01c4<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x007e<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00a3<br>PUSH1 0x04<br>DUP1<br>CALLDATASIZE<br>SUB<br>DUP2<br>ADD<br>SWAP1<br>DUP1<br>DUP1<br>CALLDATALOAD<br>PUSH4 0xffffffff<br>AND<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP3<br>SWAP2<br>SWAP1<br>POP<br>POP<br>POP<br>PUSH2 0x01ef<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP5<br>PUSH4 0xffffffff<br>AND<br>PUSH4 0xffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP4<br>PUSH1 0x00<br>NOT<br>AND<br>PUSH1 0x00<br>NOT<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x00e7<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00f0<br>PUSH2 0x0229<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x013e<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x015d<br>PUSH1 0x04<br>DUP1<br>CALLDATASIZE<br>SUB<br>DUP2<br>ADD<br>SWAP1<br>DUP1<br>DUP1<br>CALLDATALOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP3<br>SWAP2<br>SWAP1<br>POP<br>POP<br>POP<br>PUSH2 0x024f<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x017f<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x01c2<br>PUSH1 0x04<br>DUP1<br>CALLDATASIZE<br>SUB<br>DUP2<br>ADD<br>SWAP1<br>DUP1<br>DUP1<br>CALLDATALOAD<br>PUSH4 0xffffffff<br>AND<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP3<br>SWAP2<br>SWAP1<br>DUP1<br>CALLDATALOAD<br>PUSH1 0x00<br>NOT<br>AND<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP3<br>SWAP2<br>SWAP1<br>DUP1<br>CALLDATALOAD<br>PUSH4 0xffffffff<br>AND<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP3<br>SWAP2<br>SWAP1<br>POP<br>POP<br>POP<br>PUSH2 0x026b<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x01d0<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x01d9<br>PUSH2 0x03c9<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0x20<br>MSTORE<br>DUP1<br>PUSH1 0x00<br>MSTORE<br>PUSH1 0x40<br>PUSH1 0x00<br>SHA3<br>PUSH1 0x00<br>SWAP2<br>POP<br>SWAP1<br>POP<br>DUP1<br>PUSH1 0x00<br>ADD<br>PUSH1 0x00<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH4 0xffffffff<br>AND<br>SWAP1<br>DUP1<br>PUSH1 0x01<br>ADD<br>SLOAD<br>SWAP1<br>DUP1<br>PUSH1 0x02<br>ADD<br>SLOAD<br>SWAP1<br>POP<br>DUP4<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>PUSH1 0x00<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP2<br>PUSH2 0x025a<br>PUSH2 0x03c9<br>JUMP<br>JUMPDEST<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x0263<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>MOD<br>SWAP1<br>POP<br>SWAP2<br>SWAP1<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>PUSH1 0x02<br>PUSH1 0x00<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>CALLER<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>EQ<br>ISZERO<br>ISZERO<br>PUSH2 0x02c9<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x00<br>DUP3<br>PUSH4 0xffffffff<br>AND<br>GT<br>ISZERO<br>ISZERO<br>PUSH2 0x02de<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x02ed<br>DUP3<br>PUSH4 0xffffffff<br>AND<br>PUSH2 0x024f<br>JUMP<br>JUMPDEST<br>SWAP1<br>POP<br>PUSH1 0x60<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>ADD<br>PUSH1 0x40<br>MSTORE<br>DUP1<br>DUP4<br>PUSH4 0xffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP5<br>PUSH1 0x00<br>NOT<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP3<br>DUP2<br>MSTORE<br>POP<br>PUSH1 0x01<br>PUSH1 0x00<br>DUP7<br>PUSH4 0xffffffff<br>AND<br>PUSH4 0xffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x00<br>SHA3<br>PUSH1 0x00<br>DUP3<br>ADD<br>MLOAD<br>DUP2<br>PUSH1 0x00<br>ADD<br>PUSH1 0x00<br>PUSH2 0x0100<br>EXP<br>DUP2<br>SLOAD<br>DUP2<br>PUSH4 0xffffffff<br>MUL<br>NOT<br>AND<br>SWAP1<br>DUP4<br>PUSH4 0xffffffff<br>AND<br>MUL<br>OR<br>SWAP1<br>SSTORE<br>POP<br>PUSH1 0x20<br>DUP3<br>ADD<br>MLOAD<br>DUP2<br>PUSH1 0x01<br>ADD<br>SWAP1<br>PUSH1 0x00<br>NOT<br>AND<br>SWAP1<br>SSTORE<br>PUSH1 0x40<br>DUP3<br>ADD<br>MLOAD<br>DUP2<br>PUSH1 0x02<br>ADD<br>SSTORE<br>SWAP1<br>POP<br>POP<br>PUSH32 0x97f6735001b531af63662eaadac3df47a2bb56bb7079ffcfd56f5875f6e4cfee<br>DUP5<br>DUP3<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>PUSH4 0xffffffff<br>AND<br>PUSH4 0xffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP3<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>LOG1<br>POP<br>POP<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH1 0x01<br>NUMBER<br>SUB<br>BLOCKHASH<br>COINBASE<br>DIFFICULTY<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP6<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP5<br>PUSH1 0x00<br>NOT<br>AND<br>PUSH1 0x00<br>NOT<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP4<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH13 0x01000000000000000000000000<br>MUL<br>DUP2<br>MSTORE<br>PUSH1 0x14<br>ADD<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP5<br>POP<br>POP<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>SHA3<br>PUSH1 0x01<br>SWAP1<br>DIV<br>PUSH1 0x00<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>PUSH1 0x00<br>SLOAD<br>SWAP1<br>POP<br>SWAP1<br>JUMP<br>STOP<br>