PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0056<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x8aac1362<br>DUP2<br>EQ<br>PUSH2 0x0201<br>JUMPI<br>DUP1<br>PUSH4 0xe3d670d7<br>EQ<br>PUSH2 0x0241<br>JUMPI<br>DUP1<br>PUSH4 0xff9b3acf<br>EQ<br>PUSH2 0x026f<br>JUMPI<br>JUMPDEST<br>CALLER<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>DUP2<br>SWAP1<br>MSTORE<br>PUSH1 0x40<br>DUP2<br>SHA3<br>SLOAD<br>DUP2<br>SWAP1<br>DUP2<br>SWAP1<br>ISZERO<br>PUSH2 0x01dc<br>JUMPI<br>CALLER<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x01<br>PUSH1 0x20<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP4<br>SHA3<br>SLOAD<br>SWAP2<br>DUP4<br>SWAP1<br>MSTORE<br>SWAP1<br>SWAP2<br>SHA3<br>SLOAD<br>PUSH2 0x170c<br>SWAP2<br>NUMBER<br>SUB<br>SWAP1<br>PUSH1 0x64<br>SWAP1<br>PUSH1 0x05<br>MUL<br>DIV<br>MUL<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x00a2<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>TIMESTAMP<br>BLOCKHASH<br>PUSH1 0x20<br>DUP1<br>DUP4<br>ADD<br>SWAP2<br>SWAP1<br>SWAP2<br>MSTORE<br>DIFFICULTY<br>DUP3<br>DUP5<br>ADD<br>MSTORE<br>PUSH13 0x01000000000000000000000000<br>COINBASE<br>MUL<br>PUSH1 0x60<br>DUP4<br>ADD<br>MSTORE<br>DUP3<br>MLOAD<br>PUSH1 0x54<br>DUP2<br>DUP5<br>SUB<br>ADD<br>DUP2<br>MSTORE<br>PUSH1 0x74<br>SWAP1<br>SWAP3<br>ADD<br>SWAP3<br>DUP4<br>SWAP1<br>MSTORE<br>DUP2<br>MLOAD<br>SWAP5<br>SWAP1<br>SWAP4<br>DIV<br>SWAP7<br>POP<br>PUSH1 0x02<br>SWAP4<br>SWAP1<br>SWAP3<br>DUP3<br>SWAP2<br>SWAP1<br>DUP5<br>ADD<br>SWAP1<br>DUP1<br>DUP4<br>DUP4<br>JUMPDEST<br>PUSH1 0x20<br>DUP4<br>LT<br>PUSH2 0x0114<br>JUMPI<br>DUP1<br>MLOAD<br>DUP3<br>MSTORE<br>PUSH1 0x1f<br>NOT<br>SWAP1<br>SWAP3<br>ADD<br>SWAP2<br>PUSH1 0x20<br>SWAP2<br>DUP3<br>ADD<br>SWAP2<br>ADD<br>PUSH2 0x00f5<br>JUMP<br>JUMPDEST<br>MLOAD<br>DUP2<br>MLOAD<br>PUSH1 0x20<br>SWAP4<br>SWAP1<br>SWAP4<br>SUB<br>PUSH2 0x0100<br>EXP<br>PUSH1 0x00<br>NOT<br>ADD<br>DUP1<br>NOT<br>SWAP1<br>SWAP2<br>AND<br>SWAP3<br>AND<br>SWAP2<br>SWAP1<br>SWAP2<br>OR<br>SWAP1<br>MSTORE<br>PUSH1 0x40<br>MLOAD<br>SWAP3<br>ADD<br>DUP3<br>SWAP1<br>SUB<br>SWAP1<br>SWAP2<br>SHA3<br>PUSH1 0xff<br>AND<br>SWAP3<br>POP<br>POP<br>POP<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x014d<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>MOD<br>SWAP2<br>POP<br>PUSH1 0xff<br>DUP3<br>AND<br>ISZERO<br>ISZERO<br>PUSH2 0x01dc<br>JUMPI<br>POP<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x02<br>DUP4<br>MUL<br>SWAP1<br>CALLER<br>SWAP1<br>DUP3<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP1<br>DUP4<br>SWAP1<br>PUSH1 0x00<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x018d<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>PUSH1 0x02<br>SLOAD<br>PUSH1 0x40<br>MLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>SWAP1<br>SWAP2<br>AND<br>SWAP1<br>PUSH1 0x64<br>PUSH1 0x05<br>DUP5<br>MUL<br>DIV<br>DUP1<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP2<br>PUSH1 0x00<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x01da<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>JUMPDEST<br>POP<br>POP<br>CALLER<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>DUP2<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP4<br>SHA3<br>DUP1<br>SLOAD<br>CALLVALUE<br>ADD<br>SWAP1<br>SSTORE<br>PUSH1 0x01<br>SWAP1<br>SWAP2<br>MSTORE<br>SWAP1<br>SHA3<br>NUMBER<br>SWAP1<br>SSTORE<br>POP<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x020d<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x022f<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x02ad<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>SWAP2<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x024d<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x022f<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x02bf<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x027b<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0284<br>PUSH2 0x02d1<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>SWAP1<br>SWAP3<br>AND<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>PUSH1 0x20<br>DUP2<br>SWAP1<br>MSTORE<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>JUMP<br>STOP<br>