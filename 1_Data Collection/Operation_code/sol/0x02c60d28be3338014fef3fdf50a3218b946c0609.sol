PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x004b<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x66b3f6bf<br>DUP2<br>EQ<br>PUSH2 0x00eb<br>JUMPI<br>DUP1<br>PUSH4 0x8aac1362<br>EQ<br>PUSH2 0x012b<br>JUMPI<br>JUMPDEST<br>CALLER<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>DUP2<br>SWAP1<br>MSTORE<br>PUSH1 0x40<br>DUP2<br>SHA3<br>SLOAD<br>ISZERO<br>PUSH2 0x00c7<br>JUMPI<br>CALLER<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x01<br>PUSH1 0x20<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP4<br>SHA3<br>SLOAD<br>SWAP2<br>DUP4<br>SWAP1<br>MSTORE<br>SWAP1<br>SWAP2<br>SHA3<br>SLOAD<br>PUSH2 0x170c<br>SWAP2<br>NUMBER<br>SUB<br>SWAP1<br>PUSH1 0x64<br>SWAP1<br>PUSH1 0x03<br>MUL<br>DIV<br>MUL<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x0093<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>SWAP2<br>SWAP1<br>DIV<br>SWAP2<br>POP<br>CALLER<br>SWAP1<br>DUP3<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP1<br>DUP4<br>SWAP1<br>PUSH1 0x00<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x00c5<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>JUMPDEST<br>POP<br>CALLER<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x01<br>PUSH1 0x20<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP4<br>SHA3<br>NUMBER<br>SWAP1<br>SSTORE<br>SWAP1<br>DUP3<br>SWAP1<br>MSTORE<br>SWAP1<br>SHA3<br>DUP1<br>SLOAD<br>CALLVALUE<br>ADD<br>SWAP1<br>SSTORE<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x00f7<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0119<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x0159<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>SWAP2<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0137<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0119<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x016b<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>PUSH1 0x20<br>DUP2<br>SWAP1<br>MSTORE<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>DUP2<br>JUMP<br>STOP<br>