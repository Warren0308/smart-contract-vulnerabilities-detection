PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0056<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x4469ed14<br>DUP2<br>EQ<br>PUSH2 0x0135<br>JUMPI<br>DUP1<br>PUSH4 0x8a93dbdf<br>EQ<br>PUSH2 0x015c<br>JUMPI<br>DUP1<br>PUSH4 0xc57981b5<br>EQ<br>PUSH2 0x018a<br>JUMPI<br>JUMPDEST<br>CALLER<br>PUSH1 0x00<br>DUP2<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>DUP2<br>SWAP1<br>MSTORE<br>PUSH1 0x40<br>DUP2<br>SHA3<br>SLOAD<br>ISZERO<br>PUSH2 0x00b5<br>JUMPI<br>PUSH2 0x0074<br>DUP3<br>PUSH2 0x019f<br>JUMP<br>JUMPDEST<br>SWAP1<br>POP<br>ADDRESS<br>BALANCE<br>DUP2<br>LT<br>PUSH2 0x0082<br>JUMPI<br>POP<br>ADDRESS<br>BALANCE<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>DUP4<br>AND<br>SWAP1<br>DUP3<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP1<br>DUP4<br>SWAP1<br>PUSH1 0x00<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>POP<br>POP<br>POP<br>POP<br>POP<br>JUMPDEST<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>DUP3<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x01<br>PUSH1 0x20<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP4<br>SHA3<br>TIMESTAMP<br>SWAP1<br>SSTORE<br>SWAP1<br>DUP3<br>SWAP1<br>MSTORE<br>DUP2<br>SHA3<br>DUP1<br>SLOAD<br>CALLVALUE<br>SWAP1<br>DUP2<br>ADD<br>SWAP1<br>SWAP2<br>SSTORE<br>GT<br>ISZERO<br>PUSH2 0x0131<br>JUMPI<br>PUSH1 0x02<br>SLOAD<br>PUSH1 0x40<br>MLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>SWAP1<br>SWAP2<br>AND<br>SWAP1<br>PUSH1 0x64<br>PUSH1 0x0a<br>CALLVALUE<br>MUL<br>DIV<br>DUP1<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP2<br>PUSH1 0x00<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>POP<br>POP<br>POP<br>POP<br>POP<br>JUMPDEST<br>POP<br>POP<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0141<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x014a<br>PUSH2 0x01ef<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>SWAP2<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0168<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x014a<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x019f<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0196<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x014a<br>PUSH2 0x01f4<br>JUMP<br>JUMPDEST<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>DUP2<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x01<br>PUSH1 0x20<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP4<br>SHA3<br>SLOAD<br>SWAP2<br>DUP4<br>SWAP1<br>MSTORE<br>DUP3<br>SHA3<br>SLOAD<br>PUSH3 0x015180<br>SWAP2<br>TIMESTAMP<br>SUB<br>SWAP1<br>PUSH1 0x64<br>SWAP1<br>PUSH1 0x04<br>MUL<br>DIV<br>MUL<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x01e8<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>DIV<br>SWAP3<br>SWAP2<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x0a<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x04<br>DUP2<br>JUMP<br>STOP<br>