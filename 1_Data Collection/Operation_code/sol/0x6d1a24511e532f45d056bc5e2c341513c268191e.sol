PUSH1 0x80<br>PUSH1 0x40<br>DUP2<br>SWAP1<br>MSTORE<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>SWAP1<br>SWAP2<br>DUP3<br>SWAP2<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>SWAP1<br>PUSH1 0x0a<br>CALLVALUE<br>DIV<br>DUP1<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP2<br>DUP5<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>POP<br>POP<br>CALLER<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x01<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>ISZERO<br>SWAP3<br>POP<br>PUSH1 0xbd<br>SWAP2<br>POP<br>POP<br>JUMPI<br>CALLER<br>PUSH1 0x00<br>DUP2<br>DUP2<br>MSTORE<br>PUSH1 0x02<br>PUSH1 0x20<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP4<br>SHA3<br>SLOAD<br>PUSH1 0x01<br>SWAP1<br>SWAP3<br>MSTORE<br>SWAP1<br>SWAP2<br>SHA3<br>SLOAD<br>SWAP2<br>SWAP4<br>POP<br>PUSH2 0x170c<br>SWAP2<br>PUSH1 0x64<br>PUSH1 0x04<br>SWAP1<br>SWAP2<br>MUL<br>DIV<br>NUMBER<br>SWAP2<br>SWAP1<br>SWAP2<br>SUB<br>MUL<br>DIV<br>SWAP1<br>POP<br>DUP2<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH2 0x08fc<br>DUP3<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>POP<br>POP<br>POP<br>POP<br>POP<br>JUMPDEST<br>POP<br>POP<br>CALLER<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x02<br>PUSH1 0x20<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP4<br>SHA3<br>NUMBER<br>SWAP1<br>SSTORE<br>PUSH1 0x01<br>SWAP1<br>SWAP2<br>MSTORE<br>SWAP1<br>SHA3<br>DUP1<br>SLOAD<br>CALLVALUE<br>ADD<br>SWAP1<br>SSTORE<br>STOP<br>STOP<br>