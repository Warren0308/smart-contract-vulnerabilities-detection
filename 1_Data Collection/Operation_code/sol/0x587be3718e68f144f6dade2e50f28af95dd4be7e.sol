PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>CALLER<br>PUSH1 0x00<br>DUP2<br>DUP2<br>MSTORE<br>PUSH1 0x01<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>DUP2<br>SHA3<br>SLOAD<br>ISZERO<br>PUSH1 0xb3<br>JUMPI<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>DUP3<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>DUP2<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP4<br>SHA3<br>SLOAD<br>PUSH1 0x01<br>SWAP1<br>SWAP3<br>MSTORE<br>SWAP1<br>SWAP2<br>SHA3<br>SLOAD<br>PUSH3 0x015180<br>SWAP2<br>TIMESTAMP<br>SUB<br>SWAP1<br>PUSH1 0x64<br>SWAP1<br>DIV<br>PUSH1 0x14<br>MUL<br>MUL<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH1 0x60<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>DIV<br>SWAP1<br>POP<br>ADDRESS<br>BALANCE<br>DUP2<br>GT<br>ISZERO<br>PUSH1 0x6f<br>JUMPI<br>POP<br>ADDRESS<br>BALANCE<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>DUP4<br>AND<br>SWAP1<br>DUP3<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP1<br>DUP4<br>SWAP1<br>PUSH1 0x00<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH1 0xb1<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>JUMPDEST<br>POP<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>DUP2<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP4<br>SHA3<br>TIMESTAMP<br>SWAP1<br>SSTORE<br>PUSH1 0x01<br>SWAP1<br>SWAP2<br>MSTORE<br>SWAP1<br>SHA3<br>DUP1<br>SLOAD<br>CALLVALUE<br>ADD<br>SWAP1<br>SSTORE<br>STOP<br>STOP<br>