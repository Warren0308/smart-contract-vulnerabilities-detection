PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x004c<br>JUMPI<br>PUSH1 0x00<br>CALLDATALOAD<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>SWAP1<br>DIV<br>PUSH4 0xffffffff<br>AND<br>DUP1<br>PUSH4 0x3ccfd60b<br>EQ<br>PUSH2 0x004e<br>JUMPI<br>DUP1<br>PUSH4 0xa163a624<br>EQ<br>PUSH2 0x0058<br>JUMPI<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH2 0x0056<br>PUSH2 0x0062<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH2 0x0060<br>PUSH2 0x0137<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>CALLER<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>EQ<br>ISZERO<br>ISZERO<br>PUSH2 0x00bd<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH2 0x08fc<br>ADDRESS<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>BALANCE<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x0135<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>PUSH1 0x00<br>PUSH8 0x0de0b6b3a7640000<br>CALLVALUE<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x01da<br>JUMPI<br>PUSH1 0x01<br>SWAP3<br>POP<br>PUSH1 0x00<br>SWAP2<br>POP<br>PUSH1 0x02<br>CALLVALUE<br>MUL<br>SWAP1<br>POP<br>JUMPDEST<br>PUSH1 0x01<br>ISZERO<br>PUSH2 0x0196<br>JUMPI<br>DUP2<br>PUSH1 0xff<br>AND<br>DUP4<br>PUSH1 0xff<br>AND<br>LT<br>ISZERO<br>PUSH2 0x0176<br>JUMPI<br>PUSH2 0x0196<br>JUMP<br>JUMPDEST<br>DUP1<br>DUP4<br>PUSH1 0xff<br>AND<br>GT<br>ISZERO<br>PUSH2 0x0186<br>JUMPI<br>PUSH2 0x0196<br>JUMP<br>JUMPDEST<br>DUP3<br>SWAP2<br>POP<br>DUP3<br>DUP1<br>PUSH1 0x01<br>ADD<br>SWAP4<br>POP<br>POP<br>PUSH2 0x015c<br>JUMP<br>JUMPDEST<br>CALLER<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH2 0x08fc<br>DUP4<br>PUSH1 0xff<br>AND<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x01d9<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMPDEST<br>POP<br>POP<br>POP<br>JUMP<br>STOP<br>