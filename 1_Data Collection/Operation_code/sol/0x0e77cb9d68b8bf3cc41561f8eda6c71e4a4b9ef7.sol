PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0057<br>JUMPI<br>PUSH1 0x00<br>CALLDATALOAD<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>SWAP1<br>DIV<br>PUSH4 0xffffffff<br>AND<br>DUP1<br>PUSH4 0x3ccfd60b<br>EQ<br>PUSH2 0x005c<br>JUMPI<br>DUP1<br>PUSH4 0x4ba4c16b<br>EQ<br>PUSH2 0x0071<br>JUMPI<br>DUP1<br>PUSH4 0x8da5cb5b<br>EQ<br>PUSH2 0x008c<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0067<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x006f<br>PUSH2 0x00e1<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH2 0x008a<br>PUSH1 0x04<br>DUP1<br>DUP1<br>CALLDATALOAD<br>PUSH1 0xff<br>AND<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP2<br>SWAP1<br>POP<br>POP<br>PUSH2 0x01b6<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0097<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x009f<br>PUSH2 0x0299<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>CALLER<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>EQ<br>ISZERO<br>ISZERO<br>PUSH2 0x013c<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH2 0x08fc<br>ADDRESS<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>BALANCE<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x01b4<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH1 0x00<br>NOT<br>AND<br>DUP2<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>PUSH1 0xff<br>AND<br>PUSH1 0xff<br>AND<br>PUSH32 0x0100000000000000000000000000000000000000000000000000000000000000<br>MUL<br>DUP2<br>MSTORE<br>PUSH1 0x01<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>SHA3<br>PUSH1 0x00<br>NOT<br>AND<br>EQ<br>DUP1<br>ISZERO<br>PUSH2 0x0214<br>JUMPI<br>POP<br>PUSH8 0x0de0b6b3a7640000<br>CALLVALUE<br>LT<br>ISZERO<br>JUMPDEST<br>DUP1<br>ISZERO<br>PUSH2 0x0237<br>JUMPI<br>POP<br>ADDRESS<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>BALANCE<br>CALLVALUE<br>LT<br>ISZERO<br>JUMPDEST<br>ISZERO<br>PUSH2 0x0296<br>JUMPI<br>CALLER<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH2 0x08fc<br>CALLVALUE<br>ADDRESS<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>BALANCE<br>ADD<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x0295<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMPDEST<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>JUMP<br>STOP<br>