PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x004c<br>JUMPI<br>PUSH1 0x00<br>CALLDATALOAD<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>SWAP1<br>DIV<br>PUSH4 0xffffffff<br>AND<br>DUP1<br>PUSH4 0x8da5cb5b<br>EQ<br>PUSH2 0x0051<br>JUMPI<br>DUP1<br>PUSH4 0xad8733ca<br>EQ<br>PUSH2 0x00a8<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x005d<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0066<br>PUSH2 0x0133<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x00b4<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0119<br>PUSH1 0x04<br>DUP1<br>CALLDATASIZE<br>SUB<br>DUP2<br>ADD<br>SWAP1<br>DUP1<br>DUP1<br>CALLDATALOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP3<br>SWAP2<br>SWAP1<br>DUP1<br>CALLDATALOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP3<br>ADD<br>DUP1<br>CALLDATALOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP2<br>SWAP1<br>SWAP2<br>SWAP3<br>SWAP4<br>SWAP2<br>SWAP3<br>SWAP4<br>SWAP1<br>DUP1<br>CALLDATALOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP3<br>ADD<br>DUP1<br>CALLDATALOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP2<br>SWAP1<br>SWAP2<br>SWAP3<br>SWAP4<br>SWAP2<br>SWAP3<br>SWAP4<br>SWAP1<br>POP<br>POP<br>POP<br>PUSH2 0x0158<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>ISZERO<br>ISZERO<br>ISZERO<br>ISZERO<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>PUSH1 0x00<br>DUP5<br>DUP5<br>SWAP1<br>POP<br>DUP8<br>DUP8<br>SWAP1<br>POP<br>EQ<br>ISZERO<br>ISZERO<br>PUSH2 0x016e<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>PUSH1 0x96<br>DUP8<br>DUP8<br>SWAP1<br>POP<br>GT<br>ISZERO<br>ISZERO<br>ISZERO<br>PUSH2 0x017e<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>PUSH1 0x00<br>SWAP2<br>POP<br>JUMPDEST<br>DUP7<br>DUP7<br>SWAP1<br>POP<br>DUP3<br>PUSH1 0xff<br>AND<br>LT<br>ISZERO<br>PUSH2 0x02df<br>JUMPI<br>PUSH8 0x0de0b6b3a7640000<br>DUP6<br>DUP6<br>DUP5<br>PUSH1 0xff<br>AND<br>DUP2<br>DUP2<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x01ab<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>SWAP1<br>POP<br>PUSH1 0x20<br>MUL<br>ADD<br>CALLDATALOAD<br>MUL<br>SWAP1<br>POP<br>PUSH1 0x01<br>ISZERO<br>ISZERO<br>DUP9<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH4 0xa9059cbb<br>DUP10<br>DUP10<br>DUP7<br>PUSH1 0xff<br>AND<br>DUP2<br>DUP2<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x01e6<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>SWAP1<br>POP<br>PUSH1 0x20<br>MUL<br>ADD<br>CALLDATALOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP5<br>PUSH1 0x40<br>MLOAD<br>DUP4<br>PUSH4 0xffffffff<br>AND<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>MUL<br>DUP2<br>MSTORE<br>PUSH1 0x04<br>ADD<br>DUP1<br>DUP4<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP3<br>POP<br>POP<br>POP<br>PUSH1 0x20<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>PUSH1 0x00<br>DUP8<br>DUP1<br>EXTCODESIZE<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x0289<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>GAS<br>CALL<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x029d<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>RETURNDATASIZE<br>PUSH1 0x20<br>DUP2<br>LT<br>ISZERO<br>PUSH2 0x02b3<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>DUP2<br>ADD<br>SWAP1<br>DUP1<br>DUP1<br>MLOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP3<br>SWAP2<br>SWAP1<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>EQ<br>ISZERO<br>ISZERO<br>PUSH2 0x02d2<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>DUP2<br>DUP1<br>PUSH1 0x01<br>ADD<br>SWAP3<br>POP<br>POP<br>PUSH2 0x0183<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SWAP3<br>POP<br>POP<br>POP<br>SWAP6<br>SWAP5<br>POP<br>POP<br>POP<br>POP<br>POP<br>JUMP<br>STOP<br>