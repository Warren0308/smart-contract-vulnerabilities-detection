PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0040<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x7da5efc8<br>DUP2<br>EQ<br>PUSH2 0x0045<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0051<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH1 0x20<br>PUSH1 0x04<br>PUSH1 0x24<br>DUP1<br>CALLDATALOAD<br>DUP3<br>DUP2<br>ADD<br>CALLDATALOAD<br>DUP5<br>DUP2<br>MUL<br>DUP1<br>DUP8<br>ADD<br>DUP7<br>ADD<br>SWAP1<br>SWAP8<br>MSTORE<br>DUP1<br>DUP7<br>MSTORE<br>PUSH2 0x00ee<br>SWAP7<br>DUP5<br>CALLDATALOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>SWAP7<br>CALLDATASIZE<br>SWAP7<br>PUSH1 0x44<br>SWAP6<br>SWAP2<br>SWAP5<br>SWAP1<br>SWAP2<br>ADD<br>SWAP3<br>SWAP2<br>DUP3<br>SWAP2<br>DUP6<br>ADD<br>SWAP1<br>DUP5<br>SWAP1<br>DUP1<br>DUP3<br>DUP5<br>CALLDATACOPY<br>POP<br>POP<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>DUP8<br>CALLDATALOAD<br>DUP10<br>ADD<br>DUP1<br>CALLDATALOAD<br>PUSH1 0x20<br>DUP2<br>DUP2<br>MUL<br>DUP5<br>DUP2<br>ADD<br>DUP3<br>ADD<br>SWAP1<br>SWAP6<br>MSTORE<br>DUP2<br>DUP5<br>MSTORE<br>SWAP9<br>SWAP12<br>SWAP11<br>SWAP10<br>DUP10<br>ADD<br>SWAP9<br>SWAP3<br>SWAP8<br>POP<br>SWAP1<br>DUP3<br>ADD<br>SWAP6<br>POP<br>SWAP4<br>POP<br>DUP4<br>SWAP3<br>POP<br>DUP6<br>ADD<br>SWAP1<br>DUP5<br>SWAP1<br>DUP1<br>DUP3<br>DUP5<br>CALLDATACOPY<br>POP<br>SWAP5<br>SWAP8<br>POP<br>PUSH2 0x00f0<br>SWAP7<br>POP<br>POP<br>POP<br>POP<br>POP<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH1 0x00<br>JUMPDEST<br>DUP3<br>MLOAD<br>DUP2<br>LT<br>ISZERO<br>PUSH2 0x0201<br>JUMPI<br>DUP4<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH4 0xa9059cbb<br>DUP5<br>DUP4<br>DUP2<br>MLOAD<br>DUP2<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x0126<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>PUSH1 0x20<br>MUL<br>ADD<br>MLOAD<br>DUP5<br>DUP5<br>DUP2<br>MLOAD<br>DUP2<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x013e<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>PUSH1 0x20<br>MUL<br>ADD<br>MLOAD<br>PUSH1 0x40<br>MLOAD<br>DUP4<br>PUSH4 0xffffffff<br>AND<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>MUL<br>DUP2<br>MSTORE<br>PUSH1 0x04<br>ADD<br>DUP1<br>DUP4<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP3<br>POP<br>POP<br>POP<br>PUSH1 0x20<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>PUSH1 0x00<br>DUP8<br>DUP1<br>EXTCODESIZE<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x01cd<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>GAS<br>CALL<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x01e1<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>RETURNDATASIZE<br>PUSH1 0x20<br>DUP2<br>LT<br>ISZERO<br>PUSH2 0x01f7<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>POP<br>PUSH1 0x01<br>ADD<br>PUSH2 0x00f3<br>JUMP<br>JUMPDEST<br>POP<br>POP<br>POP<br>POP<br>JUMP<br>STOP<br>