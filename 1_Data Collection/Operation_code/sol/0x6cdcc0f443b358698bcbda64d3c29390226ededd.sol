PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x004c<br>JUMPI<br>PUSH1 0x00<br>CALLDATALOAD<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>SWAP1<br>DIV<br>PUSH4 0xffffffff<br>AND<br>DUP1<br>PUSH4 0x6e66f6e9<br>EQ<br>PUSH2 0x0350<br>JUMPI<br>DUP1<br>PUSH4 0x9518589a<br>EQ<br>PUSH2 0x03a7<br>JUMPI<br>JUMPDEST<br>PUSH1 0x04<br>SLOAD<br>CALLVALUE<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x0059<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>DIV<br>PUSH1 0x03<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>PUSH8 0x4563918244f40000<br>CALLVALUE<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x009d<br>JUMPI<br>PUSH1 0x04<br>SLOAD<br>CALLVALUE<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x007e<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>DIV<br>PUSH1 0x03<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>PUSH1 0x02<br>PUSH1 0x03<br>DUP1<br>SLOAD<br>MUL<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x0095<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>DIV<br>PUSH1 0x03<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>JUMPDEST<br>PUSH8 0xd02ab486cedc0000<br>CALLVALUE<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x00cf<br>JUMPI<br>PUSH1 0x04<br>SLOAD<br>CALLVALUE<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x00bb<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>DIV<br>PUSH1 0x03<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>PUSH1 0x02<br>PUSH1 0x03<br>SLOAD<br>MUL<br>PUSH1 0x03<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>JUMPDEST<br>PUSH9 0x015af1d78b58c40000<br>CALLVALUE<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x0101<br>JUMPI<br>PUSH1 0x04<br>SLOAD<br>CALLVALUE<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x00ee<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>DIV<br>PUSH1 0x03<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>PUSH1 0x03<br>DUP1<br>SLOAD<br>MUL<br>PUSH1 0x03<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>JUMPDEST<br>PUSH1 0x07<br>PUSH1 0x00<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH4 0x23b872dd<br>PUSH1 0x01<br>PUSH1 0x00<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>CALLER<br>PUSH1 0x03<br>SLOAD<br>PUSH1 0x40<br>MLOAD<br>DUP5<br>PUSH4 0xffffffff<br>AND<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>MUL<br>DUP2<br>MSTORE<br>PUSH1 0x04<br>ADD<br>DUP1<br>DUP5<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP4<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>PUSH1 0x20<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>PUSH1 0x00<br>DUP8<br>DUP1<br>EXTCODESIZE<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x021e<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>GAS<br>CALL<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x0232<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>RETURNDATASIZE<br>PUSH1 0x20<br>DUP2<br>LT<br>ISZERO<br>PUSH2 0x0248<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>DUP2<br>ADD<br>SWAP1<br>DUP1<br>DUP1<br>MLOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP3<br>SWAP2<br>SWAP1<br>POP<br>POP<br>POP<br>POP<br>PUSH1 0x0a<br>CALLVALUE<br>DUP2<br>ISZERO<br>ISZERO<br>PUSH2 0x0266<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>DIV<br>PUSH1 0x06<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>PUSH1 0x06<br>SLOAD<br>CALLVALUE<br>SUB<br>PUSH1 0x05<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>PUSH1 0x01<br>PUSH1 0x00<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH2 0x08fc<br>PUSH1 0x05<br>SLOAD<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x02e2<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>PUSH1 0x02<br>PUSH1 0x00<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH2 0x08fc<br>PUSH1 0x06<br>SLOAD<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x034d<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x035c<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0365<br>PUSH2 0x03d2<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x03b3<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x03bc<br>PUSH2 0x03f8<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>PUSH1 0x07<br>PUSH1 0x00<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x04<br>SLOAD<br>DUP2<br>JUMP<br>STOP<br>