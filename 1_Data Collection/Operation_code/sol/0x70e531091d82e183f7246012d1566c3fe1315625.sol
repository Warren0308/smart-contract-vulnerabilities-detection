PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0062<br>JUMPI<br>PUSH1 0x00<br>CALLDATALOAD<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>SWAP1<br>DIV<br>PUSH4 0xffffffff<br>AND<br>DUP1<br>PUSH4 0x1c3101d3<br>EQ<br>PUSH2 0x0067<br>JUMPI<br>DUP1<br>PUSH4 0x25979983<br>EQ<br>PUSH2 0x00a0<br>JUMPI<br>DUP1<br>PUSH4 0x4e7602d3<br>EQ<br>PUSH2 0x00b8<br>JUMPI<br>DUP1<br>PUSH4 0xd1bc76a1<br>EQ<br>PUSH2 0x0112<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0072<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x009e<br>PUSH1 0x04<br>DUP1<br>DUP1<br>CALLDATALOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP2<br>SWAP1<br>POP<br>POP<br>PUSH2 0x0175<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH2 0x00b6<br>PUSH1 0x04<br>DUP1<br>DUP1<br>CALLDATALOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP2<br>SWAP1<br>POP<br>POP<br>PUSH2 0x01db<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x00c3<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0110<br>PUSH1 0x04<br>DUP1<br>DUP1<br>CALLDATALOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP3<br>ADD<br>DUP1<br>CALLDATALOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP1<br>DUP1<br>PUSH1 0x20<br>MUL<br>PUSH1 0x20<br>ADD<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>ADD<br>PUSH1 0x40<br>MSTORE<br>DUP1<br>SWAP4<br>SWAP3<br>SWAP2<br>SWAP1<br>DUP2<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP4<br>DUP4<br>PUSH1 0x20<br>MUL<br>DUP1<br>DUP3<br>DUP5<br>CALLDATACOPY<br>DUP3<br>ADD<br>SWAP2<br>POP<br>POP<br>POP<br>POP<br>POP<br>POP<br>SWAP2<br>SWAP1<br>POP<br>POP<br>PUSH2 0x0279<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x011d<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0133<br>PUSH1 0x04<br>DUP1<br>DUP1<br>CALLDATALOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP2<br>SWAP1<br>POP<br>POP<br>PUSH2 0x02ee<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>PUSH1 0x01<br>DUP1<br>SLOAD<br>DUP1<br>PUSH1 0x01<br>ADD<br>DUP3<br>DUP2<br>PUSH2 0x0189<br>SWAP2<br>SWAP1<br>PUSH2 0x032d<br>JUMP<br>JUMPDEST<br>SWAP2<br>PUSH1 0x00<br>MSTORE<br>PUSH1 0x20<br>PUSH1 0x00<br>SHA3<br>SWAP1<br>ADD<br>PUSH1 0x00<br>DUP4<br>SWAP1<br>SWAP2<br>SWAP1<br>SWAP2<br>PUSH2 0x0100<br>EXP<br>DUP2<br>SLOAD<br>DUP2<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>MUL<br>NOT<br>AND<br>SWAP1<br>DUP4<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>MUL<br>OR<br>SWAP1<br>SSTORE<br>POP<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SWAP1<br>POP<br>JUMPDEST<br>PUSH1 0x01<br>DUP1<br>SLOAD<br>SWAP1<br>POP<br>DUP2<br>LT<br>ISZERO<br>PUSH2 0x0275<br>JUMPI<br>PUSH1 0x01<br>DUP2<br>DUP2<br>SLOAD<br>DUP2<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x01fd<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>SWAP1<br>PUSH1 0x00<br>MSTORE<br>PUSH1 0x20<br>PUSH1 0x00<br>SHA3<br>SWAP1<br>ADD<br>PUSH1 0x00<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH2 0x08fc<br>DUP4<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x0268<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>DUP1<br>DUP1<br>PUSH1 0x01<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH2 0x01e1<br>JUMP<br>JUMPDEST<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>CALLER<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>EQ<br>ISZERO<br>ISZERO<br>PUSH2 0x02d4<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>DUP1<br>PUSH1 0x01<br>SWAP1<br>DUP1<br>MLOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>PUSH2 0x02ea<br>SWAP3<br>SWAP2<br>SWAP1<br>PUSH2 0x0359<br>JUMP<br>JUMPDEST<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>DUP2<br>DUP2<br>SLOAD<br>DUP2<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x02fd<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>SWAP1<br>PUSH1 0x00<br>MSTORE<br>PUSH1 0x20<br>PUSH1 0x00<br>SHA3<br>SWAP1<br>ADD<br>PUSH1 0x00<br>SWAP2<br>POP<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>DUP2<br>SLOAD<br>DUP2<br>DUP4<br>SSTORE<br>DUP2<br>DUP2<br>ISZERO<br>GT<br>PUSH2 0x0354<br>JUMPI<br>DUP2<br>DUP4<br>PUSH1 0x00<br>MSTORE<br>PUSH1 0x20<br>PUSH1 0x00<br>SHA3<br>SWAP2<br>DUP3<br>ADD<br>SWAP2<br>ADD<br>PUSH2 0x0353<br>SWAP2<br>SWAP1<br>PUSH2 0x03e3<br>JUMP<br>JUMPDEST<br>JUMPDEST<br>POP<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>DUP3<br>DUP1<br>SLOAD<br>DUP3<br>DUP3<br>SSTORE<br>SWAP1<br>PUSH1 0x00<br>MSTORE<br>PUSH1 0x20<br>PUSH1 0x00<br>SHA3<br>SWAP1<br>DUP2<br>ADD<br>SWAP3<br>DUP3<br>ISZERO<br>PUSH2 0x03d2<br>JUMPI<br>SWAP2<br>PUSH1 0x20<br>MUL<br>DUP3<br>ADD<br>JUMPDEST<br>DUP3<br>DUP2<br>GT<br>ISZERO<br>PUSH2 0x03d1<br>JUMPI<br>DUP3<br>MLOAD<br>DUP3<br>PUSH1 0x00<br>PUSH2 0x0100<br>EXP<br>DUP2<br>SLOAD<br>DUP2<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>MUL<br>NOT<br>AND<br>SWAP1<br>DUP4<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>MUL<br>OR<br>SWAP1<br>SSTORE<br>POP<br>SWAP2<br>PUSH1 0x20<br>ADD<br>SWAP2<br>SWAP1<br>PUSH1 0x01<br>ADD<br>SWAP1<br>PUSH2 0x0379<br>JUMP<br>JUMPDEST<br>JUMPDEST<br>POP<br>SWAP1<br>POP<br>PUSH2 0x03df<br>SWAP2<br>SWAP1<br>PUSH2 0x0408<br>JUMP<br>JUMPDEST<br>POP<br>SWAP1<br>JUMP<br>JUMPDEST<br>PUSH2 0x0405<br>SWAP2<br>SWAP1<br>JUMPDEST<br>DUP1<br>DUP3<br>GT<br>ISZERO<br>PUSH2 0x0401<br>JUMPI<br>PUSH1 0x00<br>DUP2<br>PUSH1 0x00<br>SWAP1<br>SSTORE<br>POP<br>PUSH1 0x01<br>ADD<br>PUSH2 0x03e9<br>JUMP<br>JUMPDEST<br>POP<br>SWAP1<br>JUMP<br>JUMPDEST<br>SWAP1<br>JUMP<br>JUMPDEST<br>PUSH2 0x0448<br>SWAP2<br>SWAP1<br>JUMPDEST<br>DUP1<br>DUP3<br>GT<br>ISZERO<br>PUSH2 0x0444<br>JUMPI<br>PUSH1 0x00<br>DUP2<br>DUP2<br>PUSH2 0x0100<br>EXP<br>DUP2<br>SLOAD<br>SWAP1<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>MUL<br>NOT<br>AND<br>SWAP1<br>SSTORE<br>POP<br>PUSH1 0x01<br>ADD<br>PUSH2 0x040e<br>JUMP<br>JUMPDEST<br>POP<br>SWAP1<br>JUMP<br>JUMPDEST<br>SWAP1<br>JUMP<br>STOP<br>