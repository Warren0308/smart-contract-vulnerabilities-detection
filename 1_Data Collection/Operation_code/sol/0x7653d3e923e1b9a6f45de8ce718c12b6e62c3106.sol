PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0057<br>JUMPI<br>PUSH1 0x00<br>CALLDATALOAD<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>SWAP1<br>DIV<br>PUSH4 0xffffffff<br>AND<br>DUP1<br>PUSH4 0x1535884e<br>EQ<br>PUSH2 0x005c<br>JUMPI<br>DUP1<br>PUSH4 0x69b7215d<br>EQ<br>PUSH2 0x0083<br>JUMPI<br>DUP1<br>PUSH4 0x8da5cb5b<br>EQ<br>PUSH2 0x00be<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0067<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0081<br>PUSH1 0x04<br>DUP1<br>DUP1<br>CALLDATALOAD<br>PUSH1 0x00<br>NOT<br>AND<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP2<br>SWAP1<br>POP<br>POP<br>PUSH2 0x0113<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x008e<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00a8<br>PUSH1 0x04<br>DUP1<br>DUP1<br>CALLDATALOAD<br>PUSH1 0x00<br>NOT<br>AND<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>SWAP2<br>SWAP1<br>POP<br>POP<br>PUSH2 0x01de<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x00c9<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00d1<br>PUSH2 0x022d<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>PUSH1 0x00<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>CALLER<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>EQ<br>ISZERO<br>ISZERO<br>PUSH2 0x0170<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>TIMESTAMP<br>SWAP1<br>POP<br>DUP1<br>PUSH1 0x01<br>PUSH1 0x00<br>DUP5<br>PUSH1 0x00<br>NOT<br>AND<br>PUSH1 0x00<br>NOT<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x00<br>SHA3<br>DUP2<br>SWAP1<br>SSTORE<br>POP<br>PUSH32 0xf9a924cfa40e56d4657d343862d7f2fbc2c84ad277ce8a3ee65bbfc1bbef7136<br>DUP3<br>DUP3<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>PUSH1 0x00<br>NOT<br>AND<br>PUSH1 0x00<br>NOT<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP3<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP3<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>LOG1<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>PUSH1 0x01<br>PUSH1 0x00<br>DUP5<br>PUSH1 0x00<br>NOT<br>AND<br>PUSH1 0x00<br>NOT<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x00<br>SHA3<br>SLOAD<br>EQ<br>ISZERO<br>ISZERO<br>PUSH2 0x0227<br>JUMPI<br>PUSH1 0x01<br>PUSH1 0x00<br>DUP4<br>PUSH1 0x00<br>NOT<br>AND<br>PUSH1 0x00<br>NOT<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x00<br>SHA3<br>SLOAD<br>SWAP1<br>POP<br>PUSH2 0x0228<br>JUMP<br>JUMPDEST<br>JUMPDEST<br>SWAP2<br>SWAP1<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SWAP1<br>SLOAD<br>SWAP1<br>PUSH2 0x0100<br>EXP<br>SWAP1<br>DIV<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>JUMP<br>STOP<br>