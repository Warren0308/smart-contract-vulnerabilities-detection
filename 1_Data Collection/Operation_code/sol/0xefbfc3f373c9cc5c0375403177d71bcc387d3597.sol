PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0077<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x0a6fbb05<br>DUP2<br>EQ<br>PUSH2 0x0079<br>JUMPI<br>DUP1<br>PUSH4 0x181ffc5f<br>EQ<br>PUSH2 0x0084<br>JUMPI<br>DUP1<br>PUSH4 0x31fd725a<br>EQ<br>PUSH2 0x0097<br>JUMPI<br>DUP1<br>PUSH4 0x410c8005<br>EQ<br>PUSH2 0x00ad<br>JUMPI<br>DUP1<br>PUSH4 0xb456ecc5<br>EQ<br>PUSH2 0x00cc<br>JUMPI<br>DUP1<br>PUSH4 0xbea948c8<br>EQ<br>PUSH2 0x012f<br>JUMPI<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH2 0x0077<br>PUSH1 0x04<br>CALLDATALOAD<br>PUSH2 0x0137<br>JUMP<br>JUMPDEST<br>PUSH2 0x0077<br>PUSH1 0x04<br>DUP1<br>CALLDATALOAD<br>PUSH1 0x24<br>DUP2<br>ADD<br>SWAP2<br>ADD<br>CALLDATALOAD<br>PUSH2 0x0197<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x00a2<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0077<br>PUSH1 0x04<br>CALLDATALOAD<br>PUSH2 0x01ff<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x00b8<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0077<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x0237<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x00d7<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x011d<br>PUSH1 0x04<br>PUSH1 0x24<br>DUP2<br>CALLDATALOAD<br>DUP2<br>DUP2<br>ADD<br>SWAP1<br>DUP4<br>ADD<br>CALLDATALOAD<br>DUP1<br>PUSH1 0x20<br>PUSH1 0x1f<br>DUP3<br>ADD<br>DUP2<br>SWAP1<br>DIV<br>DUP2<br>MUL<br>ADD<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>ADD<br>PUSH1 0x40<br>MSTORE<br>DUP2<br>DUP2<br>MSTORE<br>SWAP3<br>SWAP2<br>SWAP1<br>PUSH1 0x20<br>DUP5<br>ADD<br>DUP4<br>DUP4<br>DUP1<br>DUP3<br>DUP5<br>CALLDATACOPY<br>POP<br>SWAP5<br>SWAP7<br>POP<br>PUSH2 0x0280<br>SWAP6<br>POP<br>POP<br>POP<br>POP<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>PUSH2 0x0077<br>PUSH2 0x02e5<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH1 0xff<br>AND<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x0151<br>JUMPI<br>POP<br>PUSH8 0x0de0b6b3a7640000<br>CALLVALUE<br>GT<br>JUMPDEST<br>DUP1<br>PUSH2 0x015c<br>JUMPI<br>POP<br>PUSH1 0x00<br>SLOAD<br>ISZERO<br>JUMPDEST<br>ISZERO<br>PUSH2 0x0194<br>JUMPI<br>PUSH1 0x00<br>DUP2<br>SWAP1<br>SSTORE<br>PUSH1 0x01<br>DUP1<br>SLOAD<br>PUSH21 0xffffffffffffffffffffffffffffffffffffffff00<br>NOT<br>AND<br>PUSH2 0x0100<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>MUL<br>OR<br>SWAP1<br>SSTORE<br>JUMPDEST<br>POP<br>JUMP<br>JUMPDEST<br>DUP2<br>DUP2<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>DUP4<br>DUP1<br>DUP3<br>DUP5<br>CALLDATACOPY<br>DUP3<br>ADD<br>SWAP2<br>POP<br>POP<br>SWAP3<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>SWAP1<br>SHA3<br>PUSH1 0x00<br>SLOAD<br>EQ<br>ISZERO<br>PUSH2 0x01fb<br>JUMPI<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH2 0x08fc<br>ADDRESS<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>BALANCE<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x01fb<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>DUP2<br>EQ<br>DUP1<br>ISZERO<br>PUSH2 0x0223<br>JUMPI<br>POP<br>PUSH1 0x01<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>PUSH2 0x0100<br>SWAP1<br>SWAP3<br>DIV<br>AND<br>EQ<br>JUMPDEST<br>ISZERO<br>PUSH2 0x0194<br>JUMPI<br>PUSH1 0x01<br>DUP1<br>SLOAD<br>PUSH1 0xff<br>NOT<br>AND<br>DUP2<br>OR<br>SWAP1<br>SSTORE<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>PUSH2 0x0100<br>SWAP1<br>SWAP3<br>DIV<br>AND<br>EQ<br>ISZERO<br>PUSH2 0x0194<br>JUMPI<br>PUSH1 0x02<br>DUP1<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP4<br>AND<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>NOT<br>SWAP1<br>SWAP2<br>AND<br>OR<br>SWAP1<br>SSTORE<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP2<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>DUP1<br>MLOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP1<br>DUP4<br>DUP4<br>JUMPDEST<br>PUSH1 0x20<br>DUP4<br>LT<br>PUSH2 0x02b2<br>JUMPI<br>DUP1<br>MLOAD<br>DUP3<br>MSTORE<br>PUSH1 0x1f<br>NOT<br>SWAP1<br>SWAP3<br>ADD<br>SWAP2<br>PUSH1 0x20<br>SWAP2<br>DUP3<br>ADD<br>SWAP2<br>ADD<br>PUSH2 0x0293<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>DUP4<br>PUSH1 0x20<br>SUB<br>PUSH2 0x0100<br>EXP<br>SUB<br>DUP1<br>NOT<br>DUP3<br>MLOAD<br>AND<br>DUP2<br>DUP5<br>MLOAD<br>AND<br>OR<br>SWAP1<br>SWAP3<br>MSTORE<br>POP<br>POP<br>POP<br>SWAP2<br>SWAP1<br>SWAP2<br>ADD<br>SWAP3<br>POP<br>PUSH1 0x40<br>SWAP2<br>POP<br>POP<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>SHA3<br>SWAP1<br>POP<br>SWAP2<br>SWAP1<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>ISZERO<br>PUSH2 0x0339<br>JUMPI<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH2 0x08fc<br>ADDRESS<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>BALANCE<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x0339<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMP<br>STOP<br>