PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x004b<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0xa035b1fe<br>DUP2<br>EQ<br>PUSH2 0x0147<br>JUMPI<br>DUP1<br>PUSH4 0xdfbf53ae<br>EQ<br>PUSH2 0x016e<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>CALLVALUE<br>LT<br>ISZERO<br>PUSH2 0x005a<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>CALLVALUE<br>GT<br>ISZERO<br>PUSH2 0x00b0<br>JUMPI<br>CALLER<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>PUSH2 0x08fc<br>PUSH1 0x00<br>SLOAD<br>CALLVALUE<br>SUB<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x00ae<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH1 0x40<br>MLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>SWAP1<br>SWAP4<br>AND<br>SWAP3<br>DUP2<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP3<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x011d<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH1 0x40<br>MLOAD<br>CALLER<br>SWAP3<br>DUP3<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>SWAP3<br>SWAP2<br>SWAP1<br>DUP2<br>DUP2<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x011b<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>JUMPDEST<br>PUSH1 0x01<br>DUP1<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>NOT<br>AND<br>CALLER<br>OR<br>SWAP1<br>SSTORE<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH1 0x02<br>MUL<br>SWAP1<br>SSTORE<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0153<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x015c<br>PUSH2 0x01ac<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>SWAP2<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x017a<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0183<br>PUSH2 0x01b2<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>SWAP1<br>SWAP3<br>AND<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>DUP2<br>JUMP<br>STOP<br>