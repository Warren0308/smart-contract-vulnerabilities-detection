PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>CALLDATASIZE<br>ISZERO<br>PUSH2 0x00e3<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x014f7f22<br>DUP2<br>EQ<br>PUSH2 0x00e8<br>JUMPI<br>DUP1<br>PUSH4 0x04554443<br>EQ<br>PUSH2 0x010d<br>JUMPI<br>DUP1<br>PUSH4 0x3ac140e8<br>EQ<br>PUSH2 0x013d<br>JUMPI<br>DUP1<br>PUSH4 0x3ccfd60b<br>EQ<br>PUSH2 0x0150<br>JUMPI<br>DUP1<br>PUSH4 0x4cc05a71<br>EQ<br>PUSH2 0x0165<br>JUMPI<br>DUP1<br>PUSH4 0x5918f348<br>EQ<br>PUSH2 0x0178<br>JUMPI<br>DUP1<br>PUSH4 0x63453ae1<br>EQ<br>PUSH2 0x01a7<br>JUMPI<br>DUP1<br>PUSH4 0x75b4d78c<br>EQ<br>PUSH2 0x01c6<br>JUMPI<br>DUP1<br>PUSH4 0x7be8f86b<br>EQ<br>PUSH2 0x01d9<br>JUMPI<br>DUP1<br>PUSH4 0x89f12d00<br>EQ<br>PUSH2 0x020c<br>JUMPI<br>DUP1<br>PUSH4 0x8da5cb5b<br>EQ<br>PUSH2 0x021f<br>JUMPI<br>DUP1<br>PUSH4 0x9b19251a<br>EQ<br>PUSH2 0x0232<br>JUMPI<br>DUP1<br>PUSH4 0xcc642784<br>EQ<br>PUSH2 0x026b<br>JUMPI<br>DUP1<br>PUSH4 0xd54a6bfb<br>EQ<br>PUSH2 0x028a<br>JUMPI<br>DUP1<br>PUSH4 0xdbd65ecc<br>EQ<br>PUSH2 0x02a9<br>JUMPI<br>DUP1<br>PUSH4 0xf2fde38b<br>EQ<br>PUSH2 0x02c7<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x00f3<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00fb<br>PUSH2 0x02e6<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0118<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0120<br>PUSH2 0x02ec<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>PUSH8 0xffffffffffffffff<br>SWAP1<br>SWAP2<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0148<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00fb<br>PUSH2 0x02f1<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x015b<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0163<br>PUSH2 0x02f7<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0170<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00fb<br>PUSH2 0x0459<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0183<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x018b<br>PUSH2 0x045f<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>SWAP2<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x01b2<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0163<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x046e<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x01d1<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00fb<br>PUSH2 0x04fa<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x01e4<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x01f8<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x0500<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>ISZERO<br>ISZERO<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0217<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00fb<br>PUSH2 0x0592<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x022a<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x018b<br>PUSH2 0x0599<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x023d<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0251<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x05a8<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>SWAP2<br>DUP3<br>MSTORE<br>ISZERO<br>ISZERO<br>PUSH1 0x20<br>DUP3<br>ADD<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>DUP2<br>ADD<br>SWAP1<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0276<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x01f8<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x05c4<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0295<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0163<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x05d9<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x02b4<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0163<br>PUSH1 0x04<br>DUP1<br>CALLDATALOAD<br>PUSH1 0x24<br>DUP2<br>ADD<br>SWAP2<br>ADD<br>CALLDATALOAD<br>PUSH2 0x0638<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x02d2<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x0163<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x06b8<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>SLOAD<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x3c<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x04<br>SLOAD<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>SLOAD<br>PUSH1 0x03<br>SLOAD<br>LT<br>PUSH2 0x0307<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>CALLER<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x05<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>PUSH1 0x01<br>ADD<br>SLOAD<br>PUSH1 0xff<br>AND<br>ISZERO<br>PUSH2 0x0330<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>CALLER<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x05<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>PUSH4 0x59682f00<br>SWAP1<br>GT<br>PUSH2 0x0359<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>CALLER<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x05<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>TIMESTAMP<br>LT<br>DUP1<br>ISZERO<br>SWAP1<br>PUSH2 0x0383<br>JUMPI<br>POP<br>PUSH3 0x4c4b40<br>NUMBER<br>GT<br>JUMPDEST<br>ISZERO<br>ISZERO<br>PUSH2 0x038e<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>CALLER<br>DUP2<br>DUP2<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x05<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>DUP1<br>DUP3<br>SHA3<br>PUSH1 0x01<br>SWAP1<br>DUP2<br>ADD<br>DUP1<br>SLOAD<br>PUSH1 0xff<br>NOT<br>AND<br>DUP3<br>OR<br>SWAP1<br>SSTORE<br>PUSH1 0x03<br>DUP1<br>SLOAD<br>DUP3<br>ADD<br>SWAP1<br>SSTORE<br>SLOAD<br>SWAP1<br>SWAP4<br>AND<br>SWAP3<br>PUSH4 0xa9059cbb<br>SWAP3<br>SWAP2<br>PUSH2 0x0168<br>SWAP2<br>MLOAD<br>PUSH1 0x20<br>ADD<br>MSTORE<br>PUSH1 0x40<br>MLOAD<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH4 0xffffffff<br>DUP6<br>AND<br>MUL<br>DUP2<br>MSTORE<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>SWAP3<br>AND<br>PUSH1 0x04<br>DUP4<br>ADD<br>MSTORE<br>PUSH1 0x24<br>DUP3<br>ADD<br>MSTORE<br>PUSH1 0x44<br>ADD<br>PUSH1 0x20<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>PUSH1 0x00<br>DUP8<br>DUP1<br>EXTCODESIZE<br>ISZERO<br>ISZERO<br>PUSH2 0x0431<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x02c6<br>GAS<br>SUB<br>CALL<br>ISZERO<br>ISZERO<br>PUSH2 0x0442<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>MLOAD<br>SWAP1<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x0457<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMP<br>JUMPDEST<br>PUSH1 0x03<br>SLOAD<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>CALLER<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x06<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>PUSH1 0xff<br>AND<br>ISZERO<br>ISZERO<br>PUSH1 0x01<br>EQ<br>PUSH2 0x0498<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>ISZERO<br>ISZERO<br>PUSH2 0x04af<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP2<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x05<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>ISZERO<br>PUSH2 0x04d2<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x05<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>TIMESTAMP<br>PUSH1 0x3c<br>ADD<br>SWAP1<br>SSTORE<br>PUSH1 0x02<br>DUP1<br>SLOAD<br>PUSH1 0x01<br>ADD<br>SWAP1<br>SSTORE<br>JUMP<br>JUMPDEST<br>PUSH2 0x0168<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP2<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x05<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>DUP2<br>SHA3<br>SLOAD<br>ISZERO<br>DUP1<br>PUSH2 0x0545<br>JUMPI<br>POP<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP3<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x05<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>PUSH1 0x01<br>SWAP1<br>DUP2<br>ADD<br>SLOAD<br>PUSH1 0xff<br>AND<br>ISZERO<br>ISZERO<br>EQ<br>JUMPDEST<br>ISZERO<br>PUSH2 0x0552<br>JUMPI<br>POP<br>PUSH1 0x00<br>PUSH2 0x058d<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP3<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x05<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>TIMESTAMP<br>LT<br>DUP1<br>ISZERO<br>SWAP1<br>PUSH2 0x057c<br>JUMPI<br>POP<br>PUSH3 0x4c4b40<br>NUMBER<br>GT<br>JUMPDEST<br>ISZERO<br>PUSH2 0x0589<br>JUMPI<br>POP<br>PUSH1 0x01<br>PUSH2 0x058d<br>JUMP<br>JUMPDEST<br>POP<br>PUSH1 0x00<br>JUMPDEST<br>SWAP2<br>SWAP1<br>POP<br>JUMP<br>JUMPDEST<br>PUSH3 0x4c4b40<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x05<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>DUP1<br>SLOAD<br>PUSH1 0x01<br>SWAP1<br>SWAP2<br>ADD<br>SLOAD<br>PUSH1 0xff<br>AND<br>DUP3<br>JUMP<br>JUMPDEST<br>PUSH1 0x06<br>PUSH1 0x20<br>MSTORE<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>PUSH1 0xff<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>PUSH2 0x05f4<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP2<br>AND<br>ISZERO<br>ISZERO<br>PUSH2 0x0609<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>DUP1<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>NOT<br>AND<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP3<br>SWAP1<br>SWAP3<br>AND<br>SWAP2<br>SWAP1<br>SWAP2<br>OR<br>SWAP1<br>SSTORE<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>PUSH2 0x0654<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH1 0x00<br>JUMPDEST<br>DUP2<br>DUP2<br>LT<br>ISZERO<br>PUSH2 0x06ab<br>JUMPI<br>PUSH1 0x01<br>PUSH1 0x06<br>PUSH1 0x00<br>DUP6<br>DUP6<br>DUP6<br>DUP2<br>DUP2<br>LT<br>PUSH2 0x0672<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>PUSH1 0x20<br>SWAP1<br>DUP2<br>MUL<br>SWAP3<br>SWAP1<br>SWAP3<br>ADD<br>CALLDATALOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>DUP4<br>MSTORE<br>POP<br>DUP2<br>ADD<br>SWAP2<br>SWAP1<br>SWAP2<br>MSTORE<br>PUSH1 0x40<br>ADD<br>PUSH1 0x00<br>SHA3<br>DUP1<br>SLOAD<br>PUSH1 0xff<br>NOT<br>AND<br>SWAP2<br>ISZERO<br>ISZERO<br>SWAP2<br>SWAP1<br>SWAP2<br>OR<br>SWAP1<br>SSTORE<br>PUSH1 0x01<br>ADD<br>PUSH2 0x0658<br>JUMP<br>JUMPDEST<br>POP<br>PUSH1 0x04<br>DUP1<br>SLOAD<br>SWAP1<br>SWAP2<br>ADD<br>SWAP1<br>SSTORE<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>PUSH2 0x06d3<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP2<br>AND<br>ISZERO<br>PUSH2 0x070b<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>NOT<br>AND<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP4<br>AND<br>OR<br>SWAP1<br>SSTORE<br>JUMPDEST<br>POP<br>JUMP<br>STOP<br>