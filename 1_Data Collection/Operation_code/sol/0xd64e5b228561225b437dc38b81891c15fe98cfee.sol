PUSH1 0x60<br>PUSH1 0x40<br>MSTORE<br>CALLDATASIZE<br>ISZERO<br>PUSH2 0x008b<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x1793c314<br>DUP2<br>EQ<br>PUSH2 0x00b3<br>JUMPI<br>DUP1<br>PUSH4 0x21e92d49<br>EQ<br>PUSH2 0x00da<br>JUMPI<br>DUP1<br>PUSH4 0x872fe620<br>EQ<br>PUSH2 0x00f3<br>JUMPI<br>DUP1<br>PUSH4 0x8da5cb5b<br>EQ<br>PUSH2 0x0108<br>JUMPI<br>DUP1<br>PUSH4 0x9066314e<br>EQ<br>PUSH2 0x0137<br>JUMPI<br>DUP1<br>PUSH4 0xa5bbc423<br>EQ<br>PUSH2 0x0151<br>JUMPI<br>DUP1<br>PUSH4 0xc46f3e25<br>EQ<br>PUSH2 0x01a4<br>JUMPI<br>DUP1<br>PUSH4 0xf341cae8<br>EQ<br>PUSH2 0x01cb<br>JUMPI<br>JUMPDEST<br>JUMPDEST<br>PUSH1 0x00<br>JUMPDEST<br>PUSH1 0x12<br>SLOAD<br>PUSH3 0x010000<br>SWAP1<br>DIV<br>PUSH1 0xff<br>AND<br>PUSH1 0x02<br>DUP2<br>GT<br>ISZERO<br>PUSH2 0x00a6<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>EQ<br>PUSH2 0x00b0<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x00be<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00b0<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>DUP2<br>AND<br>SWAP1<br>PUSH1 0x24<br>CALLDATALOAD<br>AND<br>PUSH2 0x01e6<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH2 0x00b0<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH1 0x24<br>CALLDATALOAD<br>PUSH2 0x0294<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x00fe<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00b0<br>PUSH2 0x033a<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0113<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x011b<br>PUSH2 0x042c<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>SWAP2<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x0142<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00b0<br>PUSH1 0x04<br>CALLDATALOAD<br>ISZERO<br>ISZERO<br>PUSH2 0x043b<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x015c<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00b0<br>PUSH1 0x04<br>PUSH1 0x24<br>DUP2<br>CALLDATALOAD<br>DUP2<br>DUP2<br>ADD<br>SWAP1<br>DUP4<br>ADD<br>CALLDATALOAD<br>DUP1<br>PUSH1 0x20<br>PUSH1 0x1f<br>DUP3<br>ADD<br>DUP2<br>SWAP1<br>DIV<br>DUP2<br>MUL<br>ADD<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>ADD<br>PUSH1 0x40<br>MSTORE<br>DUP2<br>DUP2<br>MSTORE<br>SWAP3<br>SWAP2<br>SWAP1<br>PUSH1 0x20<br>DUP5<br>ADD<br>DUP4<br>DUP4<br>DUP1<br>DUP3<br>DUP5<br>CALLDATACOPY<br>POP<br>SWAP5<br>SWAP7<br>POP<br>PUSH2 0x049e<br>SWAP6<br>POP<br>POP<br>POP<br>POP<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x01af<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x01b7<br>PUSH2 0x061a<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>ISZERO<br>ISZERO<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x01d6<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x00b0<br>PUSH1 0x04<br>CALLDATALOAD<br>PUSH1 0x24<br>CALLDATALOAD<br>PUSH2 0x0623<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>PUSH1 0x12<br>SLOAD<br>PUSH1 0xff<br>AND<br>ISZERO<br>DUP1<br>PUSH2 0x01fd<br>JUMPI<br>POP<br>PUSH1 0x11<br>SLOAD<br>PUSH1 0x3c<br>ADD<br>TIMESTAMP<br>GT<br>ISZERO<br>JUMPDEST<br>ISZERO<br>PUSH2 0x0207<br>JUMPI<br>PUSH2 0x028f<br>JUMP<br>JUMPDEST<br>PUSH1 0x12<br>SLOAD<br>PUSH2 0x0100<br>SWAP1<br>DIV<br>PUSH1 0xff<br>AND<br>ISZERO<br>PUSH2 0x0251<br>JUMPI<br>PUSH1 0x03<br>SLOAD<br>DUP2<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP2<br>SWAP1<br>SWAP2<br>AND<br>PUSH13 0x01000000000000000000000000<br>MUL<br>DUP2<br>MSTORE<br>PUSH1 0x14<br>ADD<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>SWAP1<br>SHA3<br>EQ<br>PUSH2 0x0251<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMPDEST<br>DUP2<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH2 0x08fc<br>ADDRESS<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>BALANCE<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x028f<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMPDEST<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>PUSH2 0x02af<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x00<br>JUMPDEST<br>PUSH1 0x12<br>SLOAD<br>PUSH3 0x010000<br>SWAP1<br>DIV<br>PUSH1 0xff<br>AND<br>PUSH1 0x02<br>DUP2<br>GT<br>ISZERO<br>PUSH2 0x02c9<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>EQ<br>PUSH2 0x02d3<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>ADDRESS<br>AND<br>BALANCE<br>DUP2<br>GT<br>ISZERO<br>PUSH2 0x02ea<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP3<br>AND<br>ISZERO<br>ISZERO<br>PUSH2 0x02ff<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP3<br>AND<br>DUP2<br>ISZERO<br>PUSH2 0x08fc<br>MUL<br>DUP3<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x0330<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>TIMESTAMP<br>PUSH1 0x11<br>SSTORE<br>JUMPDEST<br>JUMPDEST<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x03<br>SLOAD<br>CALLER<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP2<br>SWAP1<br>SWAP2<br>AND<br>PUSH13 0x01000000000000000000000000<br>MUL<br>DUP2<br>MSTORE<br>PUSH1 0x14<br>ADD<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>SWAP1<br>SHA3<br>EQ<br>PUSH2 0x0374<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>JUMPDEST<br>PUSH1 0x12<br>SLOAD<br>PUSH3 0x010000<br>SWAP1<br>DIV<br>PUSH1 0xff<br>AND<br>PUSH1 0x02<br>DUP2<br>GT<br>ISZERO<br>PUSH2 0x038e<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>EQ<br>PUSH2 0x03b5<br>JUMPI<br>TIMESTAMP<br>PUSH1 0x0f<br>SSTORE<br>PUSH1 0x12<br>DUP1<br>SLOAD<br>PUSH1 0x01<br>SWAP2<br>SWAP1<br>PUSH3 0xff0000<br>NOT<br>AND<br>PUSH3 0x010000<br>DUP4<br>JUMPDEST<br>MUL<br>OR<br>SWAP1<br>SSTORE<br>POP<br>PUSH2 0x0428<br>JUMP<br>JUMPDEST<br>PUSH1 0x0f<br>SLOAD<br>PUSH1 0x3c<br>ADD<br>TIMESTAMP<br>GT<br>PUSH2 0x03c6<br>JUMPI<br>PUSH2 0x0428<br>JUMP<br>JUMPDEST<br>JUMPDEST<br>PUSH1 0x01<br>JUMPDEST<br>PUSH1 0x12<br>SLOAD<br>PUSH3 0x010000<br>SWAP1<br>DIV<br>PUSH1 0xff<br>AND<br>PUSH1 0x02<br>DUP2<br>GT<br>ISZERO<br>PUSH2 0x03e1<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>EQ<br>PUSH2 0x03eb<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH2 0x08fc<br>ADDRESS<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>BALANCE<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x0428<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMPDEST<br>JUMPDEST<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>SLOAD<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>PUSH2 0x0456<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x00<br>JUMPDEST<br>PUSH1 0x12<br>SLOAD<br>PUSH3 0x010000<br>SWAP1<br>DIV<br>PUSH1 0xff<br>AND<br>PUSH1 0x02<br>DUP2<br>GT<br>ISZERO<br>PUSH2 0x0470<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>EQ<br>PUSH2 0x047a<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x12<br>DUP1<br>SLOAD<br>PUSH1 0xff<br>DUP2<br>AND<br>ISZERO<br>PUSH1 0xff<br>NOT<br>SWAP1<br>SWAP2<br>AND<br>OR<br>PUSH2 0xff00<br>NOT<br>AND<br>PUSH2 0x0100<br>DUP4<br>ISZERO<br>ISZERO<br>MUL<br>OR<br>SWAP1<br>SSTORE<br>JUMPDEST<br>JUMPDEST<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>DUP2<br>SWAP1<br>DUP2<br>CALLER<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>DUP1<br>MLOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP1<br>DUP4<br>DUP4<br>JUMPDEST<br>PUSH1 0x20<br>DUP4<br>LT<br>PUSH2 0x04d5<br>JUMPI<br>DUP1<br>MLOAD<br>DUP3<br>MSTORE<br>JUMPDEST<br>PUSH1 0x1f<br>NOT<br>SWAP1<br>SWAP3<br>ADD<br>SWAP2<br>PUSH1 0x20<br>SWAP2<br>DUP3<br>ADD<br>SWAP2<br>ADD<br>PUSH2 0x04b5<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>DUP4<br>PUSH1 0x20<br>SUB<br>PUSH2 0x0100<br>EXP<br>SUB<br>DUP1<br>NOT<br>DUP3<br>MLOAD<br>AND<br>DUP2<br>DUP5<br>MLOAD<br>AND<br>DUP1<br>DUP3<br>OR<br>DUP6<br>MSTORE<br>POP<br>POP<br>POP<br>POP<br>POP<br>POP<br>SWAP1<br>POP<br>ADD<br>DUP3<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH13 0x01000000000000000000000000<br>MUL<br>DUP2<br>MSTORE<br>PUSH1 0x14<br>ADD<br>SWAP3<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>SWAP1<br>SHA3<br>EQ<br>PUSH2 0x0535<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x10<br>SLOAD<br>PUSH3 0x015180<br>ADD<br>TIMESTAMP<br>GT<br>PUSH2 0x0548<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x04<br>SLOAD<br>DUP3<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP3<br>DUP1<br>MLOAD<br>SWAP1<br>PUSH1 0x20<br>ADD<br>SWAP1<br>DUP1<br>DUP4<br>DUP4<br>JUMPDEST<br>PUSH1 0x20<br>DUP4<br>LT<br>PUSH2 0x057c<br>JUMPI<br>DUP1<br>MLOAD<br>DUP3<br>MSTORE<br>JUMPDEST<br>PUSH1 0x1f<br>NOT<br>SWAP1<br>SWAP3<br>ADD<br>SWAP2<br>PUSH1 0x20<br>SWAP2<br>DUP3<br>ADD<br>SWAP2<br>ADD<br>PUSH2 0x055c<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>DUP4<br>PUSH1 0x20<br>SUB<br>PUSH2 0x0100<br>EXP<br>SUB<br>DUP1<br>NOT<br>DUP3<br>MLOAD<br>AND<br>DUP2<br>DUP5<br>MLOAD<br>AND<br>OR<br>SWAP1<br>SWAP3<br>MSTORE<br>POP<br>POP<br>POP<br>SWAP2<br>SWAP1<br>SWAP2<br>ADD<br>SWAP3<br>POP<br>PUSH1 0x40<br>SWAP2<br>POP<br>POP<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>SWAP1<br>SHA3<br>EQ<br>PUSH2 0x05b3<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x02<br>JUMPDEST<br>PUSH1 0x12<br>SLOAD<br>PUSH3 0x010000<br>SWAP1<br>DIV<br>PUSH1 0xff<br>AND<br>PUSH1 0x02<br>DUP2<br>GT<br>ISZERO<br>PUSH2 0x05cd<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>EQ<br>PUSH2 0x05d7<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>PUSH2 0x08fc<br>ADDRESS<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>BALANCE<br>SWAP1<br>DUP2<br>ISZERO<br>MUL<br>SWAP1<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x00<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>DUP4<br>SUB<br>DUP2<br>DUP6<br>DUP9<br>DUP9<br>CALL<br>SWAP4<br>POP<br>POP<br>POP<br>POP<br>ISZERO<br>ISZERO<br>PUSH2 0x028f<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>JUMPDEST<br>JUMPDEST<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x12<br>SLOAD<br>PUSH1 0xff<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>SWAP1<br>CALLER<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>DUP2<br>AND<br>SWAP2<br>AND<br>EQ<br>PUSH2 0x0643<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>DUP3<br>ISZERO<br>DUP1<br>ISZERO<br>SWAP1<br>PUSH2 0x0651<br>JUMPI<br>POP<br>DUP4<br>ISZERO<br>ISZERO<br>JUMPDEST<br>ISZERO<br>ISZERO<br>PUSH2 0x065c<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH1 0x00<br>SWAP1<br>POP<br>DUP1<br>JUMPDEST<br>PUSH1 0x0a<br>DUP3<br>PUSH1 0xff<br>AND<br>LT<br>ISZERO<br>PUSH2 0x06b1<br>JUMPI<br>DUP3<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>SWAP1<br>SHA3<br>PUSH1 0x05<br>PUSH1 0xff<br>DUP5<br>AND<br>PUSH1 0x0a<br>DUP2<br>LT<br>PUSH2 0x0692<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>ADD<br>PUSH1 0x00<br>JUMPDEST<br>POP<br>SLOAD<br>EQ<br>ISZERO<br>PUSH2 0x06a6<br>JUMPI<br>POP<br>PUSH1 0x01<br>PUSH2 0x06b1<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SWAP1<br>SWAP2<br>ADD<br>SWAP1<br>PUSH2 0x0663<br>JUMP<br>JUMPDEST<br>DUP1<br>ISZERO<br>ISZERO<br>PUSH2 0x06bd<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>TIMESTAMP<br>PUSH1 0x10<br>SSTORE<br>PUSH1 0x12<br>DUP1<br>SLOAD<br>PUSH1 0x02<br>SWAP2<br>SWAP1<br>PUSH3 0xff0000<br>NOT<br>AND<br>PUSH3 0x010000<br>DUP4<br>JUMPDEST<br>MUL<br>OR<br>SWAP1<br>SSTORE<br>POP<br>PUSH1 0x00<br>DUP5<br>SWAP1<br>SSTORE<br>JUMPDEST<br>JUMPDEST<br>POP<br>POP<br>POP<br>POP<br>JUMP<br>STOP<br>