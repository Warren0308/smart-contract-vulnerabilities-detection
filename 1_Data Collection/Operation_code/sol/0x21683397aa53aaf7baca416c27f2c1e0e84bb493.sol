PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0098<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x3659cfe6<br>DUP2<br>EQ<br>PUSH2 0x00a2<br>JUMPI<br>DUP1<br>PUSH4 0x4e71e0c8<br>EQ<br>PUSH2 0x00c3<br>JUMPI<br>DUP1<br>PUSH4 0x5c60da1b<br>EQ<br>PUSH2 0x00d8<br>JUMPI<br>DUP1<br>PUSH4 0x636f0cf8<br>EQ<br>PUSH2 0x0109<br>JUMPI<br>DUP1<br>PUSH4 0x8da5cb5b<br>EQ<br>PUSH2 0x011e<br>JUMPI<br>DUP1<br>PUSH4 0xb199efb5<br>EQ<br>PUSH2 0x0133<br>JUMPI<br>DUP1<br>PUSH4 0xdd8fee14<br>EQ<br>PUSH2 0x0148<br>JUMPI<br>DUP1<br>PUSH4 0xe30c3978<br>EQ<br>PUSH2 0x015d<br>JUMPI<br>DUP1<br>PUSH4 0xf2fde38b<br>EQ<br>PUSH2 0x0172<br>JUMPI<br>JUMPDEST<br>PUSH2 0x00a0<br>PUSH2 0x0193<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x00ae<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00a0<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x01ad<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x00cf<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00a0<br>PUSH2 0x01d0<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x00e4<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00ed<br>PUSH2 0x0258<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP1<br>SWAP3<br>AND<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0115<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00ed<br>PUSH2 0x0267<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x012a<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00ed<br>PUSH2 0x0276<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x013f<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00ed<br>PUSH2 0x0285<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0154<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00ed<br>PUSH2 0x0294<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0169<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00ed<br>PUSH2 0x02a3<br>JUMP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x017e<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00a0<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x02b2<br>JUMP<br>JUMPDEST<br>PUSH2 0x019b<br>PUSH2 0x01ab<br>JUMP<br>JUMPDEST<br>PUSH2 0x01ab<br>PUSH2 0x01a6<br>PUSH2 0x030d<br>JUMP<br>JUMPDEST<br>PUSH2 0x0332<br>JUMP<br>JUMPDEST<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>CALLER<br>EQ<br>PUSH2 0x01c4<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH2 0x01cd<br>DUP2<br>PUSH2 0x0356<br>JUMP<br>JUMPDEST<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>CALLER<br>EQ<br>PUSH2 0x01e7<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH1 0x40<br>MLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP4<br>DUP5<br>AND<br>SWAP4<br>SWAP1<br>SWAP2<br>AND<br>SWAP2<br>PUSH32 0x8be0079c531659141344cd1fd0a4f28419497f9722a3daafe3b4186f6b6457e0<br>SWAP2<br>LOG3<br>PUSH1 0x01<br>DUP1<br>SLOAD<br>PUSH1 0x00<br>DUP1<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>NOT<br>SWAP1<br>DUP2<br>AND<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP5<br>AND<br>OR<br>SWAP1<br>SWAP2<br>SSTORE<br>AND<br>SWAP1<br>SSTORE<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>PUSH2 0x0262<br>PUSH2 0x030d<br>JUMP<br>JUMPDEST<br>SWAP1<br>POP<br>SWAP1<br>JUMP<br>JUMPDEST<br>PUSH1 0x04<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x02<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x03<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x01<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>AND<br>CALLER<br>EQ<br>PUSH2 0x02c9<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP2<br>AND<br>ISZERO<br>ISZERO<br>PUSH2 0x02de<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>DUP1<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>NOT<br>AND<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>SWAP3<br>SWAP1<br>SWAP3<br>AND<br>SWAP2<br>SWAP1<br>SWAP2<br>OR<br>SWAP1<br>SSTORE<br>JUMP<br>JUMPDEST<br>PUSH32 0x7050c9e0f4ca769c69bd3a8ef740bc37934f8e2c036e5a723fd8ee048ed3f8c3<br>SLOAD<br>SWAP1<br>JUMP<br>JUMPDEST<br>CALLDATASIZE<br>PUSH1 0x00<br>DUP1<br>CALLDATACOPY<br>PUSH1 0x00<br>DUP1<br>CALLDATASIZE<br>PUSH1 0x00<br>DUP5<br>GAS<br>DELEGATECALL<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>DUP1<br>DUP1<br>ISZERO<br>PUSH2 0x0351<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>RETURN<br>JUMPDEST<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>PUSH2 0x035f<br>DUP2<br>PUSH2 0x039e<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH1 0x01<br>PUSH1 0xa0<br>PUSH1 0x02<br>EXP<br>SUB<br>DUP4<br>AND<br>DUP2<br>MSTORE<br>SWAP1<br>MLOAD<br>PUSH32 0xbc7cd75a20ee27fd9adebab32041f755214dbc6bffa90cc0225b39da2e5c2d3b<br>SWAP2<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>LOG1<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>PUSH2 0x03a9<br>DUP3<br>PUSH2 0x0461<br>JUMP<br>JUMPDEST<br>ISZERO<br>ISZERO<br>PUSH2 0x043c<br>JUMPI<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH32 0x08c379a000000000000000000000000000000000000000000000000000000000<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>PUSH1 0x04<br>DUP3<br>ADD<br>MSTORE<br>PUSH1 0x3b<br>PUSH1 0x24<br>DUP3<br>ADD<br>MSTORE<br>PUSH32 0x43616e6e6f742073657420612070726f787920696d706c656d656e746174696f<br>PUSH1 0x44<br>DUP3<br>ADD<br>MSTORE<br>PUSH32 0x6e20746f2061206e6f6e2d636f6e747261637420616464726573730000000000<br>PUSH1 0x64<br>DUP3<br>ADD<br>MSTORE<br>SWAP1<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x84<br>ADD<br>SWAP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH32 0x7050c9e0f4ca769c69bd3a8ef740bc37934f8e2c036e5a723fd8ee048ed3f8c3<br>SSTORE<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SWAP1<br>EXTCODESIZE<br>GT<br>SWAP1<br>JUMP<br>STOP<br>LOG1<br>PUSH6 0x627a7a723058<br>SHA3<br>MSTORE8<br>DUP16<br>REVERT<br>'21'(Unknown Opcode)<br>'bd'(Unknown Opcode)<br>'29'(Unknown Opcode)<br>'2e'(Unknown Opcode)<br>PUSH6 0x0c4841d0c36d<br>'0c'(Unknown Opcode)<br>