PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x0056<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x4c21eb07<br>DUP2<br>EQ<br>PUSH2 0x005b<br>JUMPI<br>DUP1<br>PUSH4 0x6d4ce63c<br>EQ<br>PUSH2 0x007d<br>JUMPI<br>DUP1<br>PUSH4 0xc2bc2efc<br>EQ<br>PUSH2 0x0107<br>JUMPI<br>JUMPDEST<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0067<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x007b<br>PUSH1 0x04<br>DUP1<br>CALLDATALOAD<br>PUSH1 0x24<br>DUP2<br>ADD<br>SWAP2<br>ADD<br>CALLDATALOAD<br>PUSH2 0x0135<br>JUMP<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0089<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0092<br>PUSH2 0x0467<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH1 0x20<br>DUP1<br>DUP3<br>MSTORE<br>DUP4<br>MLOAD<br>DUP2<br>DUP4<br>ADD<br>MSTORE<br>DUP4<br>MLOAD<br>SWAP2<br>SWAP3<br>DUP4<br>SWAP3<br>SWAP1<br>DUP4<br>ADD<br>SWAP2<br>DUP6<br>ADD<br>SWAP1<br>DUP1<br>DUP4<br>DUP4<br>PUSH1 0x00<br>JUMPDEST<br>DUP4<br>DUP2<br>LT<br>ISZERO<br>PUSH2 0x00cc<br>JUMPI<br>DUP2<br>DUP2<br>ADD<br>MLOAD<br>DUP4<br>DUP3<br>ADD<br>MSTORE<br>PUSH1 0x20<br>ADD<br>PUSH2 0x00b4<br>JUMP<br>JUMPDEST<br>POP<br>POP<br>POP<br>POP<br>SWAP1<br>POP<br>SWAP1<br>DUP2<br>ADD<br>SWAP1<br>PUSH1 0x1f<br>AND<br>DUP1<br>ISZERO<br>PUSH2 0x00f9<br>JUMPI<br>DUP1<br>DUP3<br>SUB<br>DUP1<br>MLOAD<br>PUSH1 0x01<br>DUP4<br>PUSH1 0x20<br>SUB<br>PUSH2 0x0100<br>EXP<br>SUB<br>NOT<br>AND<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>SWAP2<br>POP<br>JUMPDEST<br>POP<br>SWAP3<br>POP<br>POP<br>POP<br>PUSH1 0x40<br>MLOAD<br>DUP1<br>SWAP2<br>SUB<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0113<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x0092<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x0477<br>JUMP<br>JUMPDEST<br>CALLER<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>DUP2<br>SWAP1<br>MSTORE<br>PUSH1 0x40<br>DUP2<br>SHA3<br>SLOAD<br>PUSH1 0x60<br>SWAP2<br>SWAP1<br>DUP2<br>SWAP1<br>DUP2<br>SWAP1<br>ISZERO<br>PUSH2 0x01b9<br>JUMPI<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH32 0x08c379a000000000000000000000000000000000000000000000000000000000<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>PUSH1 0x04<br>DUP3<br>ADD<br>MSTORE<br>PUSH1 0x15<br>PUSH1 0x24<br>DUP3<br>ADD<br>MSTORE<br>PUSH32 0x6164647265737320616c726561647920626f756e640000000000000000000000<br>PUSH1 0x44<br>DUP3<br>ADD<br>MSTORE<br>SWAP1<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x64<br>ADD<br>SWAP1<br>REVERT<br>JUMPDEST<br>DUP6<br>DUP6<br>DUP1<br>DUP1<br>PUSH1 0x1f<br>ADD<br>PUSH1 0x20<br>DUP1<br>SWAP2<br>DIV<br>MUL<br>PUSH1 0x20<br>ADD<br>PUSH1 0x40<br>MLOAD<br>SWAP1<br>DUP2<br>ADD<br>PUSH1 0x40<br>MSTORE<br>DUP1<br>SWAP4<br>SWAP3<br>SWAP2<br>SWAP1<br>DUP2<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>ADD<br>DUP4<br>DUP4<br>DUP1<br>DUP3<br>DUP5<br>CALLDATACOPY<br>DUP3<br>ADD<br>SWAP2<br>POP<br>POP<br>POP<br>POP<br>POP<br>POP<br>SWAP4<br>POP<br>DUP4<br>MLOAD<br>PUSH1 0x0c<br>EQ<br>ISZERO<br>ISZERO<br>PUSH2 0x0260<br>JUMPI<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH32 0x08c379a000000000000000000000000000000000000000000000000000000000<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>PUSH1 0x04<br>DUP3<br>ADD<br>MSTORE<br>PUSH1 0x0c<br>PUSH1 0x24<br>DUP3<br>ADD<br>MSTORE<br>PUSH32 0x776f726e67206c656e6774680000000000000000000000000000000000000000<br>PUSH1 0x44<br>DUP3<br>ADD<br>MSTORE<br>SWAP1<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x64<br>ADD<br>SWAP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x00<br>SWAP3<br>POP<br>JUMPDEST<br>PUSH1 0x0c<br>DUP4<br>LT<br>ISZERO<br>PUSH2 0x03ef<br>JUMPI<br>DUP4<br>DUP4<br>DUP2<br>MLOAD<br>DUP2<br>LT<br>ISZERO<br>ISZERO<br>PUSH2 0x027c<br>JUMPI<br>'fe'(Unknown Opcode)<br>JUMPDEST<br>ADD<br>PUSH1 0x20<br>ADD<br>MLOAD<br>PUSH32 0x0100000000000000000000000000000000000000000000000000000000000000<br>SWAP1<br>DUP2<br>SWAP1<br>DIV<br>MUL<br>SWAP2<br>POP<br>PUSH32 0x3000000000000000000000000000000000000000000000000000000000000000<br>PUSH1 0x01<br>PUSH1 0xf8<br>PUSH1 0x02<br>EXP<br>SUB<br>NOT<br>DUP4<br>AND<br>LT<br>DUP1<br>ISZERO<br>SWAP1<br>PUSH2 0x030d<br>JUMPI<br>POP<br>PUSH32 0x3500000000000000000000000000000000000000000000000000000000000000<br>PUSH1 0x01<br>PUSH1 0xf8<br>PUSH1 0x02<br>EXP<br>SUB<br>NOT<br>DUP4<br>AND<br>GT<br>ISZERO<br>JUMPDEST<br>DUP1<br>PUSH2 0x0377<br>JUMPI<br>POP<br>PUSH32 0x6100000000000000000000000000000000000000000000000000000000000000<br>PUSH1 0x01<br>PUSH1 0xf8<br>PUSH1 0x02<br>EXP<br>SUB<br>NOT<br>DUP4<br>AND<br>LT<br>DUP1<br>ISZERO<br>SWAP1<br>PUSH2 0x0377<br>JUMPI<br>POP<br>PUSH32 0x7a00000000000000000000000000000000000000000000000000000000000000<br>PUSH1 0x01<br>PUSH1 0xf8<br>PUSH1 0x02<br>EXP<br>SUB<br>NOT<br>DUP4<br>AND<br>GT<br>ISZERO<br>JUMPDEST<br>ISZERO<br>ISZERO<br>PUSH2 0x03e4<br>JUMPI<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH32 0x08c379a000000000000000000000000000000000000000000000000000000000<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>PUSH1 0x04<br>DUP3<br>ADD<br>MSTORE<br>PUSH1 0x0c<br>PUSH1 0x24<br>DUP3<br>ADD<br>MSTORE<br>PUSH32 0x77726f6e672073796d626f6c0000000000000000000000000000000000000000<br>PUSH1 0x44<br>DUP3<br>ADD<br>MSTORE<br>SWAP1<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x64<br>ADD<br>SWAP1<br>REVERT<br>JUMPDEST<br>PUSH1 0x01<br>SWAP1<br>SWAP3<br>ADD<br>SWAP2<br>PUSH2 0x0265<br>JUMP<br>JUMPDEST<br>POP<br>PUSH1 0x20<br>DUP4<br>DUP2<br>ADD<br>MLOAD<br>CALLER<br>PUSH1 0x00<br>DUP2<br>DUP2<br>MSTORE<br>DUP1<br>DUP5<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>DUP2<br>SWAP1<br>SHA3<br>DUP4<br>SWAP1<br>SSTORE<br>DUP1<br>MLOAD<br>SWAP4<br>DUP5<br>ADD<br>DUP4<br>SWAP1<br>MSTORE<br>DUP1<br>DUP5<br>MSTORE<br>DUP4<br>ADD<br>DUP8<br>SWAP1<br>MSTORE<br>SWAP1<br>SWAP2<br>PUSH32 0xe308632be6ce11ab1f60ecf5ca874a0652c3dcca9900b7bc641087bebda56e3a<br>SWAP1<br>DUP9<br>SWAP1<br>DUP9<br>SWAP1<br>DUP6<br>SWAP1<br>DUP1<br>PUSH1 0x60<br>DUP2<br>ADD<br>DUP6<br>DUP6<br>DUP1<br>DUP3<br>DUP5<br>CALLDATACOPY<br>PUSH1 0x40<br>MLOAD<br>SWAP3<br>ADD<br>DUP3<br>SWAP1<br>SUB<br>SWAP7<br>POP<br>SWAP1<br>SWAP5<br>POP<br>POP<br>POP<br>POP<br>POP<br>LOG2<br>POP<br>POP<br>POP<br>POP<br>POP<br>POP<br>JUMP<br>JUMPDEST<br>PUSH1 0x60<br>PUSH2 0x0472<br>CALLER<br>PUSH2 0x0477<br>JUMP<br>JUMPDEST<br>SWAP1<br>POP<br>SWAP1<br>JUMP<br>JUMPDEST<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>DUP2<br>AND<br>PUSH1 0x00<br>SWAP1<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>DUP2<br>SWAP1<br>MSTORE<br>PUSH1 0x40<br>SWAP1<br>SHA3<br>SLOAD<br>PUSH1 0x60<br>SWAP1<br>DUP1<br>ISZERO<br>ISZERO<br>PUSH2 0x04ac<br>JUMPI<br>PUSH2 0x04c4<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>DUP1<br>DUP3<br>ADD<br>SWAP1<br>SWAP2<br>MSTORE<br>PUSH1 0x0c<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>DUP2<br>ADD<br>DUP3<br>SWAP1<br>MSTORE<br>SWAP2<br>POP<br>JUMPDEST<br>POP<br>SWAP2<br>SWAP1<br>POP<br>JUMP<br>STOP<br>