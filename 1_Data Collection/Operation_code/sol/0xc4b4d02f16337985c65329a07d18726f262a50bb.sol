PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x004b<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x4555d5c9<br>DUP2<br>EQ<br>PUSH2 0x0088<br>JUMPI<br>DUP1<br>PUSH4 0x5c60da1b<br>EQ<br>PUSH2 0x00af<br>JUMPI<br>JUMPDEST<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>PUSH1 0x00<br>SLOAD<br>AND<br>CALLDATASIZE<br>PUSH1 0x00<br>DUP1<br>CALLDATACOPY<br>PUSH1 0x00<br>DUP1<br>CALLDATASIZE<br>PUSH1 0x00<br>DUP5<br>GAS<br>DELEGATECALL<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>DUP1<br>ISZERO<br>ISZERO<br>PUSH2 0x0083<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>RETURNDATASIZE<br>PUSH1 0x00<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0094<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x009d<br>PUSH2 0x00ed<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>SWAP2<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x00bb<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00c4<br>PUSH2 0x00f2<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>SWAP1<br>SWAP3<br>AND<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>PUSH1 0x02<br>SWAP1<br>JUMP<br>JUMPDEST<br>PUSH1 0x00<br>SLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>AND<br>SWAP1<br>JUMP<br>STOP<br>LOG1<br>PUSH6 0x627a7a723058<br>SHA3<br>STOP<br>'e8'(Unknown Opcode)<br>SWAP7<br>GASPRICE<br>MSTORE8<br>CODECOPY<br>PUSH11 0x420e260d6762a2b264be1d<br>'b0'(Unknown Opcode)<br>SWAP12<br>DUP10<br>'fc'(Unknown Opcode)<br>'e9'(Unknown Opcode)<br>'24'(Unknown Opcode)<br>