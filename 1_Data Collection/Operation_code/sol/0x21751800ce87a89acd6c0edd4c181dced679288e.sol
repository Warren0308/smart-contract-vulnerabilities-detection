PUSH1 0x80<br>PUSH1 0x40<br>MSTORE<br>PUSH1 0x04<br>CALLDATASIZE<br>LT<br>PUSH2 0x004b<br>JUMPI<br>PUSH4 0xffffffff<br>PUSH29 0x0100000000000000000000000000000000000000000000000000000000<br>PUSH1 0x00<br>CALLDATALOAD<br>DIV<br>AND<br>PUSH4 0x2423081e<br>DUP2<br>EQ<br>PUSH2 0x00df<br>JUMPI<br>DUP1<br>PUSH4 0x8fe123d7<br>EQ<br>PUSH2 0x011d<br>JUMPI<br>JUMPDEST<br>CALLVALUE<br>ISZERO<br>PUSH2 0x00dd<br>JUMPI<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH32 0x08c379a000000000000000000000000000000000000000000000000000000000<br>DUP2<br>MSTORE<br>PUSH1 0x20<br>PUSH1 0x04<br>DUP3<br>ADD<br>MSTORE<br>PUSH1 0x24<br>DUP1<br>DUP3<br>ADD<br>MSTORE<br>PUSH32 0x5468697320636f6e747261637420646f657374206e6f74206163636570742065<br>PUSH1 0x44<br>DUP3<br>ADD<br>MSTORE<br>PUSH32 0x7468657200000000000000000000000000000000000000000000000000000000<br>PUSH1 0x64<br>DUP3<br>ADD<br>MSTORE<br>SWAP1<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x84<br>ADD<br>SWAP1<br>REVERT<br>JUMPDEST<br>STOP<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x00eb<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00f4<br>PUSH2 0x014b<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>SWAP1<br>SWAP3<br>AND<br>DUP3<br>MSTORE<br>MLOAD<br>SWAP1<br>DUP2<br>SWAP1<br>SUB<br>PUSH1 0x20<br>ADD<br>SWAP1<br>RETURN<br>JUMPDEST<br>CALLVALUE<br>DUP1<br>ISZERO<br>PUSH2 0x0129<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>PUSH2 0x00dd<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>PUSH1 0x04<br>CALLDATALOAD<br>AND<br>PUSH2 0x0163<br>JUMP<br>JUMPDEST<br>PUSH20 0x2628e13a3cbdc52ed96b4b8d6b1041d3ef3a409e<br>DUP2<br>JUMP<br>JUMPDEST<br>PUSH1 0x40<br>DUP1<br>MLOAD<br>PUSH32 0x8fe123d700000000000000000000000000000000000000000000000000000000<br>DUP2<br>MSTORE<br>PUSH20 0xffffffffffffffffffffffffffffffffffffffff<br>DUP4<br>AND<br>PUSH1 0x04<br>DUP3<br>ADD<br>MSTORE<br>SWAP1<br>MLOAD<br>PUSH20 0x2628e13a3cbdc52ed96b4b8d6b1041d3ef3a409e<br>SWAP2<br>PUSH4 0x8fe123d7<br>SWAP2<br>PUSH1 0x24<br>DUP1<br>DUP4<br>ADD<br>SWAP3<br>PUSH1 0x00<br>SWAP3<br>SWAP2<br>SWAP1<br>DUP3<br>SWAP1<br>SUB<br>ADD<br>DUP2<br>DUP4<br>DUP8<br>DUP1<br>EXTCODESIZE<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x01e4<br>JUMPI<br>PUSH1 0x00<br>DUP1<br>REVERT<br>JUMPDEST<br>POP<br>GAS<br>CALL<br>ISZERO<br>DUP1<br>ISZERO<br>PUSH2 0x01f8<br>JUMPI<br>RETURNDATASIZE<br>PUSH1 0x00<br>DUP1<br>RETURNDATACOPY<br>RETURNDATASIZE<br>PUSH1 0x00<br>REVERT<br>JUMPDEST<br>POP<br>POP<br>POP<br>POP<br>POP<br>JUMP<br>STOP<br>